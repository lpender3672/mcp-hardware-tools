//! RP2350 (Raspberry Pi Pico 2) harness target.
//!
//! All stimuli are emitted through one PIO "pin player": a tiny state machine
//! that clocks 3-bit samples out of its FIFO onto the shared pins GP2/GP3/GP4 at
//! a fixed rate. The protocol *timing* lives in `hwtools_harness_common::pattern`
//! (pure, host-tested); the main loop just streams the active pattern buffer and
//! regenerates it on each EMIT command. One pin set serves every protocol, so the
//! scope probes never move: GP2 = UART tx / square / SPI clk / I2C scl (CH1),
//! GP3 = SPI mosi / I2C sda (CH2), GP4 = SPI cs (CH3).
//!
//! A USB-CDC command channel (serviced in USBCTRL_IRQ via lock-free SPSC rings)
//! drives it; see `hwtools_harness_common::protocol`.

#![no_std]
#![no_main]

use core::ptr::addr_of_mut;

use panic_halt as _;
use rp235x_hal as hal;

use cortex_m::peripheral::NVIC;
use hal::pac::interrupt;
use hal::pio::{PIOExt, ShiftDirection};
use hal::Clock;
use heapless::spsc::{Consumer, Producer, Queue};
use usb_device::bus::UsbBusAllocator;
use usb_device::device::{StringDescriptors, UsbDeviceBuilder, UsbVidPid};
use usb_device::prelude::UsbDevice;
use usbd_serial::SerialPort;

use hwtools_harness_common as harness;
use hwtools_harness_common::pattern::{self, Samples};
use hwtools_harness_common::protocol::{parse, Command};

/// 12 MHz crystal on the Pico 2.
const XOSC_CRYSTAL_FREQ: u32 = 12_000_000;
/// Ring-buffer capacity for each direction.
const RING: usize = 256;
/// Identity banner returned for `ID?`.
const BANNER: &[u8] = b"hwtools-harness rp2350 v0\n";
/// GP2 is the lowest player pin; GP3, GP4 follow.
const PLAYER_PIN_BASE: u8 = 2;

/// Boot image header the RP2350 bootrom scans for.
#[link_section = ".start_block"]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

type Bus = hal::usb::UsbBus;

static mut USB_BUS: Option<UsbBusAllocator<Bus>> = None;
static mut USB_DEV: Option<UsbDevice<'static, Bus>> = None;
static mut USB_SERIAL: Option<SerialPort<'static, Bus>> = None;
static mut RX_Q: Queue<u8, RING> = Queue::new();
static mut TX_Q: Queue<u8, RING> = Queue::new();
static mut RX_PROD: Option<Producer<'static, u8>> = None; // ISR fills RX
static mut TX_CONS: Option<Consumer<'static, u8>> = None; // ISR drains TX

#[hal::entry]
fn main() -> ! {
    let mut pac = hal::pac::Peripherals::take().unwrap();
    let mut watchdog = hal::Watchdog::new(pac.WATCHDOG);

    let clocks = hal::clocks::init_clocks_and_plls(
        XOSC_CRYSTAL_FREQ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .ok()
    .unwrap();
    let sys_hz = clocks.system_clock.freq().to_Hz();

    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    // --- PIO pin player on GP2/GP3/GP4 --------------------------------------
    let _p2 = pins.gpio2.into_function::<hal::gpio::FunctionPio0>();
    let _p3 = pins.gpio3.into_function::<hal::gpio::FunctionPio0>();
    let _p4 = pins.gpio4.into_function::<hal::gpio::FunctionPio0>();

    // Clock 3 pins from each FIFO word (8 samples * 3 bits = 24 bits) at the
    // sample rate; the buffer's timing does the rest.
    let program = pattern::player_program();
    let (mut pio, sm0, _, _, _) = pac.PIO0.split(&mut pac.RESETS);
    let installed = pio.install(&program).unwrap();
    // The player is 2 instructions per sample (out pins + out null), so the SM
    // clock must be twice the sample rate.
    let divisor = (sys_hz / (2 * pattern::SAMPLE_RATE_HZ)) as u16;
    let (mut sm, _rx, mut player_tx) = hal::pio::PIOBuilder::from_installed_program(installed)
        .out_pins(PLAYER_PIN_BASE, 3)
        .clock_divisor_fixed_point(divisor, 0)
        .out_shift_direction(ShiftDirection::Right)
        .autopull(true)
        .build(sm0);
    sm.set_pindirs([
        (PLAYER_PIN_BASE, hal::pio::PinDir::Output),
        (PLAYER_PIN_BASE + 1, hal::pio::PinDir::Output),
        (PLAYER_PIN_BASE + 2, hal::pio::PinDir::Output),
    ]);
    let _sm = sm.start();

    // --- USB-CDC command channel --------------------------------------------
    let usb_bus = UsbBusAllocator::new(hal::usb::UsbBus::new(
        pac.USB,
        pac.USB_DPRAM,
        clocks.usb_clock,
        true,
        &mut pac.RESETS,
    ));
    unsafe { USB_BUS = Some(usb_bus) };
    let bus_ref: &'static UsbBusAllocator<Bus> = unsafe { (*addr_of_mut!(USB_BUS)).as_ref().unwrap() };

    let serial = SerialPort::new(bus_ref);
    let dev = UsbDeviceBuilder::new(bus_ref, UsbVidPid(0x16c0, 0x27dd))
        .strings(&[StringDescriptors::default()
            .manufacturer("hwtools")
            .product("harness-rp2350")
            .serial_number("0")])
        .unwrap()
        .device_class(usbd_serial::USB_CLASS_CDC)
        .build();

    let (rx_prod, mut rx_cons) = unsafe { (*addr_of_mut!(RX_Q)).split() };
    let (mut tx_prod, tx_cons) = unsafe { (*addr_of_mut!(TX_Q)).split() };
    unsafe {
        USB_SERIAL = Some(serial);
        USB_DEV = Some(dev);
        RX_PROD = Some(rx_prod);
        TX_CONS = Some(tx_cons);
        NVIC::unmask(hal::pac::Interrupt::USBCTRL_IRQ);
    }

    // --- emission state ------------------------------------------------------
    // Boots streaming the known UART byte on GP0... GP2 (CH1).
    let mut buffer: Samples = pattern::uart(harness::STIMULUS_BYTE as u8, 9600);
    let mut index: usize = 0;
    let mut pending: Option<u32> = None;
    let mut active = true;

    let mut line: heapless::Vec<u8, 128> = heapless::Vec::new();
    loop {
        if active {
            // Top up the player FIFO; a rejected word is held for next time.
            loop {
                let word = pending.take().unwrap_or_else(|| pack8(&buffer, &mut index));
                if !player_tx.write(word) {
                    pending = Some(word);
                    break;
                }
            }
        }

        while let Some(byte) = rx_cons.dequeue() {
            match byte {
                b'\n' | b'\r' => {
                    if !line.is_empty() {
                        let text = core::str::from_utf8(&line).unwrap_or("");
                        let mut load = |samples: Samples| {
                            buffer = samples;
                            index = 0;
                            pending = None;
                            active = true;
                        };
                        match parse(text) {
                            Ok(Command::Id) => reply(&mut tx_prod, BANNER),
                            Ok(Command::Ping) => reply(&mut tx_prod, b"PONG\n"),
                            Ok(Command::Stop) => {
                                active = false;
                                reply(&mut tx_prod, b"OK\n");
                            }
                            Ok(Command::EmitUart { byte, baud }) => {
                                load(pattern::uart(byte, baud));
                                reply(&mut tx_prod, b"OK\n");
                            }
                            Ok(Command::EmitSquare { freq_hz, duty_pct }) => {
                                load(pattern::square(freq_hz, duty_pct));
                                reply(&mut tx_prod, b"OK\n");
                            }
                            Ok(Command::EmitSpi) => {
                                load(pattern::spi());
                                reply(&mut tx_prod, b"OK\n");
                            }
                            Ok(Command::EmitI2c) => {
                                load(pattern::i2c());
                                reply(&mut tx_prod, b"OK\n");
                            }
                            Ok(Command::Bootsel) => {
                                reply(&mut tx_prod, b"OK\n");
                                cortex_m::asm::delay(2_000_000);
                                hal::reboot::reboot(
                                    hal::reboot::RebootKind::BootSel {
                                        picoboot_disabled: false,
                                        msd_disabled: false,
                                    },
                                    hal::reboot::RebootArch::Normal,
                                );
                            }
                            Err(_) => reply(&mut tx_prod, b"ERR\n"),
                        }
                        line.clear();
                    }
                }
                _ => {
                    if line.push(byte).is_err() {
                        line.clear();
                    }
                }
            }
        }
    }
}

/// Pack the next 8 samples (one 4-bit nibble each, LSB-first) into one FIFO word,
/// looping the buffer so the pattern repeats seamlessly. The player outputs 3 of
/// each nibble's bits and discards the 4th, so 8 samples fill one 32-bit word.
fn pack8(buffer: &Samples, index: &mut usize) -> u32 {
    let len = buffer.len().max(1);
    let mut word = 0u32;
    for i in 0..8 {
        word |= ((buffer[*index % len] & 0b111) as u32) << (i * 4);
        *index = (*index + 1) % len;
    }
    word
}

/// Queue bytes for transmission and nudge the USB ISR to flush them.
fn reply(tx: &mut Producer<'static, u8>, bytes: &[u8]) {
    for &b in bytes {
        if tx.enqueue(b).is_err() {
            break;
        }
    }
    NVIC::pend(hal::pac::Interrupt::USBCTRL_IRQ);
}

/// Service USB: poll the device, drain RX endpoint into the RX ring, and push
/// any queued TX bytes out the CDC port.
#[interrupt]
fn USBCTRL_IRQ() {
    let dev = unsafe { (*addr_of_mut!(USB_DEV)).as_mut().unwrap() };
    let serial = unsafe { (*addr_of_mut!(USB_SERIAL)).as_mut().unwrap() };
    let rx_prod = unsafe { (*addr_of_mut!(RX_PROD)).as_mut().unwrap() };
    let tx_cons = unsafe { (*addr_of_mut!(TX_CONS)).as_mut().unwrap() };

    if dev.poll(&mut [serial]) {
        let mut buf = [0u8; 64];
        if let Ok(n) = serial.read(&mut buf) {
            for &b in &buf[..n] {
                let _ = rx_prod.enqueue(b);
            }
        }
    }

    let mut out = [0u8; 64];
    let mut n = 0;
    while n < out.len() {
        match tx_cons.dequeue() {
            Some(b) => {
                out[n] = b;
                n += 1;
            }
            None => break,
        }
    }
    if n > 0 {
        let _ = serial.write(&out[..n]);
    }
}
