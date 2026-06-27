//! RP2350 (Raspberry Pi Pico 2) harness target.
//!
//! Emits a known UART stimulus via PIO on **GP0** (CH1) and exposes a USB-CDC
//! command channel so the host can identify the device, ping it, and reboot it
//! into the USB bootloader for scripted reflashing. Subsequent chunks add EMIT
//! UART / STOP to control the stimulus from the host.
//!
//! USB is serviced entirely in the `USBCTRL_IRQ` interrupt, which shuttles bytes
//! between the CDC endpoints and two lock-free SPSC ring buffers (RX: ISR ->
//! main, TX: main -> ISR). The main loop drives the PIO emitter and parses
//! whole command lines out of the RX ring.
//!
//! HAL-specific wiring only; the PIO program, byte, baud, divider math, and the
//! command parser come from `hwtools_harness_common`.

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
use hwtools_harness_common::protocol::{parse, Command};

/// 12 MHz crystal on the Pico 2.
const XOSC_CRYSTAL_FREQ: u32 = 12_000_000;
/// Ring-buffer capacity for each direction.
const RING: usize = 256;
/// Identity banner returned for `ID?`.
const BANNER: &[u8] = b"hwtools-harness rp2350 v0\n";

/// Boot image header the RP2350 bootrom scans for.
#[link_section = ".start_block"]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

type Bus = hal::usb::UsbBus;

// USB objects are touched only by the ISR after init; the ring halves below are
// single-owner per context. Accessed via raw pointers to avoid `static_mut_refs`.
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

    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    // --- PIO UART-TX emitter on GP0 (scope CH1) -----------------------------
    let tx_pin = pins.gpio0.into_function::<hal::gpio::FunctionPio0>();
    let tx_id = tx_pin.id().num;
    let program = harness::uart_tx_program();
    let (mut pio, sm0, _, _, _) = pac.PIO0.split(&mut pac.RESETS);
    let installed = pio.install(&program).unwrap();
    let (int_div, frac_div) = harness::uart_clock_divider(clocks.system_clock.freq().to_Hz());
    let (mut sm, _rx, mut pio_tx) = hal::pio::PIOBuilder::from_installed_program(installed)
        .out_pins(tx_id, 1)
        .side_set_pin_base(tx_id)
        .clock_divisor_fixed_point(int_div, frac_div)
        .out_shift_direction(ShiftDirection::Right)
        .autopull(false)
        .build(sm0);
    sm.set_pindirs([(tx_id, hal::pio::PinDir::Output)]);
    let _sm = sm.start();

    // --- USB-CDC command channel --------------------------------------------
    let usb_bus = UsbBusAllocator::new(hal::usb::UsbBus::new(
        pac.USB,
        pac.USB_DPRAM,
        clocks.usb_clock,
        true,
        &mut pac.RESETS,
    ));
    // SAFETY: written once here before the USB ISR is unmasked.
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
    // SAFETY: written once before unmasking; thereafter the ISR owns these halves.
    unsafe {
        USB_SERIAL = Some(serial);
        USB_DEV = Some(dev);
        RX_PROD = Some(rx_prod);
        TX_CONS = Some(tx_cons);
        NVIC::unmask(hal::pac::Interrupt::USBCTRL_IRQ);
    }

    let mut line: heapless::Vec<u8, 128> = heapless::Vec::new();
    loop {
        // Keep emitting the default stimulus; ~2 ms gap leaves the line idle
        // between bytes and gives the command parser time to run.
        while !pio_tx.write(harness::STIMULUS_BYTE) {}
        cortex_m::asm::delay(300_000);

        while let Some(byte) = rx_cons.dequeue() {
            match byte {
                b'\n' | b'\r' => {
                    if !line.is_empty() {
                        process(&line, &mut tx_prod);
                        line.clear();
                    }
                }
                _ => {
                    if line.push(byte).is_err() {
                        line.clear(); // overlong line: drop and resync
                    }
                }
            }
        }
    }
}

/// Handle one complete command line, queueing the reply for the ISR to send.
fn process(line: &[u8], tx: &mut Producer<'static, u8>) {
    let text = core::str::from_utf8(line).unwrap_or("");
    match parse(text) {
        Ok(Command::Id) => reply(tx, BANNER),
        Ok(Command::Ping) => reply(tx, b"PONG\n"),
        Ok(Command::Bootsel) => {
            reply(tx, b"OK\n");
            cortex_m::asm::delay(2_000_000); // let the reply flush over USB
            hal::reboot::reboot(
                hal::reboot::RebootKind::BootSel {
                    picoboot_disabled: false,
                    msd_disabled: false,
                },
                hal::reboot::RebootArch::Normal,
            );
        }
        Ok(Command::Stop | Command::EmitUart { .. }) => reply(tx, b"ERR unimplemented\n"),
        Err(_) => reply(tx, b"ERR\n"),
    }
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
    // SAFETY: these globals are initialised before the interrupt is unmasked and
    // are only accessed here afterwards.
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
