//! RP2350 harness firmware — fixed-emit vertical slice.
//!
//! Continuously transmits a known byte as 8N1 UART using a PIO state machine, so
//! the scope tooling + decoders can be validated against ground truth. Emission
//! is on **GP2** (physical pin 4); wire scope CH1 there, GND to pin 3.
//!
//! This is the harness's first incarnation: a free-running fixed stimulus, no
//! host control. A USB-CDC command contract grows here later.

#![no_std]
#![no_main]

use panic_halt as _;
use rp235x_hal as hal;

use hal::pio::{PIOExt, ShiftDirection};
use hal::Clock;

/// Known stimulus: byte value and UART baud.
const STIMULUS_BYTE: u32 = 0xA5;
const BAUD: u32 = 9600;
/// PIO program runs 8 cycles per bit, so the state machine clock is 8 * BAUD.
const CYCLES_PER_BIT: u32 = 8;

/// 12 MHz crystal on the Pico 2.
const XOSC_CRYSTAL_FREQ: u32 = 12_000_000;

/// Boot image header the RP2350 bootrom scans for.
#[link_section = ".start_block"]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

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

    // GP2 driven by PIO0 as the UART TX line.
    let tx_pin = pins.gpio2.into_function::<hal::gpio::FunctionPio0>();
    let tx_id = tx_pin.id().num;

    // 8n1 UART TX PIO program: idle/stop high (side-set), start bit low, then
    // shift 8 data bits LSB-first, 8 SM cycles per bit.
    let program = pio::pio_asm!(
        ".side_set 1 opt",
        "    pull   side 1 [7]",
        "    set x, 7  side 0 [7]",
        "bitloop:",
        "    out pins, 1",
        "    jmp x-- bitloop [6]",
    );

    let (mut pio, sm0, _, _, _) = pac.PIO0.split(&mut pac.RESETS);
    let installed = pio.install(&program.program).unwrap();

    // SM clock = 8 * BAUD via a fixed-point divider off the system clock.
    let sys_hz = clocks.system_clock.freq().to_Hz();
    let denom = CYCLES_PER_BIT * BAUD;
    let int_div = (sys_hz / denom) as u16;
    let frac_div = (((sys_hz % denom) * 256) / denom) as u8;

    let (mut sm, _rx, mut tx) = hal::pio::PIOBuilder::from_installed_program(installed)
        .out_pins(tx_id, 1)
        .side_set_pin_base(tx_id)
        .clock_divisor_fixed_point(int_div, frac_div)
        .out_shift_direction(ShiftDirection::Right)
        .autopull(false)
        .build(sm0);
    sm.set_pindirs([(tx_id, hal::pio::PinDir::Output)]);
    let _sm = sm.start();

    loop {
        // Block until the FIFO accepts the byte, then idle briefly so frames are
        // cleanly separated (line returns to mark between bytes).
        while !tx.write(STIMULUS_BYTE) {}
        cortex_m::asm::delay(300_000);
    }
}
