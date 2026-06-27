//! RP2350 (Raspberry Pi Pico 2) harness target.
//!
//! Emits the shared known stimulus (see `hwtools_harness_common`) as 8N1 UART on
//! **GP0** (physical pin 1), where scope CH1 is wired (GND to pin 3). The RP2350
//! USB is on dedicated pins, so GP0/GP1 are free for harness signals.
//!
//! Only the HAL-specific wiring lives here; the PIO program, byte, baud, and
//! divider math come from the common crate.

#![no_std]
#![no_main]

use panic_halt as _;
use rp235x_hal as hal;

use hal::pio::{PIOExt, ShiftDirection};
use hal::Clock;
use hwtools_harness_common as harness;

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

    // GP0 driven by PIO0 as the UART TX line (scope CH1).
    let tx_pin = pins.gpio0.into_function::<hal::gpio::FunctionPio0>();
    let tx_id = tx_pin.id().num;

    let program = harness::uart_tx_program();
    let (mut pio, sm0, _, _, _) = pac.PIO0.split(&mut pac.RESETS);
    let installed = pio.install(&program).unwrap();

    let (int_div, frac_div) = harness::uart_clock_divider(clocks.system_clock.freq().to_Hz());
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
        while !tx.write(harness::STIMULUS_BYTE) {}
        cortex_m::asm::delay(300_000);
    }
}
