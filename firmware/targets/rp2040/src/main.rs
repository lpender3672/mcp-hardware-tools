//! RP2040 (Raspberry Pi Pico) harness target.
//!
//! Same harness role as the RP2350 target, on the original Pico. Emits the shared
//! known stimulus as 8N1 UART on **GP0** via PIO. Only the HAL-specific wiring and
//! the RP2040 second-stage bootloader differ; the PIO program, byte, baud, and
//! divider math come from the common crate.
//!
//! Provided to prove the target/common split; not yet validated on RP2040 hardware.

#![no_std]
#![no_main]

use panic_halt as _;
use rp2040_hal as hal;

use hal::pio::{PIOExt, ShiftDirection};
use hal::Clock;
use hwtools_harness_common as harness;

/// 12 MHz crystal on the Pico.
const XOSC_CRYSTAL_FREQ: u32 = 12_000_000;

/// RP2040 second-stage bootloader (boots from a W25Q080-class flash).
#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

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
