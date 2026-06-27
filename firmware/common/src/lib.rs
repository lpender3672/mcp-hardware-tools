//! MCU-agnostic harness logic, shared by every target.
//!
//! Nothing here depends on a specific HAL or chip — the RP2040 and RP2350 both
//! have the same PIO block, so the UART-TX program and the configuration live
//! here, and each target crate only supplies the HAL-specific wiring (clocks,
//! pins, state-machine setup, boot block).
//!
//! `no_std` for the firmware; the host test harness (`cargo test`) builds with
//! `std` so the protocol parser can be unit-tested off-target.

#![cfg_attr(not(test), no_std)]

pub mod protocol;

use pio::Program;

/// Known stimulus emitted by the harness: byte value and UART baud.
pub const STIMULUS_BYTE: u32 = 0xA5;
pub const BAUD: u32 = 9600;

/// The PIO UART-TX program runs this many state-machine cycles per bit, so the
/// SM clock must be `CYCLES_PER_BIT * BAUD`.
pub const CYCLES_PER_BIT: u32 = 8;

/// Assembled 8N1 PIO UART-TX program.
///
/// Idle/stop high (side-set), a low start bit, then 8 data bits shifted out
/// LSB-first; 8 SM cycles per bit. Pin mapping (OUT + side-set) is bound when the
/// state machine is built, so this program is fully target- and pin-agnostic.
pub fn uart_tx_program() -> Program<32> {
    pio::pio_asm!(
        ".side_set 1 opt",
        "    pull   side 1 [7]",
        "    set x, 7  side 0 [7]",
        "bitloop:",
        "    out pins, 1",
        "    jmp x-- bitloop [6]",
    )
    .program
}

/// Fixed-point PIO clock divider `(int, frac)` for the default [`BAUD`].
pub fn uart_clock_divider(sys_hz: u32) -> (u16, u8) {
    uart_clock_divider_for(sys_hz, BAUD)
}

/// Fixed-point PIO clock divider `(int, frac)` to reach `CYCLES_PER_BIT * baud`
/// from a state-machine source clock of `sys_hz`.
pub fn uart_clock_divider_for(sys_hz: u32, baud: u32) -> (u16, u8) {
    let denom = CYCLES_PER_BIT * baud;
    let int = (sys_hz / denom) as u16;
    let frac = (((sys_hz % denom) * 256) / denom) as u8;
    (int, frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divider_matches_known_baud_at_150mhz() {
        assert_eq!(uart_clock_divider_for(150_000_000, 9600), (1953, 32));
        let (int, frac) = uart_clock_divider_for(150_000_000, 115_200);
        assert_eq!(int, 162);
        assert!((193..=195).contains(&frac));
    }
}
