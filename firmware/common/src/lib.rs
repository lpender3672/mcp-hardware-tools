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

pub mod pattern;
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

/// Hardware-PWM register values for a square wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquarePwm {
    /// Integer clock divider for the PWM slice.
    pub div_int: u8,
    /// TOP (wrap) register = period in counts minus one.
    pub top: u16,
    /// Compare level (counts the output is high) = duty * (top + 1).
    pub compare: u16,
}

/// Compute PWM registers for `freq_hz` at `duty_pct` from a `sys_hz` clock.
///
/// Picks the smallest integer divider that keeps the period within the 16-bit
/// TOP register, maximising duty resolution. The realised frequency is
/// `sys_hz / (div_int * (top + 1))` — close to requested, and the test harness
/// measures the actual value rather than assuming it.
pub fn square_pwm_params(sys_hz: u32, freq_hz: u32, duty_pct: u32) -> SquarePwm {
    let freq = freq_hz.max(1);
    let duty = duty_pct.min(100);
    let total = (sys_hz / freq).clamp(2, 255 * 65_536);
    let div = total.div_ceil(65_536).clamp(1, 255);
    let period = (total / div).clamp(2, 65_536);
    let top = (period - 1) as u16;
    let compare = ((period * duty) / 100).min(period - 1) as u16;
    SquarePwm { div_int: div as u8, top, compare }
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

    #[test]
    fn square_pwm_1khz_50pct_at_150mhz() {
        let p = square_pwm_params(150_000_000, 1_000, 50);
        let realised = 150_000_000.0 / (p.div_int as f64 * (p.top as f64 + 1.0));
        assert!((realised - 1_000.0).abs() / 1_000.0 < 0.01); // within 1%
        let duty = p.compare as f64 / (p.top as f64 + 1.0);
        assert!((duty - 0.5).abs() < 0.01);
    }

    #[test]
    fn square_pwm_handles_low_and_high_frequencies() {
        // 100 Hz needs a divider (period > 16 bits at full clock).
        let low = square_pwm_params(150_000_000, 100, 50);
        assert!(low.div_int > 1);
        // 1 MHz fits with div 1.
        let high = square_pwm_params(150_000_000, 1_000_000, 25);
        assert_eq!(high.div_int, 1);
        let duty = high.compare as f64 / (high.top as f64 + 1.0);
        assert!((duty - 0.25).abs() < 0.02);
    }
}
