//! Sample-pattern generators for the PIO "pin player".
//!
//! The harness drives all protocols through one tiny PIO program that clocks
//! 3-bit samples out of its FIFO at [`SAMPLE_RATE_HZ`] onto the shared pins
//! (GP2/GP3/GP4). The protocol *timing* lives here, in pure code: each generator
//! returns the exact sequence of pin states for one repetition. The player loops
//! the buffer, so transactions (UART/SPI/I2C) include an idle gap for the scope
//! to trigger on.
//!
//! Pin bits: A = GP2 (UART tx / square / SPI clk / I2C scl), B = GP3 (SPI mosi /
//! I2C sda), CS = GP4 (SPI chip-select, active low).

use heapless::Vec;

use crate::protocol::{I2C_TEST_ADDR, I2C_TEST_DATA, SPI_TEST_BYTES};

/// PIO sample clock — every buffer entry is emitted for 1/SAMPLE_RATE_HZ seconds.
pub const SAMPLE_RATE_HZ: u32 = 250_000;

pub const A: u8 = 0b001; // GP2
pub const B: u8 = 0b010; // GP3
pub const CS: u8 = 0b100; // GP4

/// Max samples in one repetition (≈8 ms at 250 kHz).
pub const MAX_SAMPLES: usize = 2048;
pub type Samples = Vec<u8, MAX_SAMPLES>;

/// The PIO "pin player": clock 3 pins from each FIFO word, looping forever.
///
/// Each sample occupies a 4-bit nibble (3 pin bits + 1 discarded pad) so that 8
/// samples fill exactly one 32-bit word — avoiding the leftover bits a 3-bit
/// pack would leave under the default 32-bit autopull.
pub fn player_program() -> pio::Program<32> {
    pio::pio_asm!(
        ".wrap_target",
        "    out pins, 3",
        "    out null, 1",
        ".wrap",
    )
    .program
}

fn run(buf: &mut Samples, level: u8, count: usize) {
    for _ in 0..count {
        let _ = buf.push(level);
    }
}

/// One period of a square wave on line B (GP3 / scope CH2) at `freq_hz`,
/// `duty_pct` duty — the analog ground-truth source for the recommender tests.
pub fn square(freq_hz: u32, duty_pct: u32) -> Samples {
    let period = (SAMPLE_RATE_HZ / freq_hz.max(1)).clamp(2, MAX_SAMPLES as u32) as usize;
    let high = ((period as u32 * duty_pct.min(100)) / 100) as usize;
    let mut buf = Samples::new();
    run(&mut buf, B, high);
    run(&mut buf, 0, period - high);
    buf
}

/// One 8N1 UART frame of `byte` on line A at `baud`, framed by idle-high gaps.
pub fn uart(byte: u8, baud: u32) -> Samples {
    let spb = (SAMPLE_RATE_HZ / baud.max(1)).clamp(1, 256) as usize;
    let mut buf = Samples::new();
    run(&mut buf, A, 2 * spb); // idle (mark)
    run(&mut buf, 0, spb); // start bit
    for b in 0..8 {
        run(&mut buf, if (byte >> b) & 1 == 1 { A } else { 0 }, spb); // LSB first
    }
    run(&mut buf, A, 3 * spb); // stop + gap
    buf
}

/// The fixed SPI transaction (mode 0, MSB first): clk A, mosi B, cs CS (active low).
pub fn spi() -> Samples {
    const HB: usize = 10; // half-bit samples -> ~12.5 kHz clk
    let mut buf = Samples::new();
    run(&mut buf, CS, 4 * HB); // idle: cs high, clk low
    for &byte in &SPI_TEST_BYTES {
        for bit in (0..8).rev() {
            let mosi = if (byte >> bit) & 1 == 1 { B } else { 0 };
            run(&mut buf, mosi, HB); // cs low, clk low, mosi set
            run(&mut buf, A | mosi, HB); // cs low, clk high (sampled)
        }
    }
    run(&mut buf, 0, HB); // cs still low, clk low
    run(&mut buf, CS, 4 * HB); // cs high (idle)
    buf
}

/// The fixed I2C write: scl A, sda B. START, addr+W, data (self-ACKed), STOP.
///
/// SDA only ever transitions in the *middle* of an SCL-low phase (held for `Q`
/// either side), so that even at coarse capture resolution an SCL edge and an
/// SDA edge never appear simultaneous — which would otherwise look like a
/// spurious START/STOP to the decoder.
pub fn i2c() -> Samples {
    const Q: usize = 5;
    let mut buf = Samples::new();
    let both = A | B;

    run(&mut buf, both, 8 * Q); // idle: both high
    run(&mut buf, both, 2 * Q); // setup
    run(&mut buf, A, 2 * Q); // START: sda falls while scl high
    let mut sda = false; // low after START

    i2c_byte(&mut buf, Q, (I2C_TEST_ADDR << 1) | 0, &mut sda); // address + write
    for &b in &I2C_TEST_DATA {
        i2c_byte(&mut buf, Q, b, &mut sda);
    }

    // STOP: with sda low, raise scl, then raise sda while scl is high.
    run(&mut buf, if sda { B } else { 0 }, Q);
    run(&mut buf, 0, Q); // ensure sda low (scl still low)
    run(&mut buf, A, 2 * Q); // scl high, sda low
    run(&mut buf, both, 2 * Q); // STOP: sda rises while scl high
    run(&mut buf, both, 8 * Q);
    buf
}

/// 8 data bits (MSB first) then a self-ACK bit (sda low on the 9th clock).
fn i2c_byte(buf: &mut Samples, q: usize, byte: u8, sda: &mut bool) {
    for bit in (0..8).rev() {
        i2c_bit(buf, q, (byte >> bit) & 1 == 1, sda);
    }
    i2c_bit(buf, q, false, sda); // ACK: target pulls SDA low
}

fn i2c_bit(buf: &mut Samples, q: usize, level: bool, sda: &mut bool) {
    let old = if *sda { B } else { 0 };
    let new = if level { B } else { 0 };
    run(buf, old, q); // scl low: hold previous sda
    run(buf, new, q); // scl low: switch sda mid-phase
    run(buf, A | new, 2 * q); // scl high: data valid and stable
    run(buf, new, q); // scl low
    *sda = level;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transitions(buf: &Samples, mask: u8) -> usize {
        buf.windows(2).filter(|w| (w[0] ^ w[1]) & mask != 0).count()
    }

    #[test]
    fn square_period_and_duty() {
        let s = square(1_000, 25); // 250 samples/period
        assert_eq!(s.len(), 250);
        assert_eq!(s.iter().filter(|&&v| v == B).count(), 62); // ~25% on GP3
    }

    #[test]
    fn uart_frame_has_start_and_stop() {
        let s = uart(0xA5, 9600);
        assert_eq!(*s.first().unwrap(), A); // idle high
        assert_eq!(*s.last().unwrap(), A); // stop/idle high
    }

    #[test]
    fn spi_cs_frames_one_transaction() {
        // CS goes low once and high once around the byte stream.
        assert_eq!(transitions(&spi(), CS), 2);
    }

    #[test]
    fn i2c_has_a_single_start_and_stop_on_sda_while_scl_high() {
        // Two SDA edges occur while SCL is high: the START and the STOP.
        let s = i2c();
        let edges = s
            .windows(2)
            .filter(|w| (w[0] ^ w[1]) & B != 0 && (w[0] & A != 0) && (w[1] & A != 0))
            .count();
        assert_eq!(edges, 2);
    }
}
