//! The host<->firmware command protocol: ASCII, line-based, one command per line.
//!
//! Commands (case-insensitive keyword), terminated by `\n` (or `\r`):
//!
//! ```text
//!   ID?                        -> "<banner>\n"  (e.g. "hwtools-harness rp2350 v0")
//!   PING                       -> "PONG\n"
//!   EMIT UART <hex> <baud>     -> "OK\n"   stream one byte as 8N1 UART on GP0
//!   EMIT SQUARE <hz> <duty%>   -> "OK\n"   PWM square on GP1 at <hz>, <duty> percent
//!   EMIT SPI                   -> "OK\n"   fixed SPI transaction: clk GP2, mosi GP3, cs GP4
//!   EMIT I2C                   -> "OK\n"   fixed I2C transaction: scl GP2, sda GP3
//!   STOP                       -> "OK\n"   stop emission
//!   BOOTSEL                    -> "OK\n"   then reboot into the USB bootloader
//! ```
//!
//! SPI/I2C emit a *fixed, known* transaction (payloads below) so the host can
//! validate decode against ground truth without variable-length parsing.
//!
//! Parsing is pure and `core`-only, so it is unit-tested on the host.

/// A decoded command from the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Identify: reply with the firmware banner.
    Id,
    /// Liveness check: reply `PONG`.
    Ping,
    /// Stop any active emission.
    Stop,
    /// Reboot into the USB bootloader (for scripted reflashing).
    Bootsel,
    /// Continuously transmit `byte` as 8N1 UART at `baud`.
    EmitUart { byte: u8, baud: u32 },
    /// Output a PWM square wave at `freq_hz` with `duty_pct` percent duty.
    EmitSquare { freq_hz: u32, duty_pct: u32 },
    /// Emit the fixed SPI test transaction (clk GP2, mosi GP3, cs GP4).
    EmitSpi,
    /// Emit the fixed I2C test transaction (scl GP2, sda GP3).
    EmitI2c,
}

/// Fixed SPI payload (MSB-first, mode 0) the harness clocks out.
pub const SPI_TEST_BYTES: [u8; 2] = [0xA5, 0x3C];
/// Fixed I2C target address (7-bit) and payload the harness writes.
pub const I2C_TEST_ADDR: u8 = 0x50;
pub const I2C_TEST_DATA: [u8; 2] = [0xDE, 0xAD];

/// Why a line failed to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// The line was empty/whitespace only.
    Empty,
    /// The keyword was not recognised.
    Unknown,
    /// The command was recognised but its arguments were missing/invalid.
    BadArgs,
}

/// Parse one command line.
pub fn parse(line: &str) -> Result<Command, ParseError> {
    let line = line.trim();
    if line.is_empty() {
        return Err(ParseError::Empty);
    }
    let mut tokens = line.split_ascii_whitespace();
    let keyword = tokens.next().ok_or(ParseError::Empty)?;

    if keyword.eq_ignore_ascii_case("ID?") || keyword.eq_ignore_ascii_case("ID") {
        return Ok(Command::Id);
    }
    if keyword.eq_ignore_ascii_case("PING") {
        return Ok(Command::Ping);
    }
    if keyword.eq_ignore_ascii_case("STOP") {
        return Ok(Command::Stop);
    }
    if keyword.eq_ignore_ascii_case("BOOTSEL") {
        return Ok(Command::Bootsel);
    }
    if keyword.eq_ignore_ascii_case("EMIT") {
        let sub = tokens.next().ok_or(ParseError::BadArgs)?;
        if sub.eq_ignore_ascii_case("UART") {
            let byte_tok = tokens.next().ok_or(ParseError::BadArgs)?;
            let baud_tok = tokens.next().ok_or(ParseError::BadArgs)?;
            let byte = parse_hex_byte(byte_tok).ok_or(ParseError::BadArgs)?;
            let baud = baud_tok.parse::<u32>().map_err(|_| ParseError::BadArgs)?;
            if baud == 0 {
                return Err(ParseError::BadArgs);
            }
            return Ok(Command::EmitUart { byte, baud });
        }
        if sub.eq_ignore_ascii_case("SQUARE") {
            let freq_tok = tokens.next().ok_or(ParseError::BadArgs)?;
            let duty_tok = tokens.next().ok_or(ParseError::BadArgs)?;
            let freq_hz = freq_tok.parse::<u32>().map_err(|_| ParseError::BadArgs)?;
            let duty_pct = duty_tok.parse::<u32>().map_err(|_| ParseError::BadArgs)?;
            if freq_hz == 0 || duty_pct > 100 {
                return Err(ParseError::BadArgs);
            }
            return Ok(Command::EmitSquare { freq_hz, duty_pct });
        }
        if sub.eq_ignore_ascii_case("SPI") {
            return Ok(Command::EmitSpi);
        }
        if sub.eq_ignore_ascii_case("I2C") {
            return Ok(Command::EmitI2c);
        }
        return Err(ParseError::BadArgs);
    }
    Err(ParseError::Unknown)
}

fn parse_hex_byte(token: &str) -> Option<u8> {
    let digits = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))
        .unwrap_or(token);
    u8::from_str_radix(digits, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_control_keywords_case_insensitively() {
        assert_eq!(parse("ID?"), Ok(Command::Id));
        assert_eq!(parse("id"), Ok(Command::Id));
        assert_eq!(parse("PING"), Ok(Command::Ping));
        assert_eq!(parse("  ping \r"), Ok(Command::Ping));
        assert_eq!(parse("STOP"), Ok(Command::Stop));
        assert_eq!(parse("bootsel"), Ok(Command::Bootsel));
    }

    #[test]
    fn parses_emit_uart_with_optional_hex_prefix() {
        assert_eq!(parse("EMIT UART A5 9600"), Ok(Command::EmitUart { byte: 0xA5, baud: 9600 }));
        assert_eq!(parse("emit uart 0xff 115200"), Ok(Command::EmitUart { byte: 0xFF, baud: 115200 }));
    }

    #[test]
    fn parses_emit_square() {
        assert_eq!(parse("EMIT SQUARE 1000 50"), Ok(Command::EmitSquare { freq_hz: 1000, duty_pct: 50 }));
        assert_eq!(parse("emit square 250000 25"), Ok(Command::EmitSquare { freq_hz: 250000, duty_pct: 25 }));
    }

    #[test]
    fn parses_emit_spi_and_i2c() {
        assert_eq!(parse("EMIT SPI"), Ok(Command::EmitSpi));
        assert_eq!(parse("emit i2c"), Ok(Command::EmitI2c));
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(parse(""), Err(ParseError::Empty));
        assert_eq!(parse("   "), Err(ParseError::Empty));
        assert_eq!(parse("frobnicate"), Err(ParseError::Unknown));
        assert_eq!(parse("EMIT UART"), Err(ParseError::BadArgs));
        assert_eq!(parse("EMIT FROB 1 2"), Err(ParseError::BadArgs));
        assert_eq!(parse("EMIT UART ZZ 9600"), Err(ParseError::BadArgs));
        assert_eq!(parse("EMIT UART A5 0"), Err(ParseError::BadArgs));
        assert_eq!(parse("EMIT SQUARE 1000 150"), Err(ParseError::BadArgs));
        assert_eq!(parse("EMIT SQUARE 0 50"), Err(ParseError::BadArgs));
    }
}
