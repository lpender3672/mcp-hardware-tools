# Harness firmware

Rust firmware for the harness MCU — emits **known** digital stimuli so the scope
tooling and decoders can be validated against ground truth. Built as a Cargo
workspace so the same logic targets multiple chips.

```
firmware/
  common/            # MCU-agnostic: stimulus config, PIO UART-TX program, divider math
  targets/
    rp2350/          # Raspberry Pi Pico 2 (Cortex-M33, thumbv8m) — validated on hardware
    rp2040/          # Raspberry Pi Pico  (Cortex-M0+, thumbv6m) — compiles; not yet HW-tested
```

Each target crate owns the parts that genuinely differ per chip: the HAL
(`rp235x-hal` / `rp2040-hal`), `memory.x`, the boot block (RP2350 image header vs
RP2040 second-stage bootloader), and the target triple in `.cargo/config.toml`.
Everything else lives in `common`.

Default stimulus (see `common/src/lib.rs`): `0xA5`, 8N1 UART @ 9600 baud, emitted
on **GP0** via a PIO state machine. The host can retarget it at runtime over USB.

## Command channel (RP2350)

The firmware exposes a USB-CDC serial port serviced entirely in the `USBCTRL_IRQ`
interrupt, with lock-free SPSC ring buffers between the ISR and the main loop. It
speaks a line-based ASCII protocol (parser + grammar in `common/src/protocol.rs`):

```text
ID?                     -> hwtools-harness rp2350 v0
PING                    -> PONG
EMIT UART <hex> <baud>  -> OK     stream one byte as 8N1 UART on GP0
STOP                    -> OK     stop emission (line idles high)
BOOTSEL                 -> OK     reboot into the USB bootloader (scripted reflash)
```

Drive it from the host with `hwtools.drivers.rp2350.SerialHarness`.

## Build

Build a target from its own directory so its `.cargo/config.toml` (target triple)
applies:

```bash
cd firmware/targets/rp2350 && cargo build --release
```

## Flash (RP2350)

Make a UF2 from the ELF:

```bash
ELF=firmware/target/thumbv8m.main-none-eabihf/release/hwtools-harness-rp2350
rust-objcopy -O binary "$ELF" harness.bin
# wrap harness.bin as UF2 with family id 0xe48bff59 (RP2350 ARM-S), base 0x10000000
```

Enter the bootloader, then copy the UF2 to the mounted `RP2350` drive. Once the
command-channel firmware is running, the bootloader entry is scripted — no button:

```python
from hwtools.drivers.rp2350 import SerialHarness, find_pico_port
with SerialHarness(find_pico_port()) as h:
    h.reboot_to_bootloader()   # RP2350 drive mounts; copy the UF2
```

First flash (or recovery) still uses a manual BOOTSEL: hold **BOOTSEL** + tap
**RESET** (or hold BOOTSEL while replugging USB).

## Validate

With CH1 on GP0 (10x, GND on pin 3), run the supervised round-trip test:

```bash
uv run pytest -m hardware -s
```
