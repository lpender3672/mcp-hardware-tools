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

Current stimulus (see `common/src/lib.rs`): `0xA5`, 8N1 UART @ 9600 baud, emitted
on **GP0** via a PIO state machine.

## Build

Build a target from its own directory so its `.cargo/config.toml` (target triple)
applies:

```bash
cd firmware/targets/rp2350 && cargo build --release
```

## Flash (RP2350)

The firmware has no USB, so reflashing needs a manual BOOTSEL: hold **BOOTSEL** and
tap **RESET** (or hold BOOTSEL while replugging USB) — the `RP2350` drive mounts.
Then make a UF2 and copy it across:

```bash
ELF=firmware/target/thumbv8m.main-none-eabihf/release/hwtools-harness-rp2350
rust-objcopy -O binary "$ELF" harness.bin
# wrap harness.bin as UF2 with family id 0xe48bff59 (RP2350 ARM-S), base 0x10000000
cp harness.uf2 /e/        # the mounted BOOTSEL drive
```

## Validate

With CH1 on GP0 (10x, GND on pin 3), run the supervised round-trip test:

```bash
uv run pytest -m hardware -s
```
