"""RP2350 (Raspberry Pi Pico 2) harness.

The harness firmware is Rust, under the repo's ``firmware/`` directory, flashed as
a UF2. Host-side control (a DigitalDUT implementation over USB-CDC) will live here
once the firmware exposes a command contract; today the firmware free-runs a
fixed stimulus, so there is no host driver yet.
"""
