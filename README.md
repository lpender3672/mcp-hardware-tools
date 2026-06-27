# mcp-hardware-tools

An [MCP](https://modelcontextprotocol.io) server that gives Claude direct control of an oscilloscope for analog and digital hardware debugging: configure the instrument, capture and download waveforms, analyse them in software, and — crucially — recognise when the setup is wrong and correct it.

The tool *is* the scope. The interesting part is that Claude closes the loop on its own setup: it generates a capture, judges whether it's usable, notices a missed trigger or a clipped signal or a timebase that hides the edge of interest, adjusts, and re-captures. That convergence is the product, and it's meant to live in your daily MCP setup and save the hours otherwise spent nudging knobs by hand.

> **Status:** early / clean slate. This README describes the intended architecture, not a finished implementation. Interfaces will change.

## What it does

- Drives a **Rigol DS1054Z** over SCPI — channel coupling, vertical scale/offset, timebase, trigger, acquisition, and raw waveform download.
- **Analyses captured waveforms in software** — measurements and protocol decode (SPI, I²C, UART, and signals beyond the scope's built-in set), pulled into the loop so Claude reasons about results programmatically rather than reading them off-screen.
- **Observes and corrects its own setup** — recognises a missed trigger, a saturated channel, or a timebase that buries the feature of interest, then re-configures and re-captures until the capture is usable.

## The runtime loop (the product)

A fixed acquisition script assumes you already know the right trigger level, timebase, and vertical scale. On an unfamiliar signal you don't. The work is the convergence:

```
configure → arm → capture → judge (clipping? triggered? bandwidth ok?) → adjust → re-capture
```

That judgement is what an agent in the loop is good at. Every scope tool is therefore designed to be cheap to call repeatedly and to report back decision-ready state — clipping/saturation flags, trigger status, sample-rate-vs-signal-bandwidth sanity — so the next adjustment is informed rather than blind.

## The test harness (how the tooling gets built)

To develop and trust the scope tooling, you need signals with **known** properties to point it at. That's the only job of the signal generator and microcontroller here — they **emulate a DUT** so there's ground truth to validate against:

- **JDS6600** — produces analog stimulus with known waveform, frequency, amplitude, offset.
- **MCU** — emulates a digital device: clock out a known SPI transaction, act as an I²C target at a known address, stream known UART bytes.

You tell the harness to emit `0xA5` over SPI at 1 MHz, run the scope tooling, and check that capture + auto-scale + trigger + decode round-trips back to `0xA5`. This is how the decode and the self-correction get tested without hunting for real boards to probe. **The harness is development scaffolding, not part of the everyday tool surface.**

## Black-box testing and future instruments

Once the scope tooling is trustworthy, the same loop points at a real unknown system: drive a stimulus, capture the response, and characterise the system from its output without access to its internals. The harness that emulates DUTs during development becomes the stimulus source during testing.

The instrument set is meant to grow to support this, and each new instrument plugs in behind the same driver/tool split so the loop logic doesn't change. An obvious early addition is a good wideband **noise source** — injecting flat (white or pseudo-random) noise and measuring the response is a clean one-shot way to pull out a system's frequency response and do basic system identification, rather than sweeping a sine point by point.

## Architecture

```
                 ┌──────────────────────────┐
                 │      Claude (MCP host)     │
                 └─────────────┬──────────────┘
                               │ MCP (stdio / SSE)
                 ┌─────────────▼──────────────┐
                 │     mcp-hardware-tools      │
                 │                             │
                 │   PRODUCT          HARNESS  │
                 │  ┌────────┐      ┌────────┐ │
                 │  │ scope  │      │ siggen │ │
                 │  │ decode │      │  mcu   │ │
                 │  └───┬────┘      └───┬────┘ │
                 └──────┼───────────────┼──────┘
                        │               │
                  ┌─────▼────┐    ┌──────▼──────────┐
                  │ DS1054Z  │◄───┤  emulated DUT   │
                  │(USB/LAN) │    │ (siggen + MCU)  │
                  └──────────┘    └─────────────────┘
                     measures        produces known
                                       signals
```

Concerns kept separate:

1. **Scope driver** — thin, synchronous SCPI wrapper. No agent logic; "set this, read that."
2. **Decode** — pure functions over captured sample arrays (SPI/I²C/UART → frames). No instrument awareness, so it's unit-testable against synthetic waveforms with no hardware attached. Most of the correctness lives here.
3. **Tool surface** — the MCP tools Claude calls, composing driver + decode into single useful operations that return structured, decision-ready results.
4. **Harness drivers** — siggen + MCU, used during development to stand up known test scenarios. Exposed as dev/harness tools or driven from test fixtures; kept out of the product surface.

## Microcontroller (harness)

The harness MCU is a **Raspberry Pi Pico 2 (RP2350)**. Its job is emulating a digital DUT, where arbitrary protocol flexibility matters more than compute, and the Pico 2 is a good fit:

- **PIO state machines** bit-bang arbitrary SPI/I²C/UART — as initiator *or* target — plus non-standard signals, which is exactly the "produce a known stimulus the scope then has to make sense of" job.
- **Reflash is trivially scriptable** — `picotool load` over USB, or SWD for unattended runs — so the harness can stand up a fresh test scenario programmatically.
- Cheap enough to keep several on the bench.

Division of labour: the **Pico 2 emulates digital DUTs**, the **JDS6600 supplies analog stimulus**. The Pico 2 has ADC but no true DAC, so analog generation stays with the signal generator rather than the MCU.

Define the **firmware contract** — how the harness asks the Pico to emit a given stimulus — before writing firmware.

## Getting started

> Placeholder — pin down once the scope driver exists.

```bash
git clone <repo>
cd mcp-hardware-tools
# install deps (pyvisa + a VISA backend for the scope)
# configure instrument address
# register the server with your MCP host
```

The scope needs a VISA backend (`pyvisa-py` works for USB-TMC and LXI without NI-VISA). The JDS6600 enumerates as a USB CDC serial device.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for the full text.

Copyright 2026 [penderdesigns]
