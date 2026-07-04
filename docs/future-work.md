# Future work — thread roadmap

> How to use this doc: each **Thread** below is a self-contained unit of work a
> future session can pick up on its own. Each records *what*, *why*, the design
> decisions already made this session (so they aren't re-litigated), the open forks
> that still need a human call, and where it plugs into the infrastructure that now
> exists. Threads are roughly ordered by priority, not dependency.

## Where the foundation landed

The infrastructure pass is **done and bench-validated** (see
[`signal-infrastructure-plan.md`](signal-infrastructure-plan.md) for the design and
[`hil-validation-plan.md`](hil-validation-plan.md) for the HIL log). What exists now:

- **Addressable capture store** (`session/store.py`) — frames by handle, byte-budget
  eviction, ephemeral/kept tiers, never-reused ids. One frame, many lenses, no
  re-capture (mandatory for one-shots).
- **One honest acquire primitive** (`session/acquire.py`) — sweep is a parameter, no
  forced trigger, honest misses, stop-before-reconfigure.
- **The dense `describe` lens** (`analysis/describe.py`) — always-on feature vector +
  opt-in value-histogram / time-bins / Welch PSD / joint time-vs-value grid. This is
  the **shared measurement backbone** for the threads below.
- **`classify`/`triage` skeleton** (`analysis/classify.py`) — rule-based, behind a
  stable signature.
- **`ScopeSession` facade + MCP surface** (`session/scope_session.py`, `tools/`).
- **DS1054Z driver hardened & manual-grounded** — value-grid snapping, range clamps,
  read semantics; every driver pathway has an on-scope test, values verified saved
  verbatim.

The layering is clean and debt is paid down, so these threads plug in at defined
seams rather than fighting the structure.

---

## Thread A — JDS6600 signal generator + noise-robustness suite  *(primary)*

**What/why.** The session's original goal: layer the JDS6600's 2048-point arbitrary
waveform on top of analog/digital signals to build a comprehensive noise-robustness
test suite. Protocol in [`docs/JT-JDS6600-Communication-protocol.pdf`].

**Driver status: DONE (standard-waveform surface).** `SignalGenerator` interface
(`interfaces/signal_generator.py`) + `JDS6600` driver (`drivers/joyit/jds6600.py`) +
`model/siggen.py` + `SimulatedSignalGenerator`, with unit (fake-serial), contract
(sim + real), per-driver HIL, and **cross-instrument first-light** tests — the last
closes the HIL plan's dropped "H1 — JDS6600 online". Bench-confirmed: it's a
JDS6600-15 on COM7; frequency is centi-Hz; amplitude reads as **true Vpp into the
scope's high-Z input** (no 50 Ω doubling → no load-factor correction needed). The
siggen's own read-backs are only self-consistency; its output is ground-truthed by
the scope via `describe` (see `tests/hardware/cross/`). **Still to do:** the noise
suite itself (below), and **arbitrary-waveform (`a`/`b`, 2048-pt) upload** — the
manual under-specifies the wire format, so it was deliberately deferred to a
bench-reverse-engineering pass rather than guessed.

**Design decisions already made (do not re-litigate):**

1. **Most noise-robustness belongs in *software*, not on the bench.** The decode
   pipeline separates cleanly: (a) noisy volts → `DigitalTrace` (thresholding/edges)
   is pure; (b) the scope faithfully digitising a noisy signal needs hardware; (c)
   decoder tolerance to bit-errors/jitter is pure again. (a)+(c) — the bulk — become
   synthetic-noise unit tests: deterministic, exhaustive, no bench, sweeping SNR.
2. **The JDS6600's unique value is (b): calibrating the sim scope's noise model and
   the threshold heuristic against the real front-end** — a small, anchored HIL set,
   not the primary suite.
3. **Single-channel extrapolation is two different claims — don't conflate them.**
   "One scope channel's noise behaviour extrapolates to CH2/3/4" is **safe** (same
   ADC/analog path). "Noise on one line extrapolates to the other *protocol* lines"
   is **not** — a noisy clock/CS injects spurious/missing sample edges (catastrophic)
   while a noisy data line gives single-bit flips (often recoverable). So keep
   per-*role* tests in software; only extrapolate *acquisition* noise across scope
   channels.
4. **The `describe` lens is the measurement backbone** — band thickness, per-time-bin
   percentile spread, and eye-opening ARE the SNR/noise metrics. The noise suite is a
   consumer of `describe`, not a parallel measurement stack.

**Open forks (need a human call before building the rig):**

- **Rig fork:** *summing junction* (Pico DUT signal + JDS6600 noise summed through a
  resistor/op-amp adder into one scope channel → real DUT in the loop) **vs**
  *arb-replays-everything* (encode signal+noise into the 2048-pt arb, Pico out of the
  loop → perfectly known ground truth but testing decode against a siggen reproduction,
  not a real acquisition). Both are valid; they prove different things.
- **Goal fork:** "prove the decoders are noise-robust" **vs** "prove the sim scope's
  noise model matches reality." Likely both, in some ratio — decide it explicitly.

**Concrete sub-steps** (roughly): `SignalGenerator` interface + `joyit/jds6600.py`
driver over USB-CDC; a `SignalGenerator` contract suite mirroring the scope one; a
supervised first-light; noise axes (AWGN by σ/SNR, band-limited, periodic
interference / mains hum, baseline drift, edge jitter, slew/ringing) as synthetic
generators; assert a **graceful BER-vs-SNR curve including the point where decode
correctly reports failure** rather than emitting silent garbage.

**Gap this surfaces:** the decoders carry per-frame `framing_error`/`parity_error`/
`ack` flags but **no "this decode is untrustworthy" confidence signal.** Noise
robustness needs that — see Thread D.

**Plugs into:** `describe` (metrics), the store (one noisy capture judged +
characterised + decoded off one frame), the sim scope's `noise_v` (already models
Gaussian noise for CI).

---

## Thread B — Unknown-signal triage maturation

**What/why.** The "additional goal" from the session: an unknown capture is triaged
by a classifier, then routed to the right decode/process path. The skeleton exists
(`analysis/classify.py`): rule-based `classify(features) → SignalClass` + `triage` +
`cross_channel_hint`, all behind a stable signature.

**Design decision made:** the classifier reads the *same* `describe` feature vector
the agent debugs with — one lens, multiple consumers. The rule-based body is a
deliberate skeleton behind a signature that can be swapped without touching callers.

**Future work:** replace the rules with a richer/learned classifier (more features,
or a trained model) behind the same signature; turn `triage`'s `suggested`
next-primitive into an actual routing flow (triage → recommend/decode); feed
`est_symbol_rate_hz` into decoder params (UART baud / SPI clk seed). The joint 2D
histogram is the densest classification input if the rules need more signal.

**Plugs into:** `describe` feature vector, `ScopeSession.triage`.

---

## Thread C — System identification / black-box characterization

**What/why.** The README's endgame: drive a stimulus, capture the response, and
characterise an unknown system from its output. The obvious early instrument is a
wideband **noise source** — inject flat noise, measure the response, pull out a
frequency response in one shot instead of sweeping a sine point-by-point.

**Backbone already built:** `describe`'s Welch PSD + `spectrum.py`. **Extension:**
cross-spectrum + coherence between an input frame and an output frame (two stored
captures, both live in the store at once via explicit ids), yielding a transfer
function; then a noise-source → frequency-response flow.

**Plugs into:** the addressable store (stimulus + response frames held simultaneously
— this is exactly why explicit non-`latest` ids exist), `spectrum.py` (add
cross-spectrum/coherence next to `welch_psd`).

---

## Thread D — Decode surface completeness

**What/why.** The decoders are unit-solid, but the *hardware* validation and the
harness only cover fixed, one-directional transactions.

- **SPI MISO is never exercised on hardware** (MOSI-only); **I²C is write-only** — no
  read, repeated-START, NAK, or clock-stretch, though the decoders handle them.
- The harness emits a single fixed payload per protocol (`SPI_TEST_BYTES`,
  `I2C_TEST_ADDR/DATA`). **Parametric harness transactions** (firmware) would let the
  HIL suite vary bytes/addresses/direction and exercise the untested decode paths on
  metal.
- **Decode confidence signal** (also needed by Thread A): a per-decode
  trustworthiness score, beyond the existing per-frame error flags — so a noisy or
  marginal decode is reported as such rather than as clean data. Ties to the
  [tool-design-philosophy] "never silently degrade."

**Plugs into:** the RP2350 firmware (`firmware/`), the decoders (`decode/`), the
harness driver (`SerialHarness`).

---

## Thread E — Dense-lens v2

`describe` covers the feature vector + value-histogram + time-bins + Welch PSD +
joint grid. Natural extensions when a consumer needs them:

- **Spectrogram** (STFT time×frequency grid) — the frequency analog of the joint
  histogram; catches chirps/bursts/drift.
- **Eye diagram** — fold a digital trace on the estimated symbol period; eye opening
  = margin. Needs the symbol rate first (post-triage), and directly serves Thread A's
  noise metrics.
- **Cross-spectrum / coherence** — shared with Thread C.

---

## Thread F — Deferred cleanup & open decisions

Small items intentionally left, recorded so they aren't forgotten:

- **Experimental loops.** `autoset` / `capture_until_usable` / `capture_single`
  (`session/loop.py`) are demoted to experimental convenience wrappers. Decide
  whether they stay (fast deterministic path + CI anchor) or retire in favour of the
  agent driving the primitives directly.
- **`suggest_adjustment` legacy path** (`analysis/adjust.py`) — the iterative
  nudge-toward-target heuristic, superseded by single-stage `recommend_setup`. Keep
  as a general fallback or retire.
- **Scale-snap fill divergence.** `judge` computes `fill_fraction` from the *requested*
  `config.scale`, but the driver snaps vertical scale to the 1-2-5 grid, so the
  reported fill can differ slightly from the actual on-screen fill. Options: derive
  fill from the waveform's saturation rails (the *actual* scale), or have
  `recommend` produce 1-2-5 scales so config == instrument. Clipping (the critical
  judgement) already uses the real rails, so this is minor.
- **One-shot deep-read default depth.** A default one-shot deep-reads AUTO memory,
  which the scope inflates to millions of points → the multi-chunk RAW read fires the
  benign "Stop point changed!" prompt per chunk (dozens of beeps, no harm). Left as-is
  by choice; a saner default cap (single-chunk unless the caller asks for more) would
  be quieter and lighter.
- **Compact store (v2).** The DS1054Z is 8-bit; storing raw ADC codes (uint8 + scale/
  offset) instead of float64 volts is ~8× smaller, materialising volts on demand. M3d
  already captures RAW, so the plumbing exists.
- **HIL acquire helper duplication.** `tests/hardware/_acquire.py`
  (`acquire_one_shot`/`acquire_repeating`) reimplements the arm/wait the product
  `session/acquire.py` now owns. The HIL tests could drive the real primitive instead
  of a parallel helper — less code, and the tests would exercise product code.
