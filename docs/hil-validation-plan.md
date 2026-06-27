# HIL validation plan — proving the self-correcting brain on real hardware

> Status: planning. This addresses a **major gap**: the analysis layer (judge /
> recommend / autoset / loop) — i.e. the product — is validated **only against the
> simulated scope**, whose physics we authored. That is circular, and we already
> know the sim diverges from reality (the ADC clips at ~±5 divisions, not the ±4
> the model assumed). No committed `@pytest.mark.hardware` test exercises any
> recommender. The architecture is sound but **unproven on metal**.

## Operating philosophy: expect failure

We will write each HIL test to assert the *correct* behaviour, run it on the
bench, and **expect many to fail initially**. A failing HIL test is not a setback
— it is the closed feedback loop doing its job: pinpointing exactly where the
sim-derived model diverges from the instrument. The workflow per state is:

1. Write the HIL test asserting correct behaviour (red).
2. Run on the bench; record the real result.
3. If it diverges, **fix the model/heuristic to match reality**, log the
   divergence, re-run (green).
4. Commit the now-passing HIL test *and* the fix together.

A HIL test that passes on the first attempt is **suspicious** and must be
double-checked — it usually means the assertion is too loose.

Track every real-vs-sim divergence in a running "Known divergences" section at
the bottom of this doc as we find them.

## Why a better ground-truth source is needed first

The recommenders are fundamentally **analog**: vertical scale, offset, timebase,
trigger level. To assert them on real hardware we need stimuli with *precisely
known* amplitude, frequency, offset, and DC level. The current harness — the
Pico streaming `0xA5` UART — gives only a noisy ~5.5 Vpp bursty square with ringy
overshoot and a baud-derived edge rate. That is enough to *clip* and to exercise
the digital decode slice, but it is **not** a clean, known analog signal, and it
cannot reach several states at all (clean periodic frequency, DC/flat, fast
undersampled signals, arbitrary amplitude/offset).

Per the README, the **JDS6600 signal generator is exactly this ground-truth
source** (known sine/square/DC at known V/Hz/offset). Bringing it online is a
prerequisite for honest analog HIL validation. The Pico stays the *digital* DUT;
the JDS6600 becomes the *analog* stimulus for the recommender tests.

A lighter secondary option: extend the Pico firmware with `EMIT SQUARE <hz>`,
a fast-square mode, and a `EMIT DC` (GPIO held high ≈ 3.3 V) so a subset of
states can be reached without the siggen.

---

## Target HIL coverage matrix

Every reachable recommender state, the stimulus that reaches it, the assertion,
and the **suspected divergence** (why we expect it to fail first).

### `judge_capture` — [judge.py](../src/hwtools/analysis/judge.py)

| State | Stimulus | Assertion | Suspected divergence |
|---|---|---|---|
| clipping via saturation rails | any signal, small V/div | `clipping=True`; large V/div → `False` | low (probe already confirmed); commit it |
| low-fill note | small signal, large V/div | fill<0.1 noted | margin tuning under real noise |
| undersample note (`bandwidth_ok=False`) | **fast** square at slow timebase | `bandwidth_ok=False` | frequency estimate on a real fast edge |
| not-triggered note | NORMAL sweep, bad level | `triggered=False` | real `:TRIG:STAT?` semantics vs model |
| config-rail fallback (no saturation) | n/a on real scope (always has rails) | sim-only test acceptable | — |

### `recommend_setup` — [recommend.py](../src/hwtools/analysis/recommend.py)

| State | Stimulus | Assertion | Suspected divergence |
|---|---|---|---|
| R2 fit scale (unclipped, vpp>0) | siggen sine, known Vpp | scale → ~60% fill; recovers true Vpp | real ADC rail extent; noise |
| R2 offset centring | siggen sine with **DC offset** | trace centres; `offset_v=−mid` works | **offset SIGN convention on Rigol — highest risk** |
| R5 timebase from freq | siggen sine, known Hz | ~N periods on screen | frequency estimate on real/noisy signal |
| R8 trigger level → midline | siggen sine | level ≈ measured midline → triggers | midline estimate under noise |
| R1 clipped channel → open wide | huge siggen amplitude | scale jumps wide, then sizes next pass | rail detection at the wide scale |
| R3 flat/DC | `EMIT DC` or siggen DC | keep scale, centre on DC level | freq=None path; DC offset sign |
| R6 flat → keep timebase | DC source | timebase unchanged | — |
| R4/R7/R9 missing waveform/source | n/a on real (channels present) | sim-only acceptable | — |

### `autoset` — [loop.py](../src/hwtools/analysis/loop.py)

| State | Stimulus | Assertion | Suspected divergence |
|---|---|---|---|
| AS1 wide measure unclipped → 1 recommend → usable | siggen sine | `converged`, `widen_steps=0`, true Vpp recovered, not clipped | the headline test; offset sign, rails |
| AS2 measure clipped → widen | very large signal | `widen_steps≥1`, then converges | `wide_scale=5 V/div` assumption vs real range |
| AS3 widen maxed, still clipped | signal beyond probe range | terminates, `converged=False` honestly | edge case |

### `suggest_adjustment` / `capture_until_usable` — [adjust.py](../src/hwtools/analysis/adjust.py), [loop.py](../src/hwtools/analysis/loop.py)

| State | Stimulus | Assertion | Suspected divergence |
|---|---|---|---|
| A1 clipping → grow scale | small V/div on any signal | converges to unclipped | probe-confirmed; commit it |
| A2 low-fill → zoom in | small signal, large V/div | zooms to good fill | offset sign; noise |
| T1 not triggered → set level | NORMAL sweep, bad level | converges to triggered | **NORMAL-sweep capture of an untriggered frame** — can the driver even read the source midline? |
| B1 undersampled → timebase ÷2 | fast square, slow timebase | converges to well-sampled | never exercised even in sim |
| L1/L2/L3 converge / stuck / exhausted | constructed scenarios | correct `converged` flag | — |

> Note: `suggest_adjustment` is the **legacy iterative** path. Decide during
> execution whether to keep it (general fallback) or retire it in favour of the
> single-stage `recommend_setup`/`autoset`. If retained, its HIL coverage above
> still applies.

---

## Highest-risk assumptions (most likely to fail first)

1. **Offset sign.** `recommend_setup` sets `offset_v = −midline` to centre a
   channel. This is self-consistent in the sim because we defined both sides; the
   real Rigol may centre the *opposite* way, which would push a DC-offset signal
   *toward* a rail and cause clipping. Test explicitly: set a known offset on a
   DC-offset signal and confirm the captured trace moves the expected direction.
2. **NORMAL-sweep untriggered capture.** When the trigger never fires, what does
   `:WAV:DATA?` return — the last frame, a stale buffer, or nothing? The
   trigger-fix path (T1/R8) needs the source waveform to read its midline. If the
   buffer is stale/empty, the recommendation is computed from garbage. The likely
   fix: always *measure* with the sweep forced to AUTO, then apply the chosen
   sweep — which `autoset` already does, but `capture_until_usable` does not.
3. **ADC rail extent / fill margins under noise.** Already bit us once (±5 vs ±4).
   Real noise may trip the clipping margin or the low-fill threshold.
4. **Frequency estimation on real signals.** Mean-crossing frequency on a ringy
   or bursty real signal may be unstable, breaking R5 timebase recommendations.
   A clean siggen sine/square is the antidote.
5. **`wide_scale = 5 V/div`** measurement assumption — too wide for a mV siggen
   signal (poor measurement resolution), too narrow for a large one.

---

## Execution sequence (milestones)

- **H0 — HIL infrastructure.** A `tests/hardware/conftest.py` with bench addresses
  (scope IP, siggen port, pico port) and a parametrised "known signal" fixture.
  Commit the already-probe-validated **judge clipping** HIL test (low risk) to
  establish the pattern.
- **H1 — JDS6600 online.** `SignalGenerator` interface + `joyit/jds6600.py` driver
  over USB CDC (protocol in `docs/JT-JDS6600-Communication-protocol.pdf`), plus a
  supervised first-light (set 1 kHz 2 Vpp sine, capture, confirm). Add a
  SignalGenerator contract suite mirroring the scope one.
- **H2 — autoset headline HIL.** AS1 + R2 + R5 + R8 against a known siggen sine.
  Explicitly validate the **offset sign**. Expect failures; fix the model.
- **H3 — trigger + sweep.** T1 / R8 with NORMAL sweep; fix the untriggered-measure
  strategy (force-AUTO-to-measure). Apply the fix to `capture_until_usable`.
- **H4 — undersampling.** B1 + judge undersample note with a fast square; this is
  dead in sim today, so add the sim test too.
- **H5 — clipping & widen.** A1 / R1 / AS2 / AS3 with large amplitudes; reconcile
  `wide_scale` and rail handling with reality.
- **H6 — edges & loop outcomes.** R3/R6 (DC), L1/L2/L3, AS3 not-converged. Some
  (R4/R7/R9 missing-channel) stay sim-only by nature.
- **H7 — close dead sim branches & reconcile.** Ensure the SimulatedScope is
  corrected to match every divergence found, so sim and bench finally agree.

## Definition of done

- Every reachable recommender state has a committed test.
- Every **analog** state has a committed HIL test driven by a *precisely known*
  stimulus (JDS6600 or, where sufficient, a defined Pico mode).
- All discovered sim-vs-real divergences are logged below **and** fixed in the
  SimulatedScope, so the sim is a faithful proxy for CI.
- `uv run pytest -m hardware` is green on the bench; `-m "not hardware"` green in CI.

---

## Known divergences (sim vs. real) — living log

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | judge false-positive clipping at 1.0 V/div | sim modelled ADC at ±4 div; DS1000Z digitises ~±5 div | carry true saturation rails from the preamble; judge uses them | `e59a644` |
| … | _(to be filled as HIL tests fail and teach us)_ | | | |
