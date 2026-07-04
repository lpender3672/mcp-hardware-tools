# Signal infrastructure plan — addressable captures + the dense lens

> Status: planning. An **infrastructure pass** that fixes one smell and adds one
> missing piece before the next wave of goals lands. It does not build those goals
> yet — it makes them *pluggable*.

## What this pass is for

Two problems, surfaced while designing the agent-facing output contract:

1. **The smell.** `AcquireResult`/`CaptureQuality` is tuned for exactly one
   consumer — the auto-scale loop (vpp / midline / frequency / clipping / fill).
   Every new goal (unknown-signal triage, system-ID, noise robustness, interactive
   debug) wants *different* information. Bolting their fields onto one type rots it
   into a god-object. Fix: keep the loop's view lean; put density in a separate
   lens.

2. **The missing piece.** An unknown **one-shot** capture must be judged,
   classified, decoded, *and* deep-inspected — off **one** frame, because the event
   happened once and cannot be re-captured. Today a `Capture` is a transient return
   value with no identity. Fix: captures become **server-side addressable
   artifacts** every primitive computes over by handle.

Both are prerequisites for the expansion goals from the README + this session:

| Goal | Where it plugs in after this pass |
|---|---|
| analog + digital interactive debug | `describe` dense views (agent-driven) |
| self-correcting acquisition loop | `acquire` → `AcquireResult` + advisory `recommend` |
| unknown-capture triage → decode/process | `classify(features)` off a stored frame |
| black-box / system-ID, noise-source freq response | `describe` PSD/Welch → cross-spectrum (later) |
| JDS6600 noise-robustness suite | `describe` band/percentile/SNR metrics |
| growing instrument set | driver split unchanged; store + primitives are instrument-agnostic |

Every one hangs off the same two new things: **an addressable capture store** and
**a dense characterization lens**. That's the evidence the abstraction is right —
one store + one lens serve all seven.

## Target layering

The existing separation is good; this pass adds one stateful layer above the pure
one and keeps everything below it pure and CI-testable.

```
tools/        MCP tool surface — thin wrappers: resolve capture_id, call analysis   [greenfield]
session/      STATEFUL glue: CaptureStore (handles/eviction) + acquire orchestration  [NEW]
analysis/     PURE functions over value objects: measure, judge, describe, classify,
              recommend   (describe + classify are new; rest reshaped)
decode/       PURE protocol decoders (unchanged)
drivers/      thin "set this, read that" instrument control (unchanged)
model/        value objects: Capture, Waveform, + ChannelReading, AcquireResult,
              Characterization, SignalClass  (new/reshaped)
```

Key rule that preserves testability: **analysis functions take a `Capture`, never a
handle.** Only `session`/`tools` know about ids — they resolve id → `Capture`, then
call the pure function. So `describe(capture, ...)` is unit-testable with a synthetic
`Capture` and no store; `describe_tool(capture_id, ...)` is the thin lookup wrapper.

## The missing piece: capture handles (Phase I1)

`hwtools/session/store.py`:

```python
class CaptureStore:
    """Server-side store of immutable frames, addressed by handle, bounded by BYTES.

    Frames are immutable (frozen Waveform/Capture), so a lens can run any number
    of times over one stored frame — never re-acquiring. Two tiers keep buildup
    structural rather than a cleanup chore:
      * ephemeral — working frames from loops / blind measures; a fresh acquire
        auto-drops the previous ephemeral, so a convergence loop never accumulates.
      * kept — frames the agent (or one-shot safety) promotes; exempt from auto
        eviction, removed only by explicit drop or, LOUDLY, under hard pressure.
    """
    def put(self, capture, *, keep=False, label=None) -> str: ...  # "cap-N", never reused
    def get(self, capture_id: str) -> Capture: ...   # "latest"/label resolve; evicted id RAISES
    def keep(self, capture_id, *, label=None) -> None: ...         # promote ephemeral → kept
    def drop(self, capture_id) -> None: ...                        # explicit reclaim
    def clear_ephemeral(self) -> None: ...
    def list(self) -> list[CaptureInfo]: ...   # audit: id, label, provenance, bytes, kept
    def resident_bytes(self) -> int: ...
```

### Anti-lose-track invariants

1. **IDs are monotonic and never reused.** `cap-7` is that frame forever, or it is
   gone — it can never silently become a *different* frame. A stale reference can
   only ever fail, never resolve to wrong data. This is the core guarantee.
2. **Eviction is loud and legible.** `get(evicted_id)` **raises**, naming the live
   range (`"evicted; live: cap-11..cap-14"`) — never `None`, never the wrong frame
   (per the tool-design philosophy). `list()` lets the agent audit what is resident
   at any moment.
3. **Results carry id + provenance.** Every `AcquireResult` and lens output carries
   its id and the config that produced it, so the agent's context always links a
   number back to its frame.

### Anti-buildup controls

- **Budget by BYTES, not frame count.** A deep DS1054Z frame is ~96 MB/channel
  (12M points × float64); a count cap is meaningless. Evict the oldest *unkept*
  frame until `resident_bytes ≤ budget` (configurable, default ~256 MB). A single
  frame over budget is a loud error suggesting a lower memory depth — never a silent
  OOM.
- **Ephemeral self-reclaim.** A fresh `acquire` auto-drops the previous ephemeral;
  `latest` still resolves. A loop that fires dozens of blind measures leaves the
  store near-empty. Accumulation requires a deliberate, visible `keep` — it cannot
  creep.
- **One-shot safety.** A SINGLE-sweep deep capture is **auto-kept**
  (`kept=True, reason="one-shot"`): an unrepeatable event is never an auto-eviction
  victim, only an explicit `drop`. Safety policy, not acquisition policy.
- **`latest` default + labels.** Primitives default `capture_id="latest"`, so the
  agent has no reason to hoard ids; labels are agent-friendly aliases resolving to a
  specific frame, cutting "which id was that". Explicit ids still let it hold two
  frames (stimulus vs response for system-ID).
- **Concurrency**: none — single agent, single link. A plain dict + insertion order.

> **Reserve lever (v2):** the DS1054Z is an 8-bit digitiser — storing raw ADC codes
> (uint8 + scale/offset) instead of float64 volts is ~8× smaller, volts materialized
> on demand. M3d already captures RAW, so the plumbing exists; deferred to avoid
> complicating `Waveform` now.

`hwtools/session/acquire.py` — collapse the in-flight one-shot/repeating split into
**one honest primitive**, sweep mode as a parameter (policy belongs to the agent):

```python
def acquire(scope, store, *, channels, timebase, trigger, acquire_cfg=None,
            sweep=SweepMode.SINGLE) -> tuple[str, AcquireResult]:
    """Arm, wait a timebase-scaled budget for the trigger, deep-read one frame.

    Never forces a trigger. On a missed trigger returns triggered=False with no
    frame (honest). SINGLE for a triggered event; AUTO when you want a frame before
    the trigger level is known (blind measurement). Stores the frame, returns its
    handle + the lean decision view.
    """
```

This supersedes the dirty `loop.py::_acquire_*` and `tests/hardware/_acquire.py`
work — same arm/wait mechanics, one function, sweep as a param. `force_trigger`
disappears from the acquisition path (kept on the driver only for completeness; no
primitive calls it).

## Kill the smell: reshape the decision view (Phase I2)

Replace `CaptureQuality`'s parallel dicts with per-channel objects that fold
`measure` + `judge` together — one decision-ready reading per channel.

```python
class ChannelReading(BaseModel):        # model/reading.py
    config: ChannelConfig               # what it was TAKEN with → agent computes deltas, holds no state
    vpp: float
    midline: float                      # (vmax+vmin)/2 — offset & trigger target
    mean: float                         # DC level; mean≠midline ⇒ DC offset / asym duty
    frequency: float | None             # None ⇒ flat/noise ⇒ keep timebase
    clipping: bool
    clipped_fraction: float             # 1% grazing vs 40% flat-top ⇒ widen a little vs a lot
    fill_fraction: float

class AcquireResult(BaseModel):
    triggered: bool
    bandwidth_ok: bool
    timebase: TimebaseConfig
    channels: dict[ChannelId, ChannelReading]
    notes: list[str]                    # judge's human-readable "why", unchanged
    @property
    def usable(self) -> bool: ...       # triggered and no channel clipping
```

Migration:

- `judge_capture` + the per-channel `measure` calls collapse into one assembler,
  `read_channels(capture, channels, capabilities) -> dict[ChannelId, ChannelReading]`.
- `AcquireResult` subsumes `CaptureQuality`; delete the parallel-dict type.
- **Blast radius** (all internal, sim-testable): `analysis/adjust.py::suggest_adjustment`
  (reads `quality.clipping[ch]`/`fill_fraction[ch]` → reads `result.channels[ch]`),
  `analysis/loop.py` (`quality.usable` → `result.usable`), `analysis/recommend.py`
  (already takes a `Capture`; unaffected), and the unit tests that assert on the old
  shape.
- `recommend_setup` stays a **pure advisory** that returns a `Setup` — it is *not*
  folded into acquire, and it becomes a standalone tool taking a `capture_id`. This
  is the line that keeps the agent, not Python, in the loop.
- `autoset` / `capture_until_usable` are retained but **demoted to experimental
  convenience wrappers** over the primitives (fast deterministic path for the routine
  case), clearly marked, not the main surface.

Fields deliberately **excluded** from the lean view (derivable or diagnostic — they
live in `describe`, not here): `vmin`/`vmax`, `rms`, `saturation`, `samples_per_period`.

## The dense lens: describe / characterize (Phase I3)

`hwtools/analysis/describe.py` — one pure function, two tiers. The shared substrate
for triage, interactive debug, noise-robustness, and system-ID.

```python
def describe(capture, channel, *,
    percentiles=(1,25,50,75,99),   # inverse-CDF: robust spread/outliers
    value_bins=None,               # opt-in PDF: shape / multimodality
    time_bins=None,                # segment record → non-stationarity (bursts, drift, dropout)
    spectrum=False, freq_bins=None,# opt-in PSD (Welch), down-binned to caller resolution
    joint=False, joint_bins=None,  # opt-in 2D time×value grid — the densest single view
) -> Characterization
```

`Characterization`:

- **Always-on feature vector** (small, fixed — what `classify` reads and the agent
  reads for a quick verdict): `n_levels`/modality, `edge_rate_hz`, `periodicity`
  (0..1 autocorrelation strength) + `period_s`, `duty`, `spectral_flatness` (Wiener
  entropy: ~0 pure tone, ~1 white noise), `peak_hz` + `peak_prominence`, `dc_level`,
  `vpp`.
- **Opt-in dense views** (caller sets which lenses at what resolution): value
  histogram, per-time-bin stats (percentiles/vpp/mean per window), PSD, joint 2D
  histogram. Default coarse (feature vector + a few percentiles); opt into the
  expensive grids.

Caller-set granularity generalizes from "bins" to "**which lenses, at what
resolution**" — the agent governs its *observation*, which is the good kind of agent
control (on the "feed rich state to the agent" side of the line, not policy).

Extends the existing `spectrum.py` foundation: add `spectral_flatness`, Welch PSD
(`nperseg` = caller freq resolution — already flagged there as the next addition),
and down-binning to caller resolution. Spectrogram / eye-diagram are v2 dense views.

Why the joint 2D histogram earns its place: it's the *joint* distribution, not two
marginals — digital reads as two horizontal bands, sine as a filled envelope, noise
as a fuzzy cloud, drift as sloping bands, all from one integer grid. It's the front
half of an eye diagram and the single densest classification input.

## Triage skeleton: the unknown-capture entry point (Phase I4)

`hwtools/analysis/classify.py` — consume the **feature vector** (not the raw frame)
→ a class + routing hints. Rule-based first; the "yet-to-be-decided algorithm" stays
pluggable behind this signature.

```python
class SignalClass(StrEnum):
    FLAT_DC, SINE, SQUARE_DIGITAL, NOISE, MODULATED, UNKNOWN = ...

def classify(features) -> Classification:  # class + confidence + hints:
    #   est_symbol_rate_hz (edge_rate / periodicity) → seeds UART baud / SPI clk
    #   n_active_lines (across channels) → SPI (3) vs I2C (2) vs single-line
    #   suggested_primitive → which decode/process path to route to
```

`triage(capture_id)` composes `describe` (feature vector across channels) →
`classify` → a suggested next primitive. This is the front door for "point the loop
at an unknown system," and it reads the *same* lens the agent debugs with.

## Tool surface wiring (Phase I5)

Fill the empty `tools/` package with thin MCP wrappers over `session` + `analysis`,
each resolving `capture_id` via the store:

`acquire`, `describe`, `judge`, `recommend` (advisory), `decode_uart|spi|i2c`,
`triage`. Experimental: `autoset`, `capture_until_usable`. This is where the
agent-facing surface actually lands; everything under it is already built and tested.

## Sequence

- **I0 — reconcile the in-flight acquisition change.** The dirty `loop.py` /
  `_acquire.py` (one-shot vs repeating, no force) is superseded by the single
  `session/acquire.py` primitive. Either bench-run + commit as an interim, or fold
  directly into I1 — don't leave it dangling.
- **I1 — capture store + handles + the one honest `acquire` primitive.** The backbone.
- **I2 — reshape to `ChannelReading`/`AcquireResult`; retire `CaptureQuality`;
  demote autoset/loop to experimental; make `recommend` standalone advisory.**
- **I3 — `describe` lens** (feature vector + opt-in dense views) extending `spectrum.py`.
- **I4 — `classify`/`triage` skeleton** over the feature vector.
- **I5 — `tools/` MCP surface** over session + analysis.

I1→I2 are the smell + missing-piece fix and should land together as the true infra
pass. I3→I5 bring the new goals in as pluggable surfaces.

## Testing

- **Pure (`analysis`, `decode`, `model`)** — unit tests on synthetic waveforms:
  sine → arcsine value histogram; square → bimodal + high edge-rate; noise → flat
  spectrum + Gaussian histogram; DC → single spike + `frequency=None`. `describe` and
  `classify` are fully CI-testable with zero hardware.
- **Stateful (`session`)** — `CaptureStore` eviction/`latest`/handle round-trip;
  `acquire` against the SimulatedScope (arm/wait/store/return).
- **HIL** — migrate existing hardware tests to the single `acquire` primitive; the
  reshape is otherwise invisible to them.

## Non-goals (deferred, but unblocked by this pass)

- JDS6600 driver / `SignalGenerator` interface (README H1) — separate track.
- The noise-robustness suite — builds on `describe`'s metrics later.
- Cross-spectrum / coherence system-ID and the spectrogram/eye-diagram dense views — v2.
- A sophisticated (ML) classifier — I4 ships a rule-based skeleton behind a stable
  signature so it can be swapped without touching callers.
```
