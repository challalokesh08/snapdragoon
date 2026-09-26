# Architecture

One design decision drives everything else: **the inference backend is a
pluggable boundary**, and nothing above it knows which backend it is talking to.

That is what makes the Mac-to-Snapdragon story credible. It is not "we wrote
some code for the NPU" — it is "the same pipeline, the same pre/post-processing,
and a one-line change to which runtime executes it."

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│ Front-end        snapdragoon/web/static/                     │
│                  accessible HTML + CSS + vanilla JS           │
│                  aria-live regions, real buttons, AA contrast │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTP  ·  SSE
┌───────────────────────────▼──────────────────────────────────┐
│ Transport        snapdragoon/web/app.py                     │
│                  routes, SSE caption stream, model disclosure │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ Front-ends       snapdragoon/audio/capture.py                │
│                  snapdragoon/vision/scene.py                 │
│                  → always hand off float32 @ 16 kHz           │
│                  → always hand off uint8 RGB HxWx3            │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ Boundary         snapdragoon/engines/base.py                 │
│                  Engine ABC · Transcript · Detection         │
│                  ← the only thing above this line imports    │
└───────────────────────────┬──────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ DemoEngine    │  │ OnnxEngine    │  │ QualcommEngine│
│ no deps       │  │ CPU, portable │  │ NPU (Hexagon) │
│ synthetic     │  │ dev baseline  │  │ target device │
└───────────────┘  └───────────────┘  └───────────────┘
```

## The `Engine` contract

```python
class Engine(abc.ABC):
    name: str
    device_description: str
    is_neural_accelerated: bool

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript: ...
    def classify(self, frame: np.ndarray) -> list[Detection]: ...
    def describe(self) -> dict: ...
    def warmup(self) -> None: ...
```

Two invariants make the abstraction worth its cost:

1. **Inputs are pre-normalised by the front-end, not the engine.** Every backend
   receives 16 kHz mono `float32` in `[-1, 1]`, or `uint8` RGB `HxWx3`. A
   backend that wanted 48 kHz stereo or BGR would have to convert, which means
   the conversion is visible in one place instead of three.

2. **`device_description` is displayed verbatim in the UI.** The string is not
   decoration — it is a contract that the engine must be honest about where
   inference is running. `DemoEngine` says
   `"Synthetic placeholder - no model inference is performed"`, and the front-end
   renders a warning banner off the back of it.

## Backend selection

`snapdragoon/engines/__init__.py::get_engine()`, with preference from
`SNAPDRAGOON_BACKEND`:

```
auto  →  Qualcomm  →  ONNX (if a model is present)  →  Demo
onnx  →  OnnxEngine
qualcomm → QualcommEngine   # raises if not on a Snapdragon SoC
demo  →  DemoEngine
```

`auto` tries the NPU first and logs why it fell back, so a silent CPU downgrade
is always visible in the log and in `/api/status`. This is deliberate: the most
common way an NPU demo misleads is by quietly running on the CPU and reporting
the same numbers.

## Data shapes

| Stage | Shape | Type | Notes |
|---|---|---|---|
| Mic capture | `(N,)` | `float32` | 16 kHz mono, `[-1, 1]` |
| Wav replay | `(N,)` | `float32` | resampled check enforced at 16 kHz |
| Whisper features | `(1, 80, 3000)` | `float32` | log-mel, 30 s context |
| Whisper logits | `(1, T, V)` | `float32` | `V=51864`; the vocab axis is resolved at load, not assumed |
| Camera frame | `(H, W, 3)` | `uint8` | RGB |
| Image batch | `(1, 3, 224, 224)` | `float32` | NCHW, ImageNet-normalised |
| Classifier output | `(1, 1000)` | `float32` | softmax → top 3 |

The DSP is implemented in numpy rather than pulled from a library for two
reasons: no OpenCV/PortAudio dependency is needed just to test the feature
pipeline, and the arithmetic is then identical on the NPU, so a numerical
discrepancy there is a real finding rather than a library difference.

Three implementation notes worth knowing if you extend it:

- `log_mel_spectrogram` accumulates in `float64`. Squaring a power spectrum
  overflows `float32` on loud input, and the resulting `inf` propagates into the
  model as garbage rather than an error — the kind of failure that looks like a
  bad model.
- It also normalises its own input length. The frame loop strides a fixed
  3000x400 window over the buffer with no bounds check, so a short input does not
  raise — it reads past the end of the array and returns whatever memory
  followed. Normalising the length inside the function makes that contract
  impossible to misuse, and the test suite asserts it.
- `preprocess_image` uses nearest-neighbour resize and centre-crop. Not
  high-quality, but deterministic across machines, which is what keeps the
  benchmark numbers comparable.

The mel filterbank uses the **Slaney** scale and area normalisation, and both
matter. The scale is linear below 1 kHz and logarithmic above; the normalisation
(`enorm = 2 / (edge[i+2] - edge[i])`) equalises filter area so the wide
low-frequency filters do not swamp the narrow high-frequency ones where sibilants
live. Measured on the implementation here, normalisation cuts the area spread
across the 80 bands from **7.96x to 1.16x**; the residual is discretisation loss
on the narrowest filters, not a defect. The window is **periodic**, not
symmetric — Whisper uses `torch.hann_window(periodic=True)`, which does not
end at exactly 0 the way a symmetric window does.

## Running a Whisper ONNX graph

`snapdragoon/engines/whisper_onnx.py` is the highest-risk file in the project.
Whisper's ONNX exports are unusual in ways that produce *plausible wrong answers*
rather than errors, so each trap is documented in the code and asserted in
`tests/test_whisper_onnx.py`.

**Drive everything by name and declared shape, never by position.** Calling
`session.get_inputs()[0]` and assuming it is the features tensor works for one
export and breaks silently for the next. Token ids are found by asking which
input is 2-D `int64`; the vocab axis is found by asking which declared dimension
equals `vocab_size`.

| Trap | What actually happens |
|---|---|
| `get_inputs()[0]` ordering | A different export order feeds the encoder the token ids. No error, total garbage. |
| `present.N.*` vs `past_key_values.N.*` | The *same* tensors under two names. Match them naively and the KV cache is never used: every step re-decodes from an empty context, which reads as fluent but degenerate repetition. |
| Encoder `present` on the cached path | Returns a `(0, 6, 1, 64)` **zero-batch** placeholder. Feed it back and encoder attention dies from the second token onward, so the transcript is right for a few words and then stops for no visible reason. |
| Filtering symbolic dims out of a shape | `['batch_size', 6, 'past_seq', 64]` reduced to ints becomes `[6, 64]`, which reads as `heads=64` and fails with `INVALID_ARGUMENT ... index: 1 Got: 64 Expected: 6`. Walk the shape positionally instead. |
| `logits` axis order | Exports differ between `(b, seq, vocab)` and `(b, vocab, seq)`. Resolve from the declared shape against `vocab_size`. |
| Hard-coded special-token ids | `SOT`/`EOT`/`<\|notimestamps\|>` differ per checkpoint. Read them from the model's own `config.json` and `added_tokens.json`. Note that `tiktoken.encode("<\|startoftranscript\|>")[0]` is `27`, not the special id — the BPE vocabulary and the added-token table are different things. |
| `<\|en\|>` on an English-only checkpoint | Costs accuracy for a token with no meaning there. The prompt switches on `is_multilingual` from `config.json`. |

Two export layouts are supported: encoder + decoder with a KV cache (the common
one) and a single merged graph. `describe()` reports which was loaded, the
resolved vocab axis, and where the token ids came from, so a misconfigured model
shows up in `/api/status` instead of being a mystery.

## The SSE caption stream

`POST /api/captions?source=mic|file`

```
client                          server
  |  POST /api/captions            |
  |------------------------------>|  capture all request args
  |                                |  spawn producer thread
  |  retry: 3000                   |   ├─ capture windows
  |  data: {"kind":"caption",...}  |<─┤ engine.transcribe(window)
  |  data: {"kind":"skip",...}     |<─┤ window was silent, or too short to trust
  |  data: {"kind":"error",...}    |<─┤ capture or inference failed
  |  data: {"kind":"end"}          |<─┤ producer finished
  |  data: {"kind":"closed"}       |<─┤ stream() generator returned
  |<------------------------------|
```

Design decisions in that diagram:

- **A bounded queue (64), dropping the oldest event.** Captions are transient by
  nature. A backlog of stale captions is worse than a gap, and backpressure
  would stall the microphone buffer.
- **Silence emits `skip`, and the client deliberately does not announce it.** A
  live region emitting "silence" every five seconds is unusable.
- **A window with under 2 s of real audio is skipped, not transcribed.** Whisper
  is trained on 30 s of contiguous audio; a short window is zero-padded up to that
  and the model answers a mostly-empty window with a *confident hallucination*
  rather than with nothing. On the reference clip a 1 s tail produced a stray
  `"Godfrey."` A stray proper noun is worse than no caption. The check lives in
  the engine, because only the engine knows what padding it is about to add, and
  the verdict travels back as `Transcript.skip_reason`.
- **A 15-second keepalive comment** so proxies do not time out the stream.
- **All `request` access happens on the request thread.** The producer thread
  has no Flask application context, so every request-scoped value is captured
  before the thread starts.
- **Every exit path terminates.** `end` → `closed` → the `EventSource` closes
  client-side. There is no path that hangs the connection or strands focus.

## Front-end design notes

Vanilla JS, no framework. The whole point of the project is accessibility, and a
build step would add a failure mode for no benefit.

- **All model output goes in via `textContent`.** Never `innerHTML`. A
  mis-transcribed string cannot inject markup, and this is asserted in review.
- **The live region is never created or destroyed.** It exists in the initial
  HTML. A region added at the same moment as its content is frequently not
  announced, which is the most common way this feature is quietly broken.
- **Button state is mirrored in `aria-disabled`** as well as the `disabled`
  attribute, so the state is exposed even if the attribute is mis-read.
- **The `EventSource` is closed on stop and on `beforeunload`**, so no keyboard
  trap is possible.

## Testing strategy

134 tests, and the suite runs on a clean checkout with only `numpy` and `flask`
installed — because the demo engine has no third-party dependencies. Tests that
need real weights skip rather than fail; mark them with `-m slow` to skip them
deliberately.

| File | Covers |
|---|---|
| `tests/test_engines.py` | the `Engine` contract, TTS-safe output formatting |
| `tests/test_audio.py` | silence and content detection, wav replay, image preprocessing |
| `tests/test_whisper_onnx.py` | cache shapes, token loading, prompt shape, the mel front-end, a **contract fake** for the KV cache, and correctness against a known transcript |
| `tests/test_web.py` | routes, path traversal, SSE contract, 18 accessibility assertions |
| `tests/test_benchmark.py` | that the benchmark measures inference and refuses to invent a figure |

Three choices here are worth defending:

**`filterwarnings = error::RuntimeWarning`** in `pytest.ini`. A numerical warning
in the DSP path is a real defect, not noise. This is how the out-of-bounds read
in `log_mel_spectrogram` surfaced: a perfectly shaped array of garbage, no
exception, one warning.

**A fake session that mimics the real export contract.** The encoder-cache bug
described above is invisible in normal testing — the output is a plausible
transcript that is simply truncated, with no error anywhere. `tests/` contains a
stand-in session that reproduces the `(0, 6, 1, 64)` placeholder exactly, and
asserts it is never fed back. That is the only practical way to lock in a fix for
a bug that cannot fail loudly.

**Reference assets with known-correct answers.** `assets/reference-speech.wav`
and `assets/reference-photo.jpg` exist so the accuracy tests assert something.
A test that only checks "the model returned some text" would have passed while
the encoder cache was broken.

The accessibility assertions encode defect classes that actually show up in
on-device AI demos: dead heading structure, `role="button"` on divs,
half-implemented tab patterns, `aria-hidden` on the only label, suppressed focus
outlines, and live regions that do not exist. `scripts/verify_a11y.py` runs the
same checks against the live server, so they also work as a pre-submission gate.

Markup assertions strip HTML comments first. The source is heavily commented and
several notes *discuss* markup like `<aside>` and `tabindex="0"` — left in, a
comment is indistinguishable from a real element and the assertions both fail
spuriously and pass spuriously.

## Extending it

| Goal | Where |
|---|---|
| Swap in a different ASR model | `config.py::MODEL_REGISTRY` — no engine changes |
| Add object *detection* (boxes, not classification) | new `Engine.detect()`; `Detection` is already a per-item record |
| Add multilingual captions | change the registry entry to a multilingual Whisper; the prompt switches on `is_multilingual` automatically |
| Add a new backend | subclass `Engine`, register it in `get_engine()` |
| Caption system audio | `audio/capture.py::open_loopback` (Windows/WASAPI) |
| Add an SSIM/quality metric | new method on `Engine`, new route, new panel |

The registry is the extension point worth knowing. A model swap is a config
change, which is why the ONNX and Qualcomm engines do not duplicate any model
metadata.
