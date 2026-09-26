<div align="center">

# 🐉 Snapdragoon

**An offline, on-device accessibility co-pilot for Snapdragon® AI Lab**

Live speech captions and scene description, running entirely on the NPU.
No account. No network call. No data leaves the device.

</div>

---

## The problem this solves

On-device AI demos are usually inaccessible by default. Live captions that a
screen reader never announces. Vision output locked in a canvas with no text
alternative. A `div` with a click handler pretending to be a button. Status
messages that appear silently for everyone except the person who cannot see
them.

Snapdragoon treats accessibility as a **technical requirement of the inference
pipeline**, not a UI pass at the end. The whole point is that a caption nobody
can hear is the same as no caption at all.

## What it does

| Capability | Model | Runs on |
|---|---|---|
| Live speech-to-text captions | Whisper Tiny (EN) | Snapdragon NPU via Qualcomm AI Hub / QNN |
| Scene description with confidence | MobileNet V2 (ImageNet-1k) | Snapdragon NPU via Qualcomm AI Hub / QNN |
| CPU fallback for development | ONNX Runtime | Any x86 / ARM CPU, including your Mac |

Both models are small enough to be genuinely real-time on a Snapdragon NPU, and
both are available from the Qualcomm AI Hub, so the deployment path is a compile
step rather than a rewrite.

## Verified results

Measured on this machine (Apple Silicon, macOS arm64, Python 3.13, ONNX Runtime
1.30 on CPU — **not** an NPU). Reproduce with the commands at the bottom.

| Stage | Median | p90 | Throughput |
|---|---|---|---|
| Whisper Tiny, 5 s window | **378 ms** | 396 ms | **13.2× real time** |
| MobileNet V2, 640×480 frame | **5.3 ms** | 5.8 ms | **190 fps** |

Accuracy matters as much as speed, because a fast wrong answer is worthless:

- The reference clip transcribes to
  `"And so my fellow Americans ask not what your country can do for you, ask what
  you can do for your country."`
- The reference photograph of a white Golden Retriever classifies as
  `Samoyed 59% / Pomeranian 10% / West Highland white terrier 6%`.

Both are **asserted in the test suite**, not merely printed here:
`python scripts/fetch_models.py --verify`.

**These are CPU numbers.** No Snapdragon device was available during the build, so
no NPU measurement is claimed anywhere. Exactly what is and is not verified is
tabulated in [`docs/SNAPDRAGON_DEPLOYMENT.md`](docs/SNAPDRAGON_DEPLOYMENT.md).

## Quick start

```bash
git clone <your-fork-url> Snapdragoon
cd Snapdragoon

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m snapdragoon.web.app
```

Open <http://127.0.0.1:8765>. That runs immediately on the placeholder engine —
no downloads, no account — and says so in the UI rather than pretending to be
inference.

For **real inference**, still with no account or token:

```bash
pip install -r requirements-ml.txt
python scripts/fetch_models.py --public   # ~160 MB of public ONNX exports
python scripts/fetch_models.py --verify  # proves they load *and* are correct
python scripts/fetch_models.py --check
python -m snapdragoon.web.app
```

`--public` fetches community ONNX exports of the same two architectures. They are
**not** Qualcomm AI Hub builds and contain no Hexagon-optimised kernels: they
prove the pipeline is correct. `--hub` is the route to the Snapdragon-optimised
builds.

Try the terminal demo, which is the easiest thing to record:

```bash
python scripts/demo.py --image assets/reference-photo.jpg
python scripts/demo.py --file assets/reference-speech.wav
python scripts/demo.py --mic
```

## Architecture

```
microphone / camera / image
          │
          ▼
   capture front-ends            snapdragoon/audio, snapdragoon/vision
   (16 kHz mono float32, RGB uint8)
          │
          ▼
      Engine ABC                 snapdragoon/engines/base.py
          │
   ┌──────┼──────────────┐
   ▼      ▼              ▼
 Demo   ONNX        Qualcomm NPU
 (no ML) (Mac/CPU)   (Snapdragon)
   └──────┴──────────────┘
          │
          ▼
  Flask + SSE                  snapdragoon/web/app.py
          │
          ▼
  accessible front-end         aria-live regions, real buttons,
                              skip link, roving focus, AA contrast
```

Everything above the `Engine` boundary is written once. Swapping runtimes is a
constructor change, which is what makes the Mac-to-Snapdragon story credible
rather than aspirational.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Accessibility

This is the differentiator, so it is specified rather than implied.

- Captions are announced through a **polite live region**, so they never
  interrupt what a screen reader user is currently reading.
- The **full transcript stays available to re-read**. Announcing is not the same
  as being able to revisit, and most implementations only do the first.
- Every control is a **real `<button>`** — native Enter and Space activation, no
  `div[role=button]`.
- **Skip link is the first focusable element**, and stays in the tab order.
- **Focus is never trapped** and focus outlines are never suppressed.
- Live captions are also **mirrored visually**, so the feature is not
  screen-reader-only in reverse.
- All text meets **4.5:1** contrast, with a `prefers-color-scheme` dark mode that
  is checked, not assumed.
- `prefers-reduced-motion` and Windows **forced-colors** are both honoured.
- Every target is at least **44×44 px**.

Prove it:

```bash
python -m snapdragoon.web.app &
python scripts/verify_a11y.py
```

Eleven assertions covering the exact defect classes found in real-world
on-device AI demos: dead heading structure, unnamed controls, `role=button` on
divs, half-implemented tab patterns, unlabelled live regions, focus suppression.
It exits non-zero on failure, so it works as a CI gate.

Full conformance statement: [`docs/ACCESSIBILITY.md`](docs/ACCESSIBILITY.md).

## Deployment

Snapdragoon targets **Windows on Snapdragon** — the HP Omnibook. The three
deployment paths (QNN context binaries, the DirectML QNN execution provider, and
LiteRT with a Hexagon delegate) are all implemented behind the same interface.

```bash
# Baseline, on any CPU, including this Mac
python scripts/benchmark.py --backend onnx --runs 20

# The same measurement on Snapdragon hardware
python scripts/benchmark.py --backend qualcomm --runs 50 --json out/npu.json
```

Same command, same statistics, same code path above the `Engine` boundary — so
the second run produces a real before/after comparison rather than a different
measurement of a different thing.

Step-by-step instructions, including the QNN context build:
[`docs/SNAPDRAGON_DEPLOYMENT.md`](docs/SNAPDRAGON_DEPLOYMENT.md).

## Where the model weights come from

Stated explicitly, because "we used Whisper" and "we used the Qualcomm AI Hub
build of Whisper" are different claims:

| | `--public` (default path) | `--hub` |
|---|---|---|
| Source | community ONNX exports: `onnx/models`, `onnx-community/whisper-tiny.en` | Qualcomm AI Hub |
| Credentials | none | account + API token |
| Hexagon-optimised | no | yes |
| What it proves | the engine, the DSP, the pre/post-processing and the accessibility contract are all correct | that the NPU path works |

No weights are redistributed in this repository. Both routes are downloaded
locally by `scripts/fetch_models.py`, which reports exactly which artefact is
missing rather than failing vaguely.

## Project layout

```
snapdragoon/
  config.py              model registry, paths, tunables
  engines/
    base.py              Engine ABC, Transcript, Detection, label loading
    demo.py              zero-dependency placeholder (clearly labelled)
    onnx_engine.py       real inference on CPU — the Mac path
    whisper_onnx.py      Whisper graph handling: encoder/decoder, KV cache
    qualcomm.py          NPU path via AI Hub / QNN / LiteRT
  audio/capture.py       mic capture, wav replay, silence and content gates
  vision/scene.py        camera frames, image load/save
  web/
    app.py               Flask routes, SSE caption stream
    static/              the accessible front-end
scripts/
  fetch_models.py        get models, and report honestly when it cannot
  benchmark.py           latency, p50/p90, real-time factor
  demo.py                terminal demo, built for screen recording
  verify_a11y.py         automated WCAG self-check
  make_sample_assets.py  regenerate the synthetic demo assets
docs/                    architecture, deployment, accessibility
tests/                   112 tests, including correctness against a known transcript
assets/                  reference clip and photo, plus synthetic demo assets
```

## Testing

```bash
python -m pytest tests/ -q                 # 112 tests
python -m pytest tests/ -q -m "not slow"   # skip anything needing model weights
python scripts/verify_a11y.py              # 11 accessibility assertions
```

The suite runs on a clean checkout with only `numpy` and `flask` installed,
because the demo engine has no third-party dependencies. Tests that need real
weights skip rather than fail, and `filterwarnings = error::RuntimeWarning`
means a numerical warning in the DSP path fails the build — which is how a
silent out-of-bounds read in the mel pipeline was caught.

Two reference assets are checked in specifically so the accuracy tests are real
assertions rather than smoke tests: `assets/reference-speech.wav` (a clip whose
correct transcript is known) and `assets/reference-photo.jpg` (a white Golden
Retriever, which MobileNet places in the spitz/terrier family). Without a
known-correct answer there is nothing to assert.

## Honest limitations

Stated plainly, because a submission that oversells gets caught:

- **No NPU number is claimed.** The Qualcomm path is implemented and documented,
  but the build was done on a Mac because no Snapdragon device was available. The
  performance figures above are CPU. This is the single most important thing to
  resolve before the writeup, and
  [`docs/SNAPDRAGON_DEPLOYMENT.md`](docs/SNAPDRAGON_DEPLOYMENT.md) lists exactly
  what to run first on real hardware.
- **English-only** speech recognition. Whisper Tiny EN is the smallest model that
  is still genuinely useful. The prompt construction already switches
  automatically for a multilingual checkpoint, so that is a config change, not a
  code change.
- Scene description is **whole-image classification**. It will not read text,
  count objects, or localise anything. An object *detector* would be a better
  fit for "describe the room" and is the obvious next step.
- **Captions are windowed, not streaming with context.** Each 5 s window is
  transcribed independently, so a word split across a boundary can be mis-heard.
  It is visible in the reference clip: window 0 ends "…ask not" and window 1
  begins "What your country…". A sliding window with carried-over context is the
  fix.
- The **demo engine produces synthetic captions** and labels itself as such
  everywhere it appears. It exists so a demo never dies on stage, not to pass
  off as a result.

## Licence

MIT. See [`LICENSE`](LICENSE).

<sub>Snapdragon and Qualcomm are trademarks of Qualcomm Incorporated. This
project is not affiliated with or endorsed by Qualcomm.</sub>
