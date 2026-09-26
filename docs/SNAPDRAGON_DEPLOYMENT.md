# Deploying Snapdragoon to Snapdragon

Target: **Windows 11 on Snapdragon** — the HP Omnibook (X2 Plus for 1st place,
Snapdragon X for 2nd).

---

## Read this first: what is verified and what is not

Being precise here is deliberate, and it matters for how you present the work.

| Component | Status | Evidence |
|---|---|---|
| Engine abstraction + all three backends | **Implemented** | `snapdragoon/engines/` |
| Web app, SSE caption stream, accessible UI | **Verified** | 126 tests, `scripts/verify_a11y.py` (11/11) |
| Audio + vision front-ends, DSP, preprocessing | **Verified** | `tests/test_audio.py`, `tests/test_whisper_onnx.py` |
| ONNX / CPU inference path | **Verified, runs correctly** | real transcripts, `fetch_models.py --verify` |
| Whisper ONNX graph handling | **Verified end-to-end** | exact reference transcript, asserted in `tests/` |
| Vision accuracy | **Verified** | correct class on a known subject |
| QNN / NPU inference path | **Implemented, not executed on NPU** | `engines/qualcomm.py` |
| NPU latency numbers | **Not measured** | run `scripts/benchmark.py --backend qualcomm` |

The CPU path is not a stub. It transcribes a reference clip to the known-correct
sentence and labels a known photograph correctly, both asserted in the test
suite. What is missing is only the NPU execution and its numbers, because no
Snapdragon device was available during the build.

Say exactly that in your writeup. A qualified claim survives a technical
interview; an unqualified one that gets caught ends the conversation — and the
prize here is a conversation with Qualcomm engineers.

> **In your submission, never state or imply you ran this on a Snapdragon
> device if you did not.** Say: *"the NPU execution path is implemented and
> documented; the CPU baseline is measured and correct; first hardware run is
> the next step."* Then show the benchmark command you would run.

---

## Path A — ONNX Runtime with the QNN execution provider (fastest)

Best when you want the NPU working today with minimal ceremony.

### 1. Install the QNN-enabled runtime

```powershell
# x64 Python on Windows on Snapdragon
py -m pip install onnxruntime-qnn
```

Verify the provider is actually registered — this is the step people skip:

```powershell
py -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect 'QNNExecutionProvider' in the list
```

If it is missing, the wheel does not match your Python/architecture. Check
`py -c "import platform; print(platform.machine(), platform.python_version())"`
and match the wheel.

### 2. Drop the models in

Place the ONNX exports in `models/`:

```
models/
  whisper-tiny.en/                 <- a directory, not a single file
    encoder_model.onnx
    decoder_model_merged.onnx
    config.json
    added_tokens.json
  mobilenet_v2.onnx
  imagenet_classes.txt
```

> **Whisper is a directory, not a file.** The encoder and decoder are separate
> graphs, and the special-token ids and prompt shape come from the two JSON
> files beside them. A lone `whisper-tiny.en.onnx` will not be picked up, and a
> decoder without `config.json` falls back to hard-coded token ids that are only
> correct for this one checkpoint. `fetch_models.py --check` reports each missing
> file individually for exactly this reason.

`python scripts/fetch_models.py --public` fetches the whole set with no
credentials, which is the fastest way to confirm the runtime works before you
download anything from the hub. For the NPU build itself, use the AI Hub UI
(<https://aihub.qualcomm.com>): pick the **ONNX** runtime, quantise to **INT8**,
download, and place the files as shown above.

### 3. Run on the NPU

```powershell
$env:SNAPDRAGOON_BACKEND = "qualcomm"
py scripts\fetch_models.py --verify
py scripts\benchmark.py --backend qualcomm --runs 50 --json out\npu.json
py -m snapdragoon.web.app
```

`fetch_models.py --verify` is the important one. It loads each model, runs a
genuine inference, and checks the *answer*: the transcript must contain the known
reference sentence and the classifier must place a white Golden Retriever in the
spitz/terrier family. That distinguishes "the file downloaded" from "the file
works."

### 4. If it silently falls back to CPU

The engine reports which backend it chose. Read it:

```
engine    qualcomm-npu
device    Qualcomm AI Hub / QNN on Snapdragon NPU (Hexagon)
NPU       yes
```

If it says `onnx`, the QNN provider was not picked up. Common causes:

- `onnxruntime` and `onnxruntime-qnn` are both installed and the plain one wins.
  Uninstall the plain wheel.
- The model has ops the QNN EP cannot partition, so it falls back per-node. The
  usual culprits are dynamic shapes and non-quantised custom ops. Export INT8,
  fix the input shapes, and check the provider's partitioning log.
- Wrong `QNNExecutionProvider` options for your SoC generation. See
  `snapdragoon/engines/qualcomm.py` for where options are passed.

---

## Path B — QNN context binaries (best performance)

This is the route that gets you full Hexagon offload and the best latency
numbers. It needs a compile step, so it is worth it only if you have time to
iterate.

### 1. Export and quantise

On the AI Hub, download the model compiled for **QNN** (`.so`/`.dlc` context
binary) for your target SoC — Snapdragon X (Oryon) versus X2 Plus differ.

### 2. Stage the context binaries

`engines/qualcomm.py` looks for:

```
models/qnn/
  asr.onnx          # compiled context binary for Whisper Tiny
  classifier.onnx   # compiled context binary for MobileNet V2
```

```powershell
New-Item -ItemType Directory -Force models\qnn
# copy your compiled context binaries in
```

> The filenames are a convention, not a requirement — the point is that the
> NPU artefacts live apart from the portable ONNX ones, because only the
> ONNX files are worth committing.

### 3. Bind the context binary I/O names

QNN binds tensors by name at compile time, so the names must match what
`_run()` in `engines/qualcomm.py` passes:

| Model | Input | Output |
|---|---|---|
| ASR encoder | `input_features` | `last_hidden_state` |
| ASR decoder | `input_ids`, `encoder_hidden_states`, `past_key_values.*`, `use_cache_branch` | `logits`, `present.*` |
| Classifier | *(first input)* | *(first output)* |

The decoder's cache tensors are the fiddly part. The graph names its inputs
`past_key_values.N.<block>.<key|value>` and its outputs
`present.N.<block>.<key|value>` — the same tensors under two names. Anything that
matches them naively never fires, and the symptom is not an error: the cache is
silently never used, every step re-decodes from an empty context, and the output
is fluent-looking but degenerate.

If your compile used different names, adjust the `feeds` dict in `_run()`, or
rename at compile time. This is the single most common cause of a context binary
that loads and then produces garbage.

### 4. Run

```powershell
py -m snapdragoon.web.app
```

---

## Path C — LiteRT with a Hexagon delegate

Relevant if you export from the AI Hub's **TFLite** target rather than ONNX.

```powershell
py -m pip install ai-edge-litert
py -m pip install ai-edge-litert[litert-qnn]   # Hexagon delegate
```

`engines/qualcomm.py` already probes for `ai_edge_litert` and its
`Interpreter`. Place the `.tflite` files where the loader expects and set
`SNAPDRAGOON_BACKEND=qualcomm`.

---

## Capturing system audio instead of the microphone

Captioning a lecture or a video is more impressive than captioning a quiet room.
On Windows, loopback capture goes through WASAPI:

```powershell
py -m pip install soundcard
```

```python
from snapdragoon.audio.capture import open_loopback

for window in open_loopback(seconds=5.0):
    result = engine.transcribe(window, 16000)
    print(result.text)
```

This captures whatever Windows is playing — a Zoom call, a video, a browser.
Everything downstream is identical, because the pipeline only ever sees a
`float32` array at 16 kHz.

On macOS you would use BlackHole as a virtual device; on Linux, PipeWire monitor
sources. The point is that the model path is device-independent.

---

## Proving the deployment in your writeup

Run the benchmark on whatever hardware you have, and report it honestly:

```bash
python scripts/benchmark.py --backend onnx --runs 20 --json out/cpu.json
```

Actual output from this build, on an Apple Silicon Mac (CPU, no NPU involved):

```
platform   macOS-26.6.2-arm64
engine     onnx
device     ONNX Runtime on CPU (development baseline)

ASR  (5s window, real speech from reference-speech.wav)
  mean 393.04 ms | median 392.27 ms | p90 417.60 ms | stdev 16.99 ms
  real-time factor 12.75x  (faster than real time)
  of which: encoder 329.3 ms fixed, 12.0 decode steps 45.9 ms

Vision (640x480 frame)
  mean 4.93 ms | median 4.87 ms | p90 5.26 ms | stdev 0.21 ms
  205.34 fps

In a clean venv (numpy 2.5.3 / Accelerate) the same command reports
ASR median 193.8 ms, 25.8x real time, encoder 148.4 ms, decode 55.0 ms;
vision median 5.71 ms, 175 fps. Publish the range, not the best number.

Attach the JSON. It is proof the numbers came from a real harness, and it
records `source: assets/reference-speech.wav` and `window_seconds: 5.0` so a
reader can see what was actually measured.
```

For the submission, present a table with a **CPU baseline row you actually
measured** and an **NPU row marked as projected**, with the command that will
produce the real number. One honest measured row plus one clearly-labelled
projection is far stronger than two invented ones.

Attach the JSON. It is proof the numbers came from a real harness.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Engine reports `demo` | `onnxruntime` missing, or no models present | `pip install -r requirements-ml.txt`, then `fetch_models.py --verify` |
| Engine reports `onnx` on Snapdragon | QNN EP not registered | Check `ort.get_available_providers()`; remove the plain wheel |
| Context binary loads, output is garbage | Tensor names differ from `_run()` | Align names at compile time or in `_run()` |
| Captions are empty on real speech | Wrong sample rate, or VAD threshold too high | Confirm 16 kHz mono; lower `--silence` |
| Whisper output is gibberish | `tiktoken` missing, so detokenisation fell back to an ASCII projection | `pip install -r requirements-ml.txt` |
| Whisper emits one strange word and stops | Window shorter than 2 s of real audio; the model hallucinated rather than returning nothing | Expected behaviour — the window is skipped. Raise `CAPTURE_CHUNK_SECONDS` if you need longer utterances |
| `check` says a Whisper config file is missing | Graphs present but `config.json`/`added_tokens.json` absent | Re-run `fetch_models.py --public`, or copy both files from the hub download |
| Transcripts repeat themselves | KV cache not bound, so every step re-decodes from an empty context | Confirm input/output names match — see the table in Path B |
| Camera will not open on Windows | `CAP_DSHOW` conflict | `scene.py` already tries DSHOW → MSMF → ANY |
| Latency worse on NPU than CPU | Model not actually offloaded | Check provider partitioning log |

---

## Cleanup after a hardware run

```powershell
$env:SNAPDRAGOON_BACKEND = ""
Remove-Item Env:SNAPDRAGOON_BACKEND
```

Useful when comparing backends in one shell session.
