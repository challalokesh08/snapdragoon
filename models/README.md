# Model directory

Weights are **fetched, not committed.** This directory is in `.gitignore` apart
from this file, because the two graphs are ~160 MB and their provenance is
recorded rather than vendored.

No model weights are redistributed in this repository.

## Getting them

```bash
pip install -r requirements-ml.txt

# No account, no token. Community ONNX exports of the same two architectures.
python scripts/fetch_models.py --public

# Reports exactly which artefact is missing, by name.
python scripts/fetch_models.py --check

# Loads each model, runs a real inference, and checks the answer is correct.
python scripts/fetch_models.py --verify
```

`--public` downloads from `onnx/models` (MobileNet V2) and
`onnx-community/whisper-tiny.en` (Whisper). These are **not** Qualcomm AI Hub
builds and contain no Hexagon-optimised kernels — they exist to prove the
engine, the DSP and the pre/post-processing are correct without requiring an
account. For the Snapdragon-optimised builds see
[`../docs/SNAPDRAGON_DEPLOYMENT.md`](../docs/SNAPDRAGON_DEPLOYMENT.md).

## Expected layout

```
models/
  mobilenet_v2.onnx
  imagenet_classes.txt
  whisper-tiny.en/
    encoder_model.onnx
    decoder_model_merged.onnx
    config.json
    added_tokens.json
```

**Whisper is a directory, not a single file.** This is not a convention, it is a
requirement:

- The encoder and the decoder are separate graphs.
- The special-token ids (`<|startoftranscript|>`, `<|endoftext|>`,
  `<|notimestamps|>`) come from `config.json` and `added_tokens.json`. Without
  them the runner falls back to hard-coded constants that are only correct for
  this one checkpoint.
- `is_multilingual` in `config.json` decides whether the prompt includes a
  language token. An English-only checkpoint given `<|en|>` loses accuracy.

`--check` reports each of these files individually, so a partial download is
visible rather than showing up later as a wrong answer.
