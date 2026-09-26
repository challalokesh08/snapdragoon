#!/usr/bin/env python3
"""Fetch the models Snapdragoon needs, and report honestly when it cannot.

There are two sources, and the difference matters:

``--public``
    No account, no token. The reference ONNX exports of the *same two
    architectures* from public model hubs. This is what makes the project
    reproducible by a reviewer in one command, and it is the path the
    verification numbers in the README were produced on.

``--hub``
    Qualcomm AI Hub. These are the Snapdragon-optimised exports, and the ones you
    want on real hardware. Needs an account and an API token.

Neither source is presented as a substitute for the other. The public exports
are the same networks, not the same binaries: they prove the engine and the
pre/post-processing are correct, and they are not what you would ship on a
Snapdragon X Elite.

Sub-commands:

    --check    report which artefacts are present and which are missing
    --public   download the credential-free reference exports
    --hub      Qualcomm AI Hub download help
    --verify   load each model and run one real inference, so you know the
               artefact is not merely present but functional

Usage:
    python scripts/fetch_models.py --public
    python scripts/fetch_models.py --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapdragoon import config  # noqa: E402

# --- credential-free reference exports ---------------------------------------
#
# Provenance matters more than convenience here, so it is stated rather than
# implied: these are the community ONNX exports of the same architectures named
# in config.MODEL_REGISTRY. They are NOT Qualcomm AI Hub builds and contain no
# Hexagon-optimised kernels.

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/onnx/models/main/validated/vision/"
    "classification/synset.txt"
)

MOBILENET_V2_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mobilenet/model/mobilenetv2-12.onnx"
)

_HF = "https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main"
WHISPER_FILES = {
    "onnx/encoder_model.onnx": "encoder_model.onnx",
    "onnx/decoder_model_merged.onnx": "decoder_model_merged.onnx",
    "config.json": "config.json",
    "added_tokens.json": "added_tokens.json",
}

# The AI Hub model slugs, per task. Look these up in the hub UI at
# aihub.qualcomm.com and download as the ONNX runtime.
HUB_SLUGS = {
    "asr": "whisper-tiny.en",
    "classifier": "mobilenet_v2_imagenet",
}

# Files that must accompany the Whisper graphs for it to decode correctly.
WHISPER_CONFIG_FILES = ("config.json", "added_tokens.json")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _missing(msg: str) -> None:
    print(f"  [MISS] {msg}")


def _info(msg: str) -> None:
    print(f"  [..]   {msg}")


def _download(url: str, dest: Path, label: str) -> bool:
    """Fetch ``url`` to ``dest``, reporting honestly on any failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _info(f"downloading {label} …")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Snapdragoon/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, \
                dest.open("wb") as fh:
            total = 0
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
                total += len(chunk)
    except Exception as exc:  # noqa: BLE001
        _missing(f"{label}: {exc}")
        # Leave no truncated file behind for the next run to trust.
        dest.unlink(missing_ok=True)
        return False
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb < 0.01:
        _missing(f"{label} is implausibly small ({size_mb:.3f} MB); discarded")
        dest.unlink(missing_ok=True)
        return False
    _ok(f"{label} ({size_mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _describe(spec) -> str:
    path = config.model_path(spec)
    if spec.is_dir:
        if not path.is_dir():
            return f"{path.name}/ (directory of graphs)"
        graphs = sorted(p.name for p in path.glob("*.onnx"))
        total = sum((path / g).stat().st_size for g in graphs) / (1024 * 1024)
        return f"{path.name}/ [{', '.join(graphs) or 'no graphs'}] ({total:.1f} MB)"
    if path.is_file():
        return f"{path.name} ({path.stat().st_size / (1024 * 1024):.1f} MB)"
    return path.name


def check() -> int:
    print(f"\nModel directory: {config.MODELS_DIR}\n")
    missing = 0
    for key, spec in config.MODEL_REGISTRY.items():
        if config.has_model(spec):
            _ok(f"{key:11s} {_describe(spec)}")
        else:
            _missing(f"{key:11s} {_describe(spec)}")
            missing += 1

    # Whisper's config files are not optional decoration: the special-token ids
    # and the prompt shape come from them, and the fallback constants are only
    # correct for this one checkpoint.
    asr_dir = config.model_path(config.MODEL_REGISTRY["asr"])
    if asr_dir.is_dir():
        for name in WHISPER_CONFIG_FILES:
            if (asr_dir / name).is_file():
                _ok(f"{'asr cfg':11s} {name}")
            else:
                _missing(f"{'asr cfg':11s} {name} (token ids, prompt shape)")
                missing += 1

    labels = config.MODELS_DIR / (config.MODEL_REGISTRY["classifier"].labels_path or "")
    if labels.is_file():
        _ok(f"{'labels':11s} {labels.name} "
            f"({len(labels.read_text().splitlines())} classes)")
    else:
        _missing(f"{'labels':11s} {labels.name} (class labels)")
        missing += 1

    print()
    if missing:
        print(f"{missing} artefact(s) missing. Try: --public\n")
    else:
        print("All model artefacts present. Next: --verify\n")
    return missing


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def fetch_public() -> bool:
    """Download the credential-free reference exports."""
    print("\n  Public reference exports (same architectures, NOT AI Hub builds)\n")
    ok = True

    ok &= _download(IMAGENET_LABELS_URL,
                    config.MODELS_DIR / "imagenet_classes.txt",
                    "imagenet_classes.txt")
    # The raw synset list is ordered; the count is a cheap integrity check.
    labels = config.MODELS_DIR / "imagenet_classes.txt"
    if labels.is_file():
        n = len(labels.read_text().splitlines())
        if n < 1000:
            _missing(f"label file has {n} lines, expected 1000; it was not written")
            labels.unlink(missing_ok=True)
            ok = False
        else:
            _ok(f"{n} labels verified")

    ok &= _download(MOBILENET_V2_URL,
                    config.model_path(config.MODEL_REGISTRY["classifier"]),
                    "mobilenet_v2.onnx")

    asr_dir = config.model_path(config.MODEL_REGISTRY["asr"])
    for remote, local in WHISPER_FILES.items():
        ok &= _download(f"{_HF}/{remote}", asr_dir / local, f"whisper-tiny.en/{local}")

    print()
    return bool(ok)


def fetch_from_hub() -> bool:
    token = os.environ.get("QUALCOMM_AI_HUB_TOKEN") or os.environ.get("QAI_HUB_TOKEN")
    if not token:
        _missing("QUALCOMM_AI_HUB_TOKEN is not set.")
        print(
            f"""
  Qualcomm AI Hub needs an account and a token:

    1. Sign up at https://aihub.qualcomm.com
    2. Create an API token under your profile
    3. export QUALCOMM_AI_HUB_TOKEN='<your token>'
    4. pip install qai-hub

  Then re-run:  python scripts/fetch_models.py --hub

  Manual route (no tooling required, and honestly the faster one on a deadline):
    1. Open the model page on aihub.qualcomm.com
    2. Choose the "ONNX" runtime, download the artefact
    3. Place it as shown below

  Model pages and expected layout:
    whisper-tiny.en        ->  models/whisper-tiny.en/
                                encoder_model.onnx
                                decoder_model_merged.onnx
                                config.json
                                added_tokens.json
    mobilenet_v2_imagenet  ->  models/mobilenet_v2.onnx

  Whisper needs a *directory*: the encoder and decoder are separate graphs plus a
  config, and several decoder exports are interchangeable. A single-file
  `whisper-tiny.en.onnx` will not be picked up.
"""
        )
        return False

    try:
        import qai_hub  # type: ignore  # noqa: F401
    except ImportError:
        _missing("qai-hub is not installed. Try: pip install qai-hub")
        return False

    print(
        "\n  The AI Hub client is installed and a token is present.\n"
        "  Download each model as the ONNX runtime from the hub UI, then place\n"
        "  them in models/ with the exact layout listed above.\n"
        "  (The client API for ONNX export has changed across hub versions, so\n"
        "  this script deliberately does not hard-code a call that may break.)\n"
    )
    _info("run --verify once the files are in place")
    return True


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify() -> int:
    """Load each model and run one genuine inference. Fails loudly."""
    print("\nLoading models and running one real inference each …\n")
    failures = 0
    try:
        from snapdragoon.engines.onnx_engine import OnnxEngine
    except ImportError as exc:
        _missing(f"cannot import the ONNX engine: {exc}")
        print("  pip install -r requirements-ml.txt\n")
        return 1

    try:
        engine = OnnxEngine()
    except ImportError as exc:
        _missing(f"onnxruntime not installed: {exc}")
        print("  pip install -r requirements-ml.txt\n")
        return 1

    if not engine.has_asr and not engine.has_classifier:
        _missing("no models loaded")
        print("  python scripts/fetch_models.py --public\n")
        return 1

    if engine.has_asr:
        print(f"  ASR graph: {engine.describe().get('asr')}")

    import numpy as np

    if engine.has_classifier:
        photo = config.PROJECT_ROOT / "assets" / "reference-photo.jpg"
        frame = None
        if photo.is_file():
            try:
                from snapdragoon.vision.scene import load_image

                frame = load_image(photo)
            except Exception as exc:  # noqa: BLE001
                _missing(f"could not load {photo.name}: {exc}")
        if frame is None:
            frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
        try:
            dets = engine.classify(frame)
            top = dets[0]
            _ok(f"classifier ran in {top.latency_ms:.1f} ms -> {top.label} "
                f"({top.score:.1%})")
            if photo.is_file():
                # A known subject: a white Golden Retriever. If preprocessing were
                # wrong this would be a texture or a background class.
                labels = " ".join(d.label.lower() for d in dets)
                if not any(w in labels for w in ("retriever", "samoyed", "terrier",
                                                 "pomeranian", "sled", "husky")):
                    _missing(f"classifier missed a known subject: {labels}")
                    failures += 1
                else:
                    _ok("classifier identified the known subject")
        except Exception as exc:  # noqa: BLE001
            _missing(f"classifier inference failed: {exc}")
            failures += 1

    if engine.has_asr:
        clip = config.PROJECT_ROOT / "assets" / "reference-speech.wav"
        try:
            import wave

            with wave.open(str(clip)) as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:  # noqa: BLE001
            _missing(f"no reference clip to verify against: {exc}")
            audio = None

        if audio is not None:
            try:
                res = engine.transcribe(audio, sr)
                _ok(f"ASR ran in {res.latency_ms:.1f} ms")
                print(f"         transcript: {res.text!r}")
                expected = ("and so my fellow americans ask not what your country "
                            "can do for you")
                got = " ".join(res.text.lower().split())
                if expected in got:
                    _ok("ASR matches the reference transcript")
                else:
                    _missing("ASR transcript does not match the reference")
                    print(f"         expected to contain: {expected!r}")
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                _missing(f"ASR inference failed: {exc}")
                failures += 1

    print()
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report present/missing models")
    ap.add_argument("--public", action="store_true",
                    help="download credential-free reference exports")
    ap.add_argument("--hub", action="store_true", help="Qualcomm AI Hub download help")
    ap.add_argument("--verify", action="store_true", help="load models and infer once")
    args = ap.parse_args()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.public:
        fetch_public()
        return check()
    if args.hub:
        fetch_from_hub()
        return check()
    if args.verify:
        return verify()

    missing = check()
    if not any((args.check, args.hub, args.verify)):
        print("  To fetch weights without an account: --public\n")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
