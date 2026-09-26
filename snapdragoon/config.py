"""Central configuration for Snapdragoon.

Every path and tunable lives here so the same code runs unchanged on a Mac
(CPU/ONNX) and on a Snapdragon-powered HP PC (NPU via Qualcomm AI Hub).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
MODELS_DIR = Path(os.environ.get("SNAPDRAGOON_MODELS", PROJECT_ROOT / "models"))

# Audio / capture
SAMPLE_RATE = 16_000          # Whisper-family models expect 16 kHz mono
CAPTURE_CHUNK_SECONDS = 5.0  # length of each window fed to the ASR model
CAPTURE_BLOCK_MS = 100        # PortAudio block size for low capture latency
CAPTURE_CHANNELS = 1

# Which backend to use. "auto" prefers a real NPU, then ONNX, then the
# zero-dependency demo engine. See snapdragoon.engines.get_engine.
DEFAULT_BACKEND = os.environ.get("SNAPDRAGOON_BACKEND", "auto")

# Web server
HOST = os.environ.get("SNAPDRAGOON_HOST", "127.0.0.1")
PORT = int(os.environ.get("SNAPDRAGOON_PORT", "8765"))

# Accessibility
CAPTION_CHAR_LIMIT = 400
# Longest caption history retained in the live region before it is truncated.
CAPTION_HISTORY_LIMIT = 12


@dataclass
class ModelSpec:
    """A single AI Hub model's on-disk contract.

    ``task`` is either ``"asr"`` (speech to text) or ``"classification"``.
    ``filename`` is the expected artefact inside ``MODELS_DIR``; the ONNX and
    Qualcomm engines both key off this so a model swap needs no code change.
    """

    task: str
    filename: str
    display_name: str
    # Provenance: which Qualcomm AI Hub model this came from.
    hub_model: str = ""
    # Minimum viable input; the engines clamp real audio/frames to this.
    min_seconds: float = 1.0
    labels_path: str | None = None
    # True when ``filename`` names a *directory* of graphs. Whisper needs this:
    # its common ONNX exports are an encoder plus a decoder plus a config, not a
    # single file, and several of the graphs are interchangeable.
    is_dir: bool = False
    # Free-form notes surfaced in /api/status and the docs.
    notes: str = ""
    extra: dict = field(default_factory=dict)


# Model registry. Populated by scripts/fetch_models.py --hub on a machine with
# network access, or by hand from the AI Hub download page.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "asr": ModelSpec(
        task="asr",
        filename="whisper-tiny.en",
        display_name="Whisper Tiny (English)",
        hub_model="whisper-tiny.en",
        min_seconds=1.0,
        is_dir=True,
        notes="Speech-to-text. Qualcomm AI Hub ONNX export; same graph runs on "
              "the Snapdragon NPU via QNN/LiteRT.",
    ),
    "classifier": ModelSpec(
        task="classification",
        filename="mobilenet_v2.onnx",
        display_name="MobileNet V2 (ImageNet-1k)",
        hub_model="mobilenet_v2_imagenet",
        min_seconds=0.0,
        labels_path="imagenet_classes.txt",
        notes="Scene/object classification, 1000 ImageNet classes.",
    ),
}


def model_path(spec: ModelSpec) -> Path:
    return MODELS_DIR / spec.filename


def has_model(spec: ModelSpec) -> bool:
    """True when a model's artefacts are actually present on disk."""
    path = model_path(spec)
    if spec.is_dir:
        return path.is_dir() and any(path.glob("*.onnx"))
    return path.is_file() and path.stat().st_size > 0
