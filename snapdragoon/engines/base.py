"""Execution backend abstraction.

The whole point of Snapdragoon is that one pipeline drives three very different
runtimes:

* :class:`DemoEngine`      -- zero dependencies, always works. Guarantees a demo.
* :class:`OnnxEngine`      -- real models on the CPU. What you run on a Mac.
* :class:`QualcommEngine`  -- real models on the Snapdragon NPU via Qualcomm AI
  Hub / QNN / LiteRT. What you run on the HP Omnibook.

Everything above this layer (capture, vision, web UI, accessibility) is written
against :class:`Engine` only, so swapping runtimes changes nothing else.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import config

# ImageNet label files are ``n02111889 Samoyed, Samoyede`` -- a WordNet id, a
# space, then the human-readable synset. The id is noise in every context we
# surface the label in, so it is stripped once at load time rather than
# repeatedly at every use.
_WNID = re.compile(r"^n\d{8}\s+")


def load_labels(path: Path | str | None) -> list[str]:
    """Read an ImageNet label file, stripping WordNet ids.

    Shared by every engine so the on-disk format is understood in exactly one
    place. Returns ``[]`` for a missing file, which callers treat as "unknown
    labels" and surface rather than guess.
    """
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = _WNID.sub("", line.strip())
        if line:
            labels.append(line)
    return labels


@dataclass
class Transcript:
    """A caption result for one audio window."""

    text: str
    # Seconds of audio that produced this.
    window_seconds: float
    # Wall-clock inference time in ms, for the latency readout.
    latency_ms: float
    # Rough confidence proxy in [0, 1]. None when the backend cannot supply one.
    confidence: float | None = None
    # Set when the window was not transcribed and `text` must not be shown.
    #
    # Whisper is trained on a full 30 s of contiguous audio. A shorter window is
    # zero-padded up to 30 s, and the model answers a mostly-empty window with a
    # confident hallucination -- a stray proper noun such as "Godfrey." -- rather
    # than with nothing. So the engine reports the window as unusable and the
    # caller skips it. Only the engine can make this call, because only the
    # engine knows what padding it added.
    skip_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.skip_reason is None


@dataclass
class Detection:
    """A single classification result for one frame."""

    label: str
    score: float
    latency_ms: float


class Engine(abc.ABC):
    """Common interface every backend implements."""

    #: Short identifier surfaced to the UI, e.g. "onnx" or "qualcomm-npu".
    name: str = "base"

    #: Human-readable description of where inference is actually running.
    #: The UI displays this verbatim, so it must be honest.
    device_description: str = "unknown device"

    #: True when inference is executing on a Snapdragon NPU.
    is_neural_accelerated: bool = False

    #: Whether each capability has a model behind it right now. Part of the
    #: contract rather than an ONNX-engine detail, so that ``GET /api/status``
    #: has the same shape on every backend. A client should not have to know
    #: which engine it got before it can ask whether captions will work.
    has_asr: bool = False
    has_classifier: bool = False

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Transcribe mono float32 audio in ``[-1, 1]``."""

    @abc.abstractmethod
    def classify(self, frame: np.ndarray) -> list[Detection]:
        """Classify a single RGB image frame (``uint8`` HxWx3).

        Returns detections sorted by descending score.
        """

    def describe(self) -> dict:
        """Machine-readable status, exposed at ``GET /api/status``."""
        return {
            "engine": self.name,
            "device": self.device_description,
            "neural_accelerated": self.is_neural_accelerated,
            "asr_loaded": self.has_asr,
            "classifier_loaded": self.has_classifier,
            "models": {
                key: {
                    "display_name": spec.display_name,
                    "available": config.has_model(spec),
                    "hub_model": spec.hub_model,
                    "notes": spec.notes,
                }
                for key, spec in config.MODEL_REGISTRY.items()
            },
        }

    def warmup(self) -> None:
        """Optional. Run a tiny input through the model to absorb first-call
        allocation and kernel-selection cost, so the first real caption is not
        anomalously slow. Safe to no-op."""
        return None


def _speakable_label(label: str) -> str:
    """Reduce a model label to something a screen reader says cleanly.

    ImageNet synsets are comma-separated synonym lists -- "jersey, T-shirt, tee
    shirt, teeshirt" -- which would otherwise shred the "x, y, and z" sentence
    structure we build below and make the result impossible to follow by ear.
    Keep the primary synonym, lowercase it, and drop bracket runs.
    """
    primary = label.split(",")[0].strip()
    for ch in "()[]{}":
        primary = primary.replace(ch, "")
    primary = " ".join(primary.split())
    return (primary or "unlabelled object").lower()


def format_detections(detections: Sequence[Detection], limit: int = 3) -> str:
    """Turn detections into a spoken-friendly sentence.

    Written for a screen reader first: no parentheses, no brackets, no
    punctuation runs that a TTS engine has to stumble over.
    """
    if not detections:
        return "No objects recognised in this frame."
    top = list(detections[:limit])
    parts = [f"{_speakable_label(d.label)}, {round(d.score * 100)} percent confidence"
             for d in top]
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"Likely in view: {body}."
