"""Backend selection.

``auto`` prefers, in order: a Snapdragon NPU, ONNX Runtime, then the synthetic
demo engine. The chosen backend is always reported honestly through
``/api/status`` and rendered in the UI, because a demo that misrepresents where
inference is running is worse than no demo.
"""

from __future__ import annotations

import logging

from .. import config
from .base import Detection, Engine, Transcript, format_detections

log = logging.getLogger(__name__)

__all__ = [
    "Detection",
    "Engine",
    "Transcript",
    "format_detections",
    "get_engine",
]


def get_engine(preference: str | None = None) -> Engine:
    choice = (preference or config.DEFAULT_BACKEND or "auto").lower()

    if choice == "demo":
        from .demo import DemoEngine

        return DemoEngine()

    if choice == "qualcomm":
        from .qualcomm import QualcommEngine

        return QualcommEngine()  # raises QualcommUnavailable if not on Snapdragon

    if choice == "onnx":
        from .onnx_engine import OnnxEngine

        return OnnxEngine()

    # auto
    try:
        from .qualcomm import QualcommEngine

        engine = QualcommEngine()
        log.info("Using Qualcomm NPU backend")
        return engine
    except Exception as exc:  # noqa: BLE001
        log.info("NPU backend unavailable (%s); trying ONNX", exc)

    try:
        from .onnx_engine import OnnxEngine

        engine = OnnxEngine()
        if engine.has_asr or engine.has_classifier:
            log.info("Using ONNX Runtime backend")
            return engine
        log.info("ONNX Runtime present but no models found; using demo engine")
    except ImportError as exc:
        log.info("onnxruntime not installed (%s); using demo engine", exc)

    from .demo import DemoEngine

    return DemoEngine()
