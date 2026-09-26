"""Zero-dependency fallback engine.

This exists for one reason: a deadline. If the ONNX runtime will not install, or
the AI Hub download needs credentials you do not have, you still have something
that runs, so the demo never dies on stage.

It performs **no inference**. Every string it returns is synthetic and is tagged
``synthetic=True``; the UI labels it as such. Never let this masquerade as a
real result -- the honest framing in your submission is "model inference path
unavailable in this environment, synthetic placeholder shown".
"""

from __future__ import annotations

import time

import numpy as np

from .base import Detection, Engine, Transcript

# Phrases chosen to exercise the caption UI: short, long, and punctuation-heavy.
_SYNTHETIC_LINES = [
    "This is a synthetic placeholder caption.",
    "Snapdragoon is running on the demo engine because no model runtime was detected.",
    "Install the onnxruntime extra, then fetch the models to enable real speech to text.",
    "On a Snapdragon powered PC the same pipeline runs on the neural processing unit.",
    "Live captions should be announced politely, without interrupting the user.",
]


class DemoEngine(Engine):
    name = "demo"
    device_description = "Synthetic placeholder - no model inference is performed"
    is_neural_accelerated = False

    def __init__(self) -> None:
        self._asr_index = 0
        self._vision_index = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        started = time.perf_counter()
        window_seconds = len(audio) / float(sample_rate or 1)
        text = _SYNTHETIC_LINES[self._asr_index % len(_SYNTHETIC_LINES)]
        self._asr_index += 1
        return Transcript(
            text=f"[synthetic] {text}",
            window_seconds=window_seconds,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            confidence=None,
        )

    def classify(self, frame: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        # Derive a stable pseudo-score from cheap frame statistics so successive
        # calls differ slightly and the UI's confidence bars are exercised.
        signal = float(np.mean(frame)) / 255.0 if frame.size else 0.0
        base = 0.55 + (signal * 0.4)
        labels = ["desktop", "person", "cup", "notebook", "window"]
        picks = [(labels[(self._vision_index + i) % len(labels)],
                  max(0.05, min(0.99, base - i * 0.17)))
                 for i in range(3)]
        self._vision_index += 1
        latency = (time.perf_counter() - started) * 1000.0
        return [Detection(label=l, score=s, latency_ms=latency) for l, s in picks]
