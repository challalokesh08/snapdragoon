"""Tests for the engine contract.

The point of these is the `Engine` boundary: if all three backends honour this
contract, the Mac and Snapdragon builds are interchangeable, which is the whole
architectural claim of the project.
"""

from __future__ import annotations

import numpy as np
import pytest

from snapdragoon import config
from snapdragoon.engines.base import Detection, Engine, format_detections
from snapdragoon.engines.demo import DemoEngine


def test_demo_transcribe_returns_transcript(engine, tone):
    result = engine.transcribe(tone, config.SAMPLE_RATE)
    assert result.text
    assert result.window_seconds == pytest.approx(1.0, abs=0.01)
    assert result.latency_ms >= 0.0
    # Must be self-identifying so a synthetic result can never pass as real.
    assert "synthetic" in result.text.lower()


def test_demo_transcribe_varies(engine, tone):
    first = engine.transcribe(tone, config.SAMPLE_RATE).text
    second = engine.transcribe(tone, config.SAMPLE_RATE).text
    assert first != second


def test_demo_classify_returns_sorted_detections(engine, frame):
    dets = engine.classify(frame)
    assert 1 <= len(dets) <= 3
    assert all(isinstance(d, Detection) for d in dets)
    scores = [d.score for d in dets]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_classify_handles_empty_frame(engine):
    dets = engine.classify(np.zeros((0, 0, 3), dtype=np.uint8))
    assert isinstance(dets, list)


def test_describe_reports_npu_status_honestly(engine):
    info = engine.describe()
    assert info["engine"] == "demo"
    assert info["neural_accelerated"] is False
    assert "synthetic" in info["device"].lower()
    assert "asr" in info["models"] and "classifier" in info["models"]


def test_describe_lists_model_availability(engine):
    info = engine.describe()
    for spec in info["models"].values():
        assert "available" in spec
        assert "hub_model" in spec


def test_warmup_is_safe_to_call(engine, tone, frame):
    engine.warmup()
    assert engine.transcribe(tone, config.SAMPLE_RATE).text
    assert engine.classify(frame)


# -- format_detections: written for TTS, so these matter -------------------
def test_format_detections_empty() -> None:
    assert format_detections([]) == "No objects recognised in this frame."


def test_format_detections_single_is_not_a_list_awkwardness() -> None:
    out = format_detections([Detection("cup", 0.9, 1.0)])
    assert out.startswith("Likely in view: cup,")
    assert " and " not in out


def test_format_detections_uses_and_before_last() -> None:
    out = format_detections([
        Detection("cup", 0.9, 1.0),
        Detection("book", 0.8, 1.0),
        Detection("pen", 0.7, 1.0),
    ])
    assert out.count(", and ") == 1
    assert out.endswith(".")


def test_format_detections_respects_limit() -> None:
    dets = [Detection(f"object{i}", 0.9 - i / 100, 1.0) for i in range(10)]
    out = format_detections(dets, limit=2)
    assert out.count("percent confidence") == 2


def test_format_detections_sanitises_imagenet_synonyms() -> None:
    """ImageNet labels are comma-separated synonym lists and must not leak
    into the spoken sentence, or the list structure becomes unintelligible."""
    out = format_detections([Detection("jersey, T-shirt, tee shirt, teeshirt", 0.8, 1.0)])
    assert out == "Likely in view: jersey, 80 percent confidence."


def test_format_detections_strips_brackets() -> None:
    """Brackets and parens force awkward pauses in most screen readers."""
    out = format_detections([Detection("crane (machine)", 0.7, 1.0),
                             Detection("spotlight [stage]", 0.6, 1.0)])
    for ch in "()[]{}":
        assert ch not in out
    assert "crane machine" in out
    assert "spotlight stage" in out


def test_format_detections_handles_empty_label() -> None:
    out = format_detections([Detection("", 0.5, 1.0)])
    assert "unlabelled object" in out


class QualcommStub(Engine):
    """`QualcommEngine` cannot be constructed off-Snapdragon, by design.

    This exercises the same `Engine` contract for the schema assertion without
    pretending that hardware is present.
    """

    name = "qualcomm-npu"
    device_description = "test stub"
    is_neural_accelerated = True

    def transcribe(self, audio, sample_rate):
        raise NotImplementedError

    def classify(self, frame):
        raise NotImplementedError



def test_status_schema_is_identical_on_every_backend():
    """`GET /api/status` must not change shape with the engine.

    It did: `asr_loaded` and `classifier_loaded` were added by the ONNX engine
    only, so a client had to know which backend it had been handed before it
    could ask whether captions would work. A judge hitting the demo engine got a
    different response, and the natural defensive response is to treat a missing
    key as "unknown" rather than "no" — which is exactly the wrong default.
    """
    from snapdragoon.engines.base import Engine

    required = {
        "engine", "device", "neural_accelerated",
        "asr_loaded", "classifier_loaded", "models",
    }

    described = {e.name: e.describe() for e in (DemoEngine(), QualcommStub())}
    for name, info in described.items():
        missing = required - set(info)
        assert not missing, f"{name} is missing {sorted(missing)} from /api/status"

    # And the two capability flags are always real booleans, never absent or null,
    # so `info["asr_loaded"]` is safe to read on any backend.
    for name, info in described.items():
        assert isinstance(info["asr_loaded"], bool), name
        assert isinstance(info["classifier_loaded"], bool), name


def test_demo_engine_reports_that_it_loaded_no_models():
    """The honest answer is `false`, not a missing key.

    A missing key reads as "unknown" to a client and invites a guess. The demo
    engine knows perfectly well that it has no weights behind it.
    """
    info = DemoEngine().describe()
    assert info["asr_loaded"] is False
    assert info["classifier_loaded"] is False


def test_base_engine_defaults_the_capability_flags_to_false():
    from snapdragoon.engines.base import Engine

    assert Engine.has_asr is False
    assert Engine.has_classifier is False
