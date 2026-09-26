"""Tests for `scripts/benchmark.py`.

The benchmark is the source of every performance number in the writeup, so the
thing worth protecting is not its arithmetic -- it is that it measures *inference*
and reports nothing when it cannot. Two real failures motivated these:

* It fed `np.zeros(...)`. Once the engine gained a short-window gate, that input
  was declined, so the harness began timing the gate and reporting 0.03 ms and a
  real-time factor of 166,666x. A number that absurd should have been the alarm,
  and it was not.
* It ran against an engine with no model loaded, where `transcribe` returns an
  error string immediately -- indistinguishable from a fast inference unless you
  look at the text.

Both produced a plausible-looking number and neither raised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapdragoon import config  # noqa: E402
from snapdragoon.engines.base import Transcript  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark  # noqa: E402


class _Args:
    def __init__(self, runs=3, warmup=1):
        self.runs = runs
        self.warmup = warmup
        self.vision_only = False
        self.asr_only = True
        self.backend = "onnx"
        self.json = None


class _Engine:
    """Stands in for a real engine. Records what it was asked to transcribe."""

    def __init__(self, has_asr=True, usable=True, text="a caption"):
        self.has_asr = has_asr
        self._usable = usable
        self._text = text
        self.windows: list[np.ndarray] = []

    def transcribe(self, audio, sample_rate):
        self.windows.append(np.asarray(audio))
        return Transcript(text=self._text, window_seconds=1.0, latency_ms=1.0,
                          skip_reason=None if self._usable else "short_window")


def test_benchmark_uses_real_audio_not_silence(capsys):
    """A silent input is declined by the engine, so it measures the gate."""
    engine = _Engine()
    benchmark._benchmark_asr(engine, _Args(), {})

    assert engine.windows, "nothing was transcribed"
    for w in engine.windows:
        assert w.size == int(config.CAPTURE_CHUNK_SECONDS * config.SAMPLE_RATE)
        assert np.count_nonzero(w != 0.0) > 0, (
            "the benchmark fed silence, so it is timing the short-window gate "
            "rather than the model"
        )
        assert w.size == w[:config.SAMPLE_RATE * 5].size, "window was truncated"


def test_benchmark_reports_no_asr_number_when_no_model_is_loaded(capsys):
    engine = _Engine(has_asr=False)
    results: dict = {}
    benchmark._benchmark_asr(engine, _Args(), {})

    out = capsys.readouterr().out
    assert "no speech model" in out
    assert "asr" not in results, "a figure was reported for an engine with no model"
    assert not engine.windows, "it transcribed anyway"


def test_benchmark_reports_no_number_when_every_window_is_refused(capsys):
    """The case that produced 0.03 ms: usable=False on every single window."""
    engine = _Engine(usable=False)
    results: dict = {}
    benchmark._benchmark_asr(engine, _Args(runs=3, warmup=1), {})

    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "asr" not in results


def test_benchmark_treats_a_bracketed_error_string_as_a_refusal(capsys):
    """`[no model] ...` is marked usable, so the flag check alone misses it."""
    engine = _Engine(text="[no model] Speech recognition unavailable")
    results: dict = {}
    benchmark._benchmark_asr(engine, _Args(runs=3, warmup=1), {})

    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "asr" not in results


def test_benchmark_warns_when_only_some_windows_are_refused(capsys):
    engine = _Engine()
    calls = {"n": 0}

    def _transcribe(audio, sr):
        calls["n"] += 1
        usable = calls["n"] > 2
        return Transcript(text="x", window_seconds=1.0, latency_ms=1.0,
                          skip_reason=None if usable else "short_window")

    engine.transcribe = _transcribe
    results: dict = {}
    benchmark._benchmark_asr(engine, _Args(runs=3, warmup=1), {})

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "NOT" in out, "the warning must say the numbers are not inference times"


def test_benchmark_records_its_provenance_in_the_json(capsys):
    """A reader must be able to see what was measured, not just how fast."""
    engine = _Engine()
    results: dict = {}
    benchmark._benchmark_asr(engine, _Args(), results)

    assert results["asr"]["source"] == "assets/reference-speech.wav"
    assert results["asr"]["window_seconds"] == pytest.approx(
        config.CAPTURE_CHUNK_SECONDS, abs=0.01)


def test_benchmark_refuses_to_divide_by_a_sub_resolution_median(capsys):
    """A median that rounds to 0.0 must not become a real-time factor of 0.

    Dividing by it yields 0, and comparing 0 < 1 then prints "SLOWER than real
    time" for the fastest possible result -- exactly backwards. A stub engine
    that returns instantly is the easiest way to reach that branch, which is
    also the easiest way to test that it is handled.
    """
    results: dict = {}
    benchmark._benchmark_asr(_Engine(), _Args(), results)

    out = capsys.readouterr().out
    assert results["asr"]["median_ms"] == 0.0, "stub should be below resolution"
    assert "realtime_factor" not in results["asr"], (
        "a factor was invented from a zero median"
    )
    assert "below this harness" in out
    assert "SLOWER" not in out, "an instant result was reported as slower than real time"
