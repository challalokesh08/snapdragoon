"""Shared fixtures.

The suite is designed to run with only numpy installed: the demo engine has no
third-party dependencies, so `pytest` passes on a clean checkout. Tests that
need real models skip themselves rather than failing.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapdragoon import config  # noqa: E402
from snapdragoon.engines.demo import DemoEngine  # noqa: E402


@pytest.fixture
def engine() -> DemoEngine:
    return DemoEngine()


@pytest.fixture
def tone() -> np.ndarray:
    t = np.linspace(0, 1.0, config.SAMPLE_RATE, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(config.SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((240, 320, 3)) * 255).astype(np.uint8)


@pytest.fixture
def wav_file(tmp_path: Path, tone: np.ndarray) -> Path:
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes((tone * 32767).astype(np.int16).tobytes())
    return path


@pytest.fixture
def client():
    from snapdragoon.web.app import create_app

    app = create_app(DemoEngine())
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c
