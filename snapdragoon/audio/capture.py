"""Microphone / audio capture.

Windows-on-Snapdragon note: ``sounddevice`` needs no special driver work for a
standard USB or internal mic, but if you want to caption *system* audio (rather
than the room) on Windows, ``sounddevice`` cannot loop it out of the box. Use
WASAPI loopback via ``soundcard`` or ``pycaw`` and add it in
:func:`open_loopback`. The rest of the pipeline is unchanged either way, because
everything downstream only sees a numpy float32 array.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from .. import config


def open_stream(device: int | None = None) -> "object":
    """Open a mono 16 kHz input stream.

    Raises RuntimeError with an actionable message if no input device exists,
    because "cannot open audio" is otherwise a baffling blank UI.
    """
    import sounddevice as sd

    try:
        return sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=int(config.SAMPLE_RATE * config.CAPTURE_BLOCK_MS / 1000),
            dtype="float32",
            channels=config.CAPTURE_CHANNELS,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not open an audio input device. On macOS grant Microphone "
            "access in System Settings > Privacy & Security > Microphone, then "
            "restart the server."
        ) from exc


def iter_windows(
    stream: "object",
    seconds: float = config.CAPTURE_CHUNK_SECONDS,
) -> Iterator[np.ndarray]:
    """Yield fixed-length mono float32 windows of captured audio.

    Windows are emitted on a fixed cadence rather than on silence detection.
    That is a deliberate simplification: VAD would cut latency and cost an extra
    dependency, and predictable cadence is easier to caption reliably.
    """
    import sounddevice as sd

    want = int(seconds * config.SAMPLE_RATE)
    buffer = np.zeros(want, dtype=np.float32)
    filled = 0
    started = time.perf_counter()

    with stream:
        while True:
            indata, _overflowed = stream.read(config.SAMPLE_RATE // 10)
            chunk = np.asarray(indata, dtype=np.float32).reshape(-1)
            buffer[filled:filled + len(chunk)] = chunk
            filled += len(chunk)
            if filled >= want:
                window = buffer.copy()
                buffer = np.zeros(want, dtype=np.float32)
                filled = 0
                yield window
            _ = started  # reserved for a future streaming/latency readout

    _ = sd  # keep the import meaningful for type checkers


def rms(window: np.ndarray) -> float:
    """Root-mean-square level of a window, used to skip silent input."""
    if window.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))


def is_silent(window: np.ndarray, threshold: float = 1e-3) -> bool:
    return rms(window) < threshold


def padding_ratio(window: np.ndarray) -> float:
    """Fraction of samples that are *exactly* zero.

    An amplitude threshold is the obvious way to ask "how much of this window is
    padding?" and the wrong one: it has to be tuned, and any real signal spends
    some fraction of its samples near zero anyway (a 220 Hz sine crosses zero
    twice per cycle), so a quiet-but-real window is indistinguishable from
    padding. Exact zeros sidestep that -- padding is constructed with
    ``np.zeros``, and 16-bit PCM's smallest non-zero magnitude is ~3e-5, so
    ``== 0.0`` cannot misfire on real audio.
    """
    if window.size == 0:
        return 1.0
    return float(np.count_nonzero(window == 0.0) / window.size)


def content_seconds(window: np.ndarray, sample_rate: int) -> float:
    """How much real audio a window contains, ignoring zero padding.

    This is the quantity that decides whether a window is worth transcribing.
    Padding *per se* is not the problem: a 5 s window padded to Whisper's 30 s
    context transcribes correctly. Too little audio is the problem -- a 1 s
    window padded the same way returns a hallucinated proper noun.
    """
    if window.size == 0:
        return 0.0
    return float(np.count_nonzero(window != 0.0) / max(sample_rate, 1))


def open_loopback(seconds: float = config.CAPTURE_CHUNK_SECONDS):
    """Capture system audio instead of the microphone (Windows only).

    Returns a generator of float32 windows, same shape as :func:`iter_windows`,
    so the ASR stage is identical. Falls back to raising a clear error off
    Windows.
    """
    import sys

    if sys.platform != "win32":
        raise RuntimeError(
            "System-audio loopback is implemented for Windows only (WASAPI). "
            "On macOS use the microphone, or route audio with BlackHole."
        )

    import soundcard as sc  # type: ignore

    loopback = sc.get_microphone(id="loopback", include_loopback=True)
    with loopback.recorder(samplerate=config.SAMPLE_RATE, channels=1) as rec:
        while True:
            data = rec.record(numframes=int(seconds * config.SAMPLE_RATE))
            yield np.asarray(data[:, 0], dtype=np.float32)
