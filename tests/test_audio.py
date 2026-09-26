"""Tests for the audio front-end and the DSP helpers.

No microphone is opened: `open_stream` is monkeypatched. These cover the parts
that would otherwise only fail live, in front of a camera.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from snapdragoon import config
from snapdragoon.audio import capture
from snapdragoon.engines.onnx_engine import (
    log_mel_spectrogram,
    pad_or_trim,
    preprocess_image,
)


# -- silence detection ------------------------------------------------------
def test_rms_of_silence_is_zero(silence):
    assert capture.rms(silence) == 0.0


def test_rms_of_tone_is_positive(tone):
    assert capture.rms(tone) > 0.0


def test_silence_detected(silence):
    assert capture.is_silent(silence)


def test_tone_not_silent(tone):
    assert not capture.is_silent(tone)


def test_empty_window_is_silent():
    assert capture.is_silent(np.array([], dtype=np.float32))


def test_silence_threshold_is_respected(tone):
    # A threshold above the tone's RMS must classify it as silent.
    assert capture.is_silent(tone, threshold=capture.rms(tone) * 2)


# -- padding ----------------------------------------------------------------
def test_pad_or_trim_pads_short_input():
    out = pad_or_trim(np.ones(10, dtype=np.float32), 100)
    assert out.shape == (100,)
    assert out[:10].sum() == pytest.approx(10.0)
    assert out[10:].sum() == 0.0


def test_pad_or_trim_trims_long_input():
    out = pad_or_trim(np.ones(200, dtype=np.float32), 100)
    assert out.shape == (100,)


def test_pad_or_trim_exact_length():
    out = pad_or_trim(np.ones(50, dtype=np.float32), 50)
    assert out.shape == (50,)


# -- Whisper features -------------------------------------------------------
def test_log_mel_shape_matches_whisper_contract():
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    feats = log_mel_spectrogram(np.zeros(30 * config.SAMPLE_RATE, dtype=np.float32),
                                _mel_filterbank())
    # Whisper exports expect (batch, n_mels, 3000)
    assert feats.shape == (1, 80, 3000)
    assert feats.dtype == np.float32


def test_log_mel_pads_a_short_input_instead_of_reading_past_the_buffer():
    """Regression: a short input silently read out of bounds.

    The frame loop strides a fixed 3000x400 window over the audio buffer with
    no bounds check, so a 1 s input returned a correctly shaped array full of
    whatever memory followed it. `pytest -W error` surfaced it as an overflow
    warning inside the power spectrum.
    """
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    short = np.zeros(config.SAMPLE_RATE, dtype=np.float32)  # 1 s, not 30 s
    feats = log_mel_spectrogram(short, _mel_filterbank())
    assert feats.shape == (1, 80, 3000)
    assert np.isfinite(feats).all()


def test_log_mel_truncates_an_over_long_input():
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    long = np.zeros(45 * config.SAMPLE_RATE, dtype=np.float32)
    feats = log_mel_spectrogram(long, _mel_filterbank())
    assert feats.shape == (1, 80, 3000)


def test_log_mel_is_finite_for_silence():
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    feats = log_mel_spectrogram(np.zeros(30 * config.SAMPLE_RATE, dtype=np.float32),
                                _mel_filterbank())
    assert np.isfinite(feats).all(), "log of zero must be clamped, not -inf"


def test_log_mel_is_finite_for_loud_input():
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(30 * config.SAMPLE_RATE) * 0.5).astype(np.float32)
    feats = log_mel_spectrogram(loud, _mel_filterbank())
    assert np.isfinite(feats).all()


def test_log_mel_distinguishes_signal_from_silence():
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    fb = _mel_filterbank()
    quiet = log_mel_spectrogram(np.zeros(30 * config.SAMPLE_RATE, dtype=np.float32), fb)
    t = np.linspace(0, 30, 30 * config.SAMPLE_RATE, endpoint=False)
    loud = log_mel_spectrogram(
        (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), fb
    )
    assert not np.allclose(quiet, loud)


# -- image preprocessing ----------------------------------------------------
def test_preprocess_image_shape_and_dtype(frame):
    out = preprocess_image(frame)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32


def test_preprocess_image_handles_grayscale():
    out = preprocess_image(np.full((100, 100), 128, dtype=np.uint8))
    assert out.shape == (1, 3, 224, 224)


def test_preprocess_image_strips_alpha():
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    assert preprocess_image(rgba).shape == (1, 3, 224, 224)


def test_preprocess_image_handles_non_square(frame):
    assert preprocess_image(frame).shape == (1, 3, 224, 224)


def test_preprocess_image_is_deterministic(frame):
    a = preprocess_image(frame)
    b = preprocess_image(frame)
    assert np.array_equal(a, b), "preprocessing must be reproducible across machines"


def test_preprocess_image_normalises_range(frame):
    out = preprocess_image(frame)
    # ImageNet normalisation of a 0..255 input lands in a modest range.
    assert out.min() > -3.0 and out.max() < 3.0


# -- wav replay -------------------------------------------------------------
def test_iter_wav_yields_fixed_windows(wav_file):
    from snapdragoon.web.app import _iter_wav

    got = list(_iter_wav(str(wav_file)))
    want = int(config.CAPTURE_CHUNK_SECONDS * config.SAMPLE_RATE)
    assert got, "expected at least one window"
    assert all(len(w) == want for w in got)
    assert all(w.dtype == np.float32 for w in got)


def test_iter_wav_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "8k.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 100)

    from snapdragoon.web.app import _iter_wav

    with pytest.raises(ValueError, match="8000"):
        list(_iter_wav(str(path)))


def test_iter_wav_rejects_empty_path():
    from snapdragoon.web.app import _iter_wav

    with pytest.raises(FileNotFoundError):
        list(_iter_wav(""))


def test_open_stream_wraps_failure_with_actionable_message(monkeypatch):
    """A missing or denied device must not surface as a bare OSError."""
    import sys
    import types

    boom = types.ModuleType("sounddevice")

    def explode(*_args, **_kwargs):
        raise OSError("no default audio device")

    boom.RawInputStream = explode
    monkeypatch.setitem(sys.modules, "sounddevice", boom)

    with pytest.raises(RuntimeError) as exc:
        capture.open_stream()
    # The message has to tell the user what to actually do.
    assert "Microphone" in str(exc.value) or "audio input" in str(exc.value)


def test_log_mel_filterbank_shape():
    from snapdragoon.engines.onnx_engine import _mel_filterbank

    fb = _mel_filterbank()
    assert fb.shape == (80, 201), "80 mels over a 400-point rFFT"
    assert (fb >= 0).all()


class TestPaddingRatio:
    """The silence gate answers "is there anything"; this answers "how much"."""

    def test_full_signal_window_has_negligible_padding(self, tone):
        from snapdragoon.audio.capture import padding_ratio

        # A pure digital sine still lands on exactly 0.0 for roughly one sample
        # per cycle-pair, so this is "negligible" rather than "zero". The point
        # is that the ~2% of samples near zero crossings are not counted.
        assert padding_ratio(tone) < 1e-3

    def test_all_zero_window_is_all_padding(self, silence):
        from snapdragoon.audio.capture import padding_ratio

        assert padding_ratio(silence) == 1.0

    def test_empty_window_counts_as_padding(self):
        import numpy as np

        from snapdragoon.audio.capture import padding_ratio

        assert padding_ratio(np.zeros(0, dtype=np.float32)) == 1.0

    def test_very_quiet_signal_is_not_padding(self):
        """The reason for exact-zero comparison over an amplitude threshold.

        16-bit PCM's smallest non-zero magnitude is ~3e-5, so this is real audio
        and must never be mistaken for padding.
        """
        import numpy as np

        from snapdragoon.audio.capture import padding_ratio

        quiet = np.full(1600, 3e-5, dtype=np.float32)
        assert padding_ratio(quiet) == 0.0

    def test_mixed_window_reports_the_padding_fraction(self):
        import numpy as np

        from snapdragoon.audio.capture import padding_ratio

        w = np.concatenate([np.full(2500, 0.5, dtype=np.float32),
                            np.zeros(7500, dtype=np.float32)])
        assert padding_ratio(w) == pytest.approx(0.75, abs=0.01)


class TestContentSeconds:
    """The gate that decides whether a window is worth transcribing."""

    def test_five_second_window_has_five_seconds_of_content(self, tone):
        from snapdragoon.audio.capture import content_seconds

        assert content_seconds(tone, config.SAMPLE_RATE) == pytest.approx(1.0, abs=0.01)

    def test_pure_silence_has_no_content(self, silence):
        from snapdragoon.audio.capture import content_seconds

        assert content_seconds(silence, config.SAMPLE_RATE) == 0.0

    def test_empty_window_has_no_content(self):
        import numpy as np

        from snapdragoon.audio.capture import content_seconds

        assert content_seconds(np.zeros(0, dtype=np.float32), 16000) == 0.0

    def test_a_long_window_with_a_speech_burst_reports_only_the_burst(self):
        """Padding is excluded, so a 30 s window with 3 s of speech reports 3 s."""
        import numpy as np

        from snapdragoon.audio.capture import content_seconds

        burst = np.full(3 * config.SAMPLE_RATE, 0.3, dtype=np.float32)
        w = np.concatenate([burst, np.zeros(27 * config.SAMPLE_RATE, dtype=np.float32)])
        assert content_seconds(w, config.SAMPLE_RATE) == pytest.approx(3.0, abs=0.01)

    def test_very_quiet_signal_still_counts(self):
        """16-bit PCM's smallest non-zero magnitude is ~3e-5 -- that is audio."""
        import numpy as np

        from snapdragoon.audio.capture import content_seconds

        w = np.full(config.SAMPLE_RATE, 3e-5, dtype=np.float32)
        assert content_seconds(w, config.SAMPLE_RATE) == pytest.approx(1.0, abs=0.01)

    def test_zero_sample_rate_does_not_divide_by_zero(self, tone):
        from snapdragoon.audio.capture import content_seconds

        assert content_seconds(tone, 0) == pytest.approx(len(tone), abs=1.0)
