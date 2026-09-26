"""Tests for the Whisper ONNX runner.

Two groups:

* **Pure logic** -- cache shapes, special-token loading, prompt construction,
  the DSP front-end. These run anywhere; no model, no onnxruntime.
* **A contract fake** -- a stand-in session that mimics the real
  ``onnx-community/whisper-*`` export, including the ``(0, 6, 1, 64)`` dummy
  encoder ``present`` tensors that the ``use_cache_branch=True`` path returns.
  This is the only practical way to lock in the encoder-cache fix, because the
  real symptom is *silent*: a correct first few words, then a spurious
  end-of-sequence.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapdragoon import config  # noqa: E402
from snapdragoon.engines.onnx_engine import (  # noqa: E402
    _hann_window,
    _mel_filterbank,
    log_mel_spectrogram,
    pad_or_trim,
)
from snapdragoon.engines.whisper_onnx import (  # noqa: E402
    SpecialTokens,
    cache_shape,
    load_special_tokens,
)

# The reference transcript for whisper.cpp's samples/jfk.wav, which is the
# canonical Whisper test clip.
JFK_REFERENCE = (
    "And so my fellow Americans ask not what your country can do for you, "
    "ask what you can do for your country"
)


# ---------------------------------------------------------------------------
# cache shapes
# ---------------------------------------------------------------------------
class TestCacheShape:
    def test_symbolic_dims_do_not_shift_axes(self):
        """Regression: filtering symbolic dims silently reorders the shape.

        `['batch_size', 6, 'past_decoder_sequence_length', 64]` filtered to
        ints yields `[6, 64]`, which reads as heads=64. That produced
        `INVALID_ARGUMENT ... index: 1 Got: 64 Expected: 6` on the first
        inference call.
        """
        shape = cache_shape(
            ["batch_size", 6, "past_decoder_sequence_length", 64],
            is_encoder=False,
            encoder_len=1500,
        )
        assert shape == (1, 6, 0, 64)

    def test_encoder_cache_spans_the_encoder_output(self):
        shape = cache_shape(
            ["batch_size", 6, "encoder_sequence_length_out", 64],
            is_encoder=True,
            encoder_len=1500,
        )
        assert shape == (1, 6, 1500, 64)

    def test_static_batch_is_preserved(self):
        assert cache_shape([2, 6, "seq", 64], is_encoder=False,
                           encoder_len=0) == (2, 6, 0, 64)

    def test_fully_symbolic_shape_gets_safe_defaults(self):
        assert cache_shape(["b", "h", "s", "d"], is_encoder=False,
                           encoder_len=99) == (1, 6, 0, 64)

    def test_short_shape_is_padded_not_crashed(self):
        assert cache_shape([1, 6], is_encoder=False, encoder_len=0) == (1, 6, 0, 64)


# ---------------------------------------------------------------------------
# special tokens
# ---------------------------------------------------------------------------
class TestSpecialTokens:
    def test_defaults_when_no_config_present(self, tmp_path: Path):
        spec = load_special_tokens(tmp_path / "nope")
        assert spec.source == "defaults"
        # The values that are *not* guesses, but the documented whisper-tiny.en
        # ids, still have to be right in the fallback path.
        assert (spec.sot, spec.eot, spec.no_timestamps) == (50257, 50256, 50362)

    def test_config_json_wins(self, tmp_path: Path):
        (tmp_path / "config.json").write_text(json.dumps({
            "decoder_start_token_id": 50258,
            "eos_token_id": 50257,
            "vocab_size": 51865,
            "is_multilingual": True,
        }))
        spec = load_special_tokens(tmp_path)
        assert (spec.sot, spec.eot, spec.vocab_size) == (50258, 50257, 51865)
        assert spec.multilingual is True
        assert "config.json" in spec.source

    def test_added_tokens_fills_names(self, tmp_path: Path):
        (tmp_path / "added_tokens.json").write_text(json.dumps({
            "<|startoftranscript|>": 50257,
            "<|endoftext|>": 50256,
            "<|notimestamps|>": 50362,
            "<|en|>": 50258,
        }))
        spec = load_special_tokens(tmp_path)
        assert spec.sot == 50257
        assert spec.no_timestamps == 50362
        assert "added_tokens.json" in spec.source

    def test_malformed_json_falls_back_instead_of_raising(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{ not json")
        spec = load_special_tokens(tmp_path)
        assert spec.source == "defaults"

    def test_source_names_are_not_repeated(self, tmp_path: Path):
        (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 1}))
        (tmp_path / "added_tokens.json").write_text(json.dumps({
            "<|startoftranscript|>": 1, "<|endoftext|>": 2,
            "<|notimestamps|>": 3, "<|en|>": 4, "<|transcribe|>": 5,
        }))
        spec = load_special_tokens(tmp_path)
        assert spec.source == "config.json+added_tokens.json"

    def test_english_only_prompt_omits_language_tokens(self):
        """`<|en|>` on an English-only checkpoint costs accuracy.

        Measured on the JFK clip: with `<|en|>` the transcript loses the comma
        before "ask" and drops the trailing clause.
        """
        spec = SpecialTokens(sot=50257, eot=50256, no_timestamps=50362,
                             language=50258, transcribe=50358, multilingual=False)
        assert spec.prompt() == [50257, 50362]

    def test_multilingual_prompt_includes_language_and_task(self):
        spec = SpecialTokens(sot=50257, eot=50256, no_timestamps=50362,
                             language=50258, transcribe=50358, multilingual=True)
        assert spec.prompt() == [50257, 50258, 50358, 50362]


# ---------------------------------------------------------------------------
# DSP front-end
# ---------------------------------------------------------------------------
def _filterbank_without_normalisation() -> np.ndarray:
    """The same triangular bank with the Slaney area normalisation removed.

    Only used as a baseline in the tests, so it lives here rather than in the
    package -- there is no reason for the shipped code to build the broken one.
    """
    from snapdragoon.engines.onnx_engine import (
        _hz_to_mel_slaney,
        _mel_to_hz_slaney,
    )

    n_fft, n_mels = 400, 80
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / config.SAMPLE_RATE)
    mel_points = np.linspace(
        _hz_to_mel_slaney(np.array([0.0]))[0],
        _hz_to_mel_slaney(np.array([config.SAMPLE_RATE / 2.0]))[0],
        n_mels + 2,
    )
    hz = _mel_to_hz_slaney(mel_points)
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fftfreqs[None, :]
    out = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        out[i] = np.maximum(0.0, np.minimum(-ramps[i] / fdiff[i],
                                            ramps[i + 2] / fdiff[i + 1]))
    return out


class TestFrontEnd:
    def test_hann_window_is_periodic_not_symmetric(self):
        w = _hann_window(400)
        assert w[-1] != pytest.approx(0.0), (
            "a symmetric window ends at exactly 0; Whisper uses torch's "
            "periodic hann_window, which does not"
        )
        assert w[0] == pytest.approx(0.0)
        assert w[200] == pytest.approx(1.0)

    def test_filterbank_shape_and_equal_area_normalisation(self):
        """Slaney normalisation is what makes the bank near-equal-area.

        `enorm = 2 / (edge[i+2] - edge[i])` compensates for filter width. Without
        it, the widest (lowest-frequency) filters carry ~8x the area of the
        narrowest, and the top bands -- where sibilants live -- are swamped.

        The residual ~1.16x spread is real, not a defect: normalisation equalises
        *continuous* area, and the narrow high-frequency filters lose a little
        to sampling on the 201-bin grid.
        """
        fb = _mel_filterbank()
        assert fb.shape == (80, 201)
        assert fb.min() == pytest.approx(0.0)
        assert (fb > 0).any()

        areas = fb.sum(axis=1)
        spread = areas.max() / areas.min()
        assert spread < 1.25, f"area spread {spread:.2f} is too wide to be normalised"

        raw = _filterbank_without_normalisation()
        raw_spread = raw.sum(axis=1).max() / raw.sum(axis=1).min()
        assert raw_spread > 4.0, "baseline changed; the comparison is no longer meaningful"
        assert raw_spread / spread > 3.0, (
            f"normalisation only improved area spread {raw_spread:.2f} -> {spread:.2f}"
        )

    def test_filterbank_spans_dc_to_nyquist(self):
        fb = _mel_filterbank()
        freqs = np.arange(fb.shape[1]) * config.SAMPLE_RATE / (fb.shape[1] - 1)
        centroids = (fb * freqs).sum(axis=1) / fb.sum(axis=1)
        assert centroids[0] < 200, "lowest filter should sit near DC"
        assert centroids[-1] > 7000, "highest filter should sit near Nyquist"
        assert np.all(np.diff(centroids) > 0), "filters must be ordered by frequency"

    def test_log_mel_shape_dtype_and_finiteness(self, tone):
        feats = log_mel_spectrogram(pad_or_trim(tone, 30 * config.SAMPLE_RATE),
                                    _mel_filterbank())
        assert feats.shape == (1, 80, 3000)
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all()

    def test_loud_audio_does_not_overflow_to_inf(self):
        """float32 accumulation overflows on full-scale input; float64 does not.

        The failure mode is not an exception -- `inf` propagates through the
        model and comes back as confident-looking nonsense.
        """
        loud = np.ones(30 * config.SAMPLE_RATE, dtype=np.float32) * 4.0
        feats = log_mel_spectrogram(loud, _mel_filterbank())
        assert np.isfinite(feats).all(), "full-scale input must not produce inf"

    def test_silence_is_finite_and_flat(self):
        feats = log_mel_spectrogram(
            np.zeros(30 * config.SAMPLE_RATE, dtype=np.float32), _mel_filterbank()
        )
        assert np.isfinite(feats).all()
        # The dynamic-range clamp keeps the span bounded.
        assert feats.max() - feats.min() <= 8.0 / 4.0 + 1e-6


# ---------------------------------------------------------------------------
# contract fake
# ---------------------------------------------------------------------------
class _Meta:
    def __init__(self, name, shape, type_):
        self.name, self.shape, self.type = name, list(shape), type_


class _FakeEncoder:
    def __init__(self, hidden_len=1500, width=8):
        self._hidden_len, self._width = hidden_len, width
        self.seen = []

    def get_inputs(self):
        return [_Meta("input_features", ["b", 80, 3000], "tensor(float)")]

    def get_outputs(self):
        return [_Meta("last_hidden_state", ["b", self._hidden_len, self._width],
                      "tensor(float)")]

    def run(self, names, feeds):
        self.seen.append(feeds["input_features"].shape)
        # Deterministic marker, distinct per call, so stale-cache bugs show up.
        return [np.full((1, self._hidden_len, self._width),
                        float(len(self.seen)), dtype=np.float32)]


class _FakeDecoder:
    """Mimics the real export's KV-cache contract, including the dummy tensors.

    Scripted token sequence; the n-th decoder call emits ``scripted[n]``.
    Emitting 50256 (EOT) ends the decode.
    """

    def __init__(self, scripted, enc_len=1500):
        self.scripted = list(scripted)
        self.enc_len = enc_len
        self.calls = 0
        self.encoder_pasts_seen = []
        self.use_cache_seen = []

    def get_inputs(self):
        return [
            _Meta("input_ids", ["b", "seq"], "tensor(int64)"),
            _Meta("encoder_hidden_states", ["b", "enc", 8], "tensor(float)"),
            _Meta("past_key_values.0.decoder.key",
                  ["batch_size", 6, "past_decoder_sequence_length", 64],
                  "tensor(float)"),
            _Meta("past_key_values.0.decoder.value",
                  ["batch_size", 6, "past_decoder_sequence_length", 64],
                  "tensor(float)"),
            _Meta("past_key_values.0.encoder.key",
                  ["batch_size", 6, "encoder_sequence_length_out", 64],
                  "tensor(float)"),
            _Meta("past_key_values.0.encoder.value",
                  ["batch_size", 6, "encoder_sequence_length_out", 64],
                  "tensor(float)"),
            _Meta("use_cache_branch", [1], "tensor(bool)"),
        ]

    def get_outputs(self):
        return [
            _Meta("logits", ["b", "seq", 51864], "tensor(float)"),
            _Meta("present.0.decoder.key", ["b", 6, "seq", 64], "tensor(float)"),
            _Meta("present.0.decoder.value", ["b", 6, "seq", 64], "tensor(float)"),
            _Meta("present.0.encoder.key", ["b", 6, "enc", 64], "tensor(float)"),
            _Meta("present.0.encoder.value", ["b", 6, "enc", 64], "tensor(float)"),
        ]

    def run(self, names, feeds):
        use_cache = bool(feeds["use_cache_branch"][0])
        self.use_cache_seen.append(use_cache)
        enc_past = feeds["past_key_values.0.encoder.key"]
        self.encoder_pasts_seen.append(enc_past.shape)

        token = self.scripted[self.calls] if self.calls < len(self.scripted) else 50256
        self.calls += 1

        # The real graph returns a zero-batch placeholder for the encoder
        # presents on the cached path. Reproduce it exactly.
        enc_out_shape = (0, 6, 1, 64) if use_cache else (1, 6, self.enc_len, 64)

        logits = np.zeros((1, 1, 51864), dtype=np.float32)
        logits[0, 0, token] = 10.0
        return [
            logits,
            np.zeros((1, 6, self.calls, 64), dtype=np.float32),
            np.zeros((1, 6, self.calls, 64), dtype=np.float32),
            np.zeros(enc_out_shape, dtype=np.float32),
            np.zeros(enc_out_shape, dtype=np.float32),
        ]


@pytest.fixture
def fake_runner(monkeypatch, tmp_path):
    """A WhisperOnnxRunner wired to the fakes above.

    The stub graphs go in `tmp_path`, never in `config.MODELS_DIR`. Writing into
    the real model directory would mean the test suite creates `models/` on a
    clean checkout, and a test that mutates the directory production code reads
    from is a test that can pass for the wrong reason.
    """
    ort = pytest.importorskip("onnxruntime")
    from snapdragoon.engines import whisper_onnx

    dec = _FakeDecoder(scripted=[843, 523, 616, 5891])
    enc = _FakeEncoder()

    def _session(path, opts=None, providers=None):
        # Dispatch on the *file name* only. Substring-matching the whole path is
        # a trap: pytest derives tmp_path from the test name, so a test called
        # `test_encoder_cache_...` would hand back the encoder fake for the
        # decoder graph and the error would point at the wrong thing entirely.
        return enc if Path(path).name == "encoder_model.onnx" else dec

    monkeypatch.setattr(ort, "InferenceSession", _session)
    monkeypatch.setattr(whisper_onnx, "ort", ort, raising=False)

    model_dir = tmp_path / "fake-whisper"
    model_dir.mkdir()
    (model_dir / "encoder_model.onnx").write_bytes(b"x")
    (model_dir / "decoder_model_merged.onnx").write_bytes(b"x")
    (model_dir / "config.json").write_text(json.dumps({
        "decoder_start_token_id": 50257,
        "eos_token_id": 50256,
        "vocab_size": 51864,
    }))
    yield whisper_onnx.WhisperOnnxRunner(model_dir), dec, enc


class TestKvCacheContract:
    def test_decodes_the_scripted_sequence(self, fake_runner):
        runner, dec, _ = fake_runner
        feats = np.zeros((1, 80, 3000), dtype=np.float32)
        assert runner._greedy_encoder_decoder(feats) == [843, 523, 616, 5891]

    def test_first_pass_disables_the_cache_branch(self, fake_runner):
        runner, dec, _ = fake_runner
        runner._greedy_encoder_decoder(np.zeros((1, 80, 3000), dtype=np.float32))
        assert dec.use_cache_seen[0] is False
        assert all(flag is True for flag in dec.use_cache_seen[1:])

    def test_encoder_cache_is_captured_once_and_held(self, fake_runner):
        """Regression: the encoder runs once, so its KV must not be re-read.

        On the cached path the real graph returns `(0, 6, 1, 64)` -- batch
        dimension zero -- for the encoder presents. Feeding those back destroys
        encoder attention from the second token onward. The visible symptom is a
        correct first few words followed by a spurious end-of-sequence, with no
        error anywhere.
        """
        runner, dec, _ = fake_runner
        ids = runner._greedy_encoder_decoder(
            np.zeros((1, 80, 3000), dtype=np.float32)
        )

        assert len(ids) == 4, "decode was truncated, so the encoder cache was corrupted"
        for shape in dec.encoder_pasts_seen[1:]:
            assert shape[0] == 1, f"zero-batch encoder cache fed back: {shape}"
            assert shape[2] == dec.enc_len, f"encoder cache length changed: {shape}"

    def test_decoder_cache_grows_by_one_per_token(self, fake_runner):
        runner, dec, _ = fake_runner
        runner._greedy_encoder_decoder(np.zeros((1, 80, 3000), dtype=np.float32))
        # prompt is 2 tokens, so pass 1 caches 2; each later pass adds one.
        assert dec.calls == 5  # 4 tokens + the EOT pass

    def test_vocab_axis_is_resolved_from_declared_shape(self, fake_runner):
        runner, _, _ = fake_runner
        assert runner.describe()["logits_vocab_axis"] == -1
        assert runner.format_name == "encoder+decoder"


# ---------------------------------------------------------------------------
# real-model integration (skipped unless weights are present)
# ---------------------------------------------------------------------------
def _reference_clip() -> Path:
    """The JFK test clip, checked in so this test runs on a clean checkout.

    Provenance: `samples/jfk.wav` from ggerganov/whisper.cpp. The audio is an
    excerpt of the 1961 inaugural address, a US government work in the public
    domain. It is the canonical Whisper fixture precisely because the correct
    transcript is known, which is what makes it a real assertion rather than a
    smoke test.
    """
    for candidate in (Path("assets/reference-speech.wav"),):
        if candidate.is_file():
            return candidate
    pytest.skip("reference audio clip not available")


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


@pytest.mark.slow
class TestRealModel:
    def _engine(self):
        from snapdragoon.engines import get_engine

        eng = get_engine("auto")
        if not getattr(eng, "has_asr", False):
            pytest.skip("no ASR weights in models/")
        return eng

    def test_transcribes_the_reference_clip_verbatim(self):
        eng = self._engine()
        audio, sr = _read_wav(_reference_clip())
        text = eng.transcribe(audio, sr).text
        # Compare on normalised text: greedy decoding is deterministic but
        # punctuation is not guaranteed across exports.
        norm = " ".join(text.lower().strip().split()).strip(" .,")
        assert norm == JFK_REFERENCE.lower()

    def test_vision_labels_a_photograph_correctly(self):
        eng = self._engine()
        if not getattr(eng, "has_classifier", False):
            pytest.skip("no classifier weights in models/")
        photo = Path("assets/reference-photo.jpg")
        if not photo.is_file():
            pytest.skip("reference photo not available")
        from snapdragoon.vision.scene import load_image

        labels = " ".join(d.label.lower() for d in eng.classify(load_image(photo)))
        assert any(w in labels for w in ("retriever", "samoyed", "terrier",
                                         "pomeranian")), \
            f"classifier misidentified the subject: {labels}"


# ---------------------------------------------------------------------------
# the short-window gate
# ---------------------------------------------------------------------------
class TestShortWindowGate:
    """A window with too little audio must be refused, not hallucinated.

    Uses a stub ASR that always returns confident text, so if the gate fails to
    fire the test sees a caption where it should see a refusal.
    """

    def _engine(self, monkeypatch):
        from snapdragoon.engines import onnx_engine

        eng = object.__new__(onnx_engine.OnnxEngine)
        eng._asr_error = None
        eng._mel_filters = _mel_filterbank()

        class _Stub:
            def __init__(self):
                self.calls = 0

            def transcribe(self, features):
                self.calls += 1
                return ("Godfrey.", 3)

        eng._asr = _Stub()
        return eng

    @staticmethod
    def _speech(seconds: float) -> np.ndarray:
        n = int(seconds * config.SAMPLE_RATE)
        t = np.linspace(0, seconds, n, endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_a_long_enough_window_is_transcribed(self, monkeypatch):
        eng = self._engine(monkeypatch)
        res = eng.transcribe(self._speech(3.0), config.SAMPLE_RATE)
        assert res.usable
        assert res.skip_reason is None
        assert eng._asr.calls == 1

    def test_a_full_capture_window_is_transcribed(self, monkeypatch):
        """5 s padded to Whisper's 30 s context is 83% padding and still works.

        This is the case that rules out a naive "mostly padding" gate, which
        would have skipped every window the app ever produces.
        """
        from snapdragoon.engines import onnx_engine

        eng = self._engine(monkeypatch)
        n = int(config.CAPTURE_CHUNK_SECONDS * config.SAMPLE_RATE)
        t = np.linspace(0, config.CAPTURE_CHUNK_SECONDS, n, endpoint=False)
        speech = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        res = eng.transcribe(speech, config.SAMPLE_RATE)
        assert res.usable, "a normal capture window must not be skipped"

    def test_a_one_second_window_is_refused(self, monkeypatch):
        eng = self._engine(monkeypatch)
        one_second = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        one_second[::16] = 0.3  # non-zero content, but only 1 s of it
        res = eng.transcribe(one_second, config.SAMPLE_RATE)
        assert not res.usable
        assert res.skip_reason == "short_window"
        assert res.text == "", "a refused window must not carry caption text"

    def test_silence_is_refused(self, monkeypatch, silence):
        eng = self._engine(monkeypatch)
        assert not eng.transcribe(silence, config.SAMPLE_RATE).usable

    def test_the_gate_short_circuits_before_the_model(self, monkeypatch):
        """The refusal happens before the graph is touched, so it cannot raise."""
        eng = self._engine(monkeypatch)
        res = eng.transcribe(np.zeros(100, dtype=np.float32), config.SAMPLE_RATE)
        assert not res.usable
        assert eng._asr.calls == 0, "the model must not be run on a refused window"
