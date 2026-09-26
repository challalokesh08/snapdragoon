"""ONNX Runtime backend -- real model inference on a CPU.

This is what runs during development on a Mac, and it is the reference
implementation that proves the pre/post-processing is correct. The exact same
tensors feed :mod:`snapdragoon.engines.qualcomm`, so a behaviour difference on
the NPU is a genuine finding rather than a code-path accident.

Input/output names are discovered at load time rather than hard-coded, because
ONNX exports from Qualcomm AI Hub and from upstream projects disagree on
names (``input_features`` vs ``input``, ``logits`` vs ``output``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import config
from ..audio.capture import content_seconds
from .base import Detection, Engine, Transcript, load_labels

# Minimum real audio in a window before it is worth transcribing.
#
# Chosen by measurement on whisper-tiny.en rather than by taste: a 5 s window
# padded to Whisper's 30 s context (83% padding) transcribes correctly, while a
# 1 s window (97% padding) returns a hallucinated proper noun. The question is
# never "how much padding" but "how much speech", so the gate is expressed in
# seconds of content and sits conservatively between the two measured cases.
_MIN_TRANSCRIBE_SECONDS = 2.0

# Whisper audio front-end constants (fixed by the model, not tunable).
_N_FFT = 400
_HOP = 160
_N_MELS = 80
_N_SAMPLES = 30 * config.SAMPLE_RATE  # 30 s context window


class OnnxEngine(Engine):
    name = "onnx"
    device_description = "ONNX Runtime on CPU (development baseline)"

    def __init__(
        self,
        asr_path: Path | None = None,
        classifier_path: Path | None = None,
        labels_path: Path | None = None,
    ) -> None:
        import onnxruntime as ort  # imported lazily so the package imports without it

        self._ort = ort
        opts = ort.SessionOptions()
        # Keep logs quiet; the UI surfaces our own status instead.
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 0  # 0 == let onnxruntime decide

        # ASR: delegate graph handling to WhisperOnnxRunner, which resolves
        # encoder/decoder layout, KV-cache plumbing, special-token ids and the
        # logits axis from the model's own declared signature. Position-based
        # guessing was tried first and is wrong for every real export.
        self._asr = None
        self._asr_error: str | None = None
        spec_asr = config.MODEL_REGISTRY["asr"]
        asr_dir = Path(asr_path) if asr_path else config.model_path(spec_asr)
        if asr_dir.is_dir() and any(asr_dir.glob("*.onnx")):
            try:
                from .whisper_onnx import WhisperOnnxRunner

                self._asr = WhisperOnnxRunner(asr_dir)
            except Exception as exc:  # noqa: BLE001
                # A malformed or unsupported export must not stop the vision
                # path from working, so record why and carry on.
                self._asr_error = str(exc)
        elif not asr_dir.exists():
            self._asr_error = f"{asr_dir} not found; run scripts/fetch_models.py"

        self._cls = None
        self._cls_in: str | None = None
        spec_cls = config.MODEL_REGISTRY["classifier"]
        classifier_path = classifier_path or config.model_path(spec_cls)
        if classifier_path.is_file():
            self._cls = ort.InferenceSession(str(classifier_path), opts, providers=["CPUExecutionProvider"])
            self._cls_in = self._cls.get_inputs()[0].name

        self._labels = load_labels(
            labels_path or (config.MODELS_DIR / (spec_cls.labels_path or ""))
        )
        self._mel_filters = _mel_filterbank()

    # -- capability -------------------------------------------------------
    @property
    def has_asr(self) -> bool:
        return self._asr is not None

    @property
    def has_classifier(self) -> bool:
        return self._cls is not None

    def describe(self) -> dict:
        info = super().describe()
        info["asr_loaded"] = self.has_asr
        info["classifier_loaded"] = self.has_classifier
        if self._asr is not None:
            info["asr"] = self._asr.describe()
        elif self._asr_error:
            info["asr_error"] = self._asr_error
        return info

    def warmup(self) -> None:
        # Absorb first-call graph optimisation and memory allocation, so the
        # first real caption is not anomalously slow.
        try:
            if self._asr is not None:
                silence = np.zeros(_N_SAMPLES, dtype=np.float32)
                self.transcribe(silence, config.SAMPLE_RATE)
        except Exception:  # noqa: BLE001 - warmup must never break startup
            pass
        if self._cls is not None:
            try:
                self._cls.run(None, {self._cls_in: np.zeros((1, 3, 224, 224), dtype=np.float32)})
            except Exception:  # noqa: BLE001
                pass

    # -- inference --------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        started = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        window_seconds = len(audio) / float(sample_rate or 1)

        if self._asr is None:
            reason = self._asr_error or "model not found"
            return Transcript(
                text=f"[no model] Speech recognition unavailable: {reason}",
                window_seconds=window_seconds,
                latency_ms=0.0,
            )

        # Whisper wants a full 30 s context, so a shorter window is zero-padded
        # up to it. That is fine on its own -- a 5 s window transcribes
        # correctly. What fails is a window with almost no audio in it: the
        # model answers with a confident hallucination rather than with
        # nothing, observed as a stray "Godfrey." on a 1 s tail of real speech.
        #
        # This check lives in the engine because only the engine knows what
        # padding it is about to add; the caller sees the original window and
        # would compute a different answer.
        speech = content_seconds(audio, sample_rate or config.SAMPLE_RATE)
        if speech < _MIN_TRANSCRIBE_SECONDS:
            return Transcript(
                text="",
                window_seconds=window_seconds,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                skip_reason="short_window",
            )

        features = log_mel_spectrogram(pad_or_trim(audio, _N_SAMPLES),
                                       self._mel_filters)
        try:
            text, _steps = self._asr.transcribe(features)
        except Exception as exc:  # noqa: BLE001
            return Transcript(
                text=f"[inference error] {exc}",
                window_seconds=window_seconds,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        return Transcript(
            text=text.strip() or "[no speech detected]",
            window_seconds=window_seconds,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def classify(self, frame: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        if self._cls is None:
            return [Detection(label="[no model] classifier not loaded", score=0.0,
                              latency_ms=0.0)]

        batch = preprocess_image(frame)
        logits = self._cls.run(None, {self._cls_in: batch})[0]
        flat = np.asarray(logits, dtype=np.float32).reshape(-1)
        probs = _softmax(flat)
        order = np.argsort(probs)[::-1][:3]
        latency = (time.perf_counter() - started) * 1000.0
        return [
            Detection(
                label=self._labels[i] if i < len(self._labels) else f"class {i}",
                score=float(probs[i]),
                latency_ms=latency,
            )
            for i in order
        ]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.pad(audio, (0, target - len(audio)))


def _hz_to_mel_slaney(freqs: np.ndarray) -> np.ndarray:
    """Slaney (librosa ``htk=False``) Hz -> mel.

    Slaney's low-frequency region is linear below 1 kHz and logarithmic above,
    which is what Whisper's filterbank uses. HTK's all-logarithmic formula is
    close enough to look plausible and wrong enough to degrade transcription,
    so the piecewise form is reproduced exactly.
    """
    f_sp = 200.0 / 3.0
    mels = freqs / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp          # 15.0
    logstep = np.log(6.4) / 27.0
    log_region = freqs >= min_log_hz
    mels = np.where(
        log_region,
        min_log_mel + np.log(np.maximum(freqs, min_log_hz) / min_log_hz) / logstep,
        mels,
    )
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_hz_to_mel_slaney`."""
    f_sp = 200.0 / 3.0
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    return np.where(
        log_region,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        freqs,
    )


def _mel_filterbank() -> np.ndarray:
    """Whisper's 80-filter mel bank: librosa ``mel(norm='slaney')``.

    Faithful to ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)``, which
    is what Whisper ships as ``mel_filters.npz``. Two details are load-bearing:

    * **Slaney, not HTK.** The mel scale is piecewise below/above 1 kHz.
    * **Slaney area normalisation.** Each filter is scaled by
      ``2 / (edge[i+2] - edge[i])``, which corrects for the narrowing filter
      bandwidth at high frequency. Without it the top mel bands are attenuated
      and Whisper mis-hears sibilants.

    Returns (80, 201), matching a 400-point rFFT.
    """
    n_bins = _N_FFT // 2 + 1
    fftfreqs = np.fft.rfftfreq(n=_N_FFT, d=1.0 / config.SAMPLE_RATE)

    # Evenly spaced *mel* points from DC to Nyquist, converted to Hz.
    mel_points = np.linspace(
        _hz_to_mel_slaney(np.array([0.0]))[0],
        _hz_to_mel_slaney(np.array([config.SAMPLE_RATE / 2.0]))[0],
        _N_MELS + 2,
    )
    hz_points = _mel_to_hz_slaney(mel_points)

    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fftfreqs[None, :]

    weights = np.zeros((_N_MELS, n_bins), dtype=np.float64)
    for i in range(_N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalisation: equalise each filter's contribution by its width.
    enorm = 2.0 / (hz_points[2:_N_MELS + 2] - hz_points[:_N_MELS])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def _hann_window(n: int) -> np.ndarray:
    """**Periodic** Hann window, matching ``torch.hann_window(n)``.

    ``np.hanning(n)`` is the *symmetric* variant. The two differ in the last
    sample (symmetric forces it to exactly 0) and produce measurably different
    spectra. Whisper uses torch's default, which is periodic.
    """
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper-compatible log-mel features, shaped (1, 80, 3000).

    Accumulates in float64: squaring a power spectrum overflows float32 on
    loud input and poisons the whole feature map with ``inf``, which then
    propagates into the model as garbage rather than raising.

    Frame count matches Whisper exactly. ``torch.stft(..., center=True)`` on
    30 s yields 3001 frames and Whisper drops the last with ``[..., :-1]``;
    taking 3000 frames directly is equivalent.

    The input length is normalised here rather than trusted. The frame loop
    below strides a fixed 3000x400 window over the buffer with no bounds check,
    so a shorter input does not raise -- it reads past the end and returns
    whatever memory followed. Silent garbage, not an error.
    """
    window = _hann_window(_N_FFT).astype(np.float64)
    n_frames = _N_SAMPLES // _HOP  # 3000
    audio = pad_or_trim(np.asarray(audio, dtype=np.float64), _N_SAMPLES)
    padded = np.pad(audio, (_N_FFT // 2, _N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, _N_FFT),
        strides=(padded.strides[0] * _HOP, padded.strides[0]),
        writeable=False,
    )
    spec = np.abs(np.fft.rfft(frames * window, n=_N_FFT)) ** 2
    mel = mel_filters.astype(np.float64) @ spec.T
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)  # Whisper dynamic range clamp
    # `mel` is already (n_mels, n_frames) = (80, 3000), which is exactly the
    # layout Whisper exports expect. Do not transpose.
    return ((log + 4.0) / 4.0)[np.newaxis, :, :].astype(np.float32)


def preprocess_image(frame: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB -> (1, 3, 224, 224) float32, NCHW, ImageNet-normalised.

    Uses nearest-neighbour resize on purpose: no Pillow/OpenCV dependency, and it
    is deterministic across machines, which keeps the Mac and NPU numbers
    comparable.
    """
    img = np.asarray(frame)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    # Centre-crop the shorter side to a square, then resize to 224.
    if h > w:
        top = (h - w) // 2
        img = img[top:top + w, :, :]
    elif w > h:
        left = (w - h) // 2
        img = img[:, left:left + h, :]

    ys = (np.arange(224) * img.shape[0] // 224).clip(0, img.shape[0] - 1)
    xs = (np.arange(224) * img.shape[1] // 224).clip(0, img.shape[1] - 1)
    img = img[ys][:, xs]

    x = img.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(x, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
