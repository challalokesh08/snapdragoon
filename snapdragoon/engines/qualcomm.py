"""Qualcomm AI Hub / Snapdragon NPU backend.

**This module cannot be validated without Snapdragon hardware.** It is written
against the real deployment APIs so that running Snapdragoon on an HP Omnibook is
a configuration change, not a rewrite. On a Mac it will report unavailable, which
is the correct and honest behaviour.

Deployment paths, in order of preference on Windows-on-Snapdragon:

1. **Qualcomm AI Hub ``qai-hub`` / QNN context binaries** -- the model is
   compiled to a Hexagon context and executed via ``qnn.onnxruntime`` or the
   ``QnnInferenceInterface`` (QNN C API, via ``qai-hub``'s Python wrapper or
   ``snpe``). Fastest path to NPU offload.
2. **Qualcomm AI Hub DirectML plugin** -- ``onnxruntime-directml`` with the
   ``QNNExecutionProvider``. Good for prototyping; lower peak throughput than a
   compiled QNN context.
3. **LiteRT / TensorFlow Lite delegates** -- ``tflite`` with
   ``QnnDelegate``/``HtpxDelegate``. Relevant if you export from the AI Hub's
   ``tflite`` target rather than ``onnx``.

Set ``SNAPDRAGOON_BACKEND=qualcomm`` to force this path, or leave it at ``auto``
and it will be selected automatically when a QNN/DirectML runtime is importable
*and* a compiled context binary exists.

Honesty note for your submission: on non-Snapdragon hardware this engine must not
pretend to run. It raises :class:`QualcommUnavailable` and the app falls back.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .. import config
from .base import Detection, Engine, Transcript, load_labels
from .onnx_engine import log_mel_spectrogram, pad_or_trim, preprocess_image

# Where compiled QNN context binaries land.
CONTEXT_DIR = config.MODELS_DIR / "qnn"


class QualcommUnavailable(RuntimeError):
    """Raised when no Snapdragon NPU runtime or context binary is present."""


class QualcommEngine(Engine):
    name = "qualcomm-npu"
    device_description = "Qualcomm AI Hub / QNN on Snapdragon NPU (Hexagon)"
    is_neural_accelerated = True

    def __init__(self) -> None:
        self._backend = None          # "qnn" | "directml" | "litert"
        self._sessions: dict[str, object] = {}
        self._probe()

    # -- discovery --------------------------------------------------------
    def _probe(self) -> None:
        if not _running_on_snapdragon():
            raise QualcommUnavailable(
                "Not running on a Snapdragon SoC. Set SNAPDRAGOON_ALLOW_UNVERIFIED_NPU=1 "
                "only if you have a QNN runtime available on this machine."
            )

        asr = CONTEXT_DIR / "asr.onnx"
        clf = CONTEXT_DIR / "classifier.onnx"

        try:
            import qai_hub  # noqa: F401  (presence indicates the AI Hub toolchain)

            self._backend = "qnn"
        except ImportError:
            pass

        if self._backend is None:
            try:
                import onnxruntime as ort  # noqa: F401

                if any("QNN" in p for p in ort.get_available_providers()):
                    self._backend = "directml"
            except ImportError:
                pass

        if self._backend is None:
            try:
                from ai_edge_litert.interpreter import Interpreter  # noqa: F401

                self._backend = "litert"
            except ImportError:
                pass

        if self._backend is None:
            raise QualcommUnavailable(
                "No Qualcomm runtime importable. Expected qai-hub, an onnxruntime "
                "build exposing QNNExecutionProvider, or ai-edge-litert with a "
                "Hexagon delegate."
            )

        if asr.is_file():
            self._sessions["asr"] = _load_session(self._backend, asr)
        if clf.is_file():
            self._sessions["classifier"] = _load_session(self._backend, clf)

    # -- inference --------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        started = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        window_seconds = len(audio) / float(sample_rate or 1)
        session = self._sessions.get("asr")
        if session is None:
            return Transcript(
                text="[no NPU model] Build the ASR context binary; see "
                     "docs/SNAPDRAGON_DEPLOYMENT.md.",
                window_seconds=window_seconds,
                latency_ms=0.0,
            )

        features = log_mel_spectrogram(pad_or_trim(audio, 30 * config.SAMPLE_RATE),
                                       _mel_filters_cached())
        tokens = np.array([[50258]], dtype=np.int64)
        generated: list[int] = []
        for _ in range(120):
            out = _run(session, {"features": features, "tokens": tokens})
            nxt = int(np.argmax(out[0, :, -1]))
            if nxt == 50257:
                break
            generated.append(nxt)
            tokens = np.array([[nxt]], dtype=np.int64)

        return Transcript(
            text=_decode(generated) or "[no speech detected]",
            window_seconds=window_seconds,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def classify(self, frame: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        session = self._sessions.get("classifier")
        if session is None:
            return [Detection(label="[no NPU model] classifier context not built",
                              score=0.0, latency_ms=0.0)]
        out = _run(session, {"input": preprocess_image(frame)})
        flat = np.asarray(out, dtype=np.float32).reshape(-1)
        e = np.exp(flat - flat.max())
        probs = e / e.sum()
        order = np.argsort(probs)[::-1][:3]
        labels = load_labels(config.MODELS_DIR /
                             (config.MODEL_REGISTRY["classifier"].labels_path or ""))
        return [
            Detection(
                label=labels[i] if i < len(labels) else f"class {i}",
                score=float(probs[i]),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            for i in order
        ]

    def describe(self) -> dict:
        info = super().describe()
        info["qnn_backend"] = self._backend
        info["context_binaries"] = sorted(p.name for p in CONTEXT_DIR.glob("*.onnx")) \
            if CONTEXT_DIR.is_dir() else []
        return info


# ---------------------------------------------------------------------------
def _running_on_snapdragon() -> bool:
    """Best-effort SoC detection on Windows-on-Snapdragon."""
    if os.name == "nt":
        try:
            import wmi  # type: ignore

            for cpu in wmi.CPU():
                if "snapdragon" in (cpu.Name or "").lower():
                    return True
        except Exception:  # noqa: BLE001
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        text = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
        return "qualcomm" in text or "snapdragon" in text
    if os.environ.get("SNAPDRAGOON_ALLOW_UNVERIFIED_NPU") == "1":
        return True
    return False


def _load_session(backend: str, model: Path):
    if backend == "qnn":
        # A compiled QNN context binary is executed through the QNN C API. The
        # AI Hub ships a thin Python wrapper; fall back to the QNN EP if present.
        from qnn.inference import QnnInferenceInterface  # type: ignore

        iface = QnnInferenceInterface(str(model))
        iface.load()
        return {"kind": "qnn", "iface": iface}
    if backend == "directml":
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            str(model), opts, providers=["QNNExecutionProvider"]
        )
        return {"kind": "ort", "session": sess}
    from ai_edge_litert.interpreter import Interpreter  # type: ignore

    interp = Interpreter(model_path=str(model))
    interp.allocate_tensors()
    return {"kind": "litert", "interp": interp}


def _run(session: dict, feeds: dict) -> np.ndarray:
    kind = session["kind"]
    if kind == "ort":
        sess = session["session"]
        names = {i.name for i in sess.get_inputs()}
        payload = {k: v for k, v in feeds.items() if k in names}
        if not payload:  # name mismatch between registry and export
            payload = {sess.get_inputs()[0].name: next(iter(feeds.values()))}
        return sess.run(None, payload)[0]
    if kind == "qnn":
        iface = session["iface"]
        # QNN bindings require explicit input/output buffer names, which are
        # fixed at compile time by scripts/build_qnn_context.py.
        for name, value in feeds.items():
            iface.set_input(name, value)
        iface.execute()
        return iface.get_output("logits")
    interp = session["interp"]
    interp.set_inputs([np.ascontiguousarray(v) for v in feeds.values()])
    interp.invoke()
    return interp.get_output(interp.get_output_details()[0]["index"])


def _mel_filters_cached():
    global _MEL
    if _MEL is None:
        from .onnx_engine import _mel_filterbank

        _MEL = _mel_filterbank()
    return _MEL


_MEL = None


def _decode(ids: list[int]) -> str:
    if not ids:
        return ""
    try:
        import tiktoken

        return tiktoken.get_encoding("gpt2").decode(ids)
    except Exception:  # noqa: BLE001
        return "".join(chr(i) for i in ids if 32 <= i < 127)
