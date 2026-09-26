"""Flask application: capture pipeline plus a screen-reader-first web UI.

Routes
------
``GET  /``                 the interface
``GET  /api/status``       which engine is live, and whether it is honest about it
``POST /api/captions``     Server-Sent Events stream of live captions
``POST /api/describe``     one-shot scene description from an uploaded still
``GET  /api/health``       liveness probe

The SSE endpoint is the interesting one: it streams caption events as they are
produced. The browser side is responsible for pushing them into an
``aria-live="polite"`` region, which is the whole accessibility story of this
project in one sentence.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .. import config
from ..engines import get_engine

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(engine=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False
    eng = engine if engine is not None else get_engine()

    @app.get("/")
    def index() -> Response:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # Inline the status so the first paint already tells the user which
        # engine is running, rather than flashing "loading".
        status = json.dumps(eng.describe())
        return Response(html.replace("__SNAPDRAGOON_STATUS__", status), mimetype="text/html")

    @app.get("/api/status")
    def status():
        return jsonify(eng.describe())

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "engine": eng.name})

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        target = (STATIC_DIR / filename).resolve()
        # Path traversal guard: never serve outside the static directory.
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return jsonify({"error": "not found"}), 404
        # content_type, not mimetype: Flask appends its own charset otherwise,
        # yielding "text/css; charset=utf-8; charset=utf-8".
        return Response(target.read_bytes(), content_type=_guess_type(target))

    @app.post("/api/describe")
    def describe():
        """One-shot scene description. Accepts an uploaded image or ?image=path.

        ``vision.scene`` has no module-level OpenCV dependency, so importing it
        always succeeds; the ImportError we care about happens inside
        ``load_image``, and is handled below.
        """
        from ..vision.scene import load_image

        frame = None
        try:
            if "image" in request.files:
                import cv2
                import numpy as np

                data = np.frombuffer(request.files["image"].read(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif request.args.get("image"):
                frame = load_image(_resolve_asset(request.args["image"]))
        except ModuleNotFoundError as exc:
            # A missing optional dependency is a setup problem, not an internal
            # error. Say which package and how to get it, or the user is left
            # staring at "No module named 'cv2'" with no idea what to do.
            return jsonify({
                "error": f"Image support needs {exc.name}. "
                         f"Install it with: pip install opencv-python-headless"
            }), 503
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        if frame is None:
            return jsonify({"error": "no image supplied"}), 400

        started = time.perf_counter()
        try:
            detections = eng.classify(frame)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"inference failed: {exc}"}), 500

        from ..engines.base import format_detections

        return jsonify({
            "text": format_detections(detections),
            "detections": [
                {"label": d.label, "score": round(d.score, 4),
                 "latency_ms": round(d.latency_ms, 2)}
                for d in detections
            ],
            "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "engine": eng.name,
        })

    @app.post("/api/captions")
    def captions() -> Response:
        """Stream live captions as SSE.

        Accepts ``?source=mic`` (default) or ``?source=file&path=...`` to replay
        a wav file, which is what the demo script and tests use so the streaming
        path is verifiable with no microphone.
        """
        # Every request-scoped value is read here, on the request thread. The
        # producer thread below has no access to Flask's `request` context.
        source = request.args.get("source", "mic")
        path = request.args.get("path", "")
        silence_threshold = float(request.args.get("silence", 1e-3))

        events: queue.Queue[dict | None] = queue.Queue(maxsize=64)

        def emit(**payload) -> None:
            try:
                events.put_nowait(payload)
            except queue.Full:
                # Drop the oldest event rather than stall capture: captions are
                # transient by nature and a backlog is worse than a gap.
                try:
                    events.get_nowait()
                    events.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

        def produce() -> None:
            try:
                from ..audio.capture import is_silent, iter_windows, open_stream

                chunk_seconds = config.CAPTURE_CHUNK_SECONDS

                if source == "file":
                    # `path` is captured above, before the thread starts: the
                    # `request` proxy is unbound outside the request context.
                    for window in _iter_wav(path):
                        _emit_for_window(emit, eng, window, silence_threshold)
                else:
                    stream = open_stream()
                    for window in iter_windows(stream, chunk_seconds):
                        _emit_for_window(emit, eng, window, silence_threshold)
            except Exception as exc:  # noqa: BLE001
                log.exception("caption producer failed")
                emit(kind="error", message=str(exc))
            finally:
                emit(kind="end")

        threading.Thread(target=produce, daemon=True).start()

        def stream() -> Response:
            yield "retry: 3000\n\n"
            while True:
                try:
                    event = events.get(timeout=15)
                except queue.Empty:
                    # Comment frame doubles as a keepalive.
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("kind") == "end":
                    break
            yield "data: {\"kind\": \"closed\"}\n\n"

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    return app


def _emit_for_window(emit, eng, window, silence_threshold: float) -> None:
    from ..audio.capture import is_silent

    if is_silent(window, threshold=silence_threshold):
        emit(kind="skip", reason="silence")
        return
    result = eng.transcribe(window, config.SAMPLE_RATE)
    # The engine owns the model's input contract, so it is the only component
    # that can know a window is too short to transcribe honestly. Honour its
    # verdict instead of captioning a hallucination.
    if not result.usable:
        emit(kind="skip", reason=result.skip_reason or "unusable")
        return
    emit(
        kind="caption",
        text=result.text,
        window_seconds=round(result.window_seconds, 2),
        latency_ms=round(result.latency_ms, 1),
        timestamp=time.time(),
    )


def _iter_wav(path: str):
    """Replay a 16 kHz mono wav file as fixed windows."""
    import wave

    from .. import config as _cfg

    if not path:
        raise FileNotFoundError("no wav path supplied")
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != _cfg.SAMPLE_RATE:
            raise ValueError(
                f"expected {_cfg.SAMPLE_RATE} Hz wav, got {wf.getframerate()} Hz"
            )
        want = int(_cfg.CAPTURE_CHUNK_SECONDS * _cfg.SAMPLE_RATE)
        while True:
            raw = wf.readframes(want)
            if not raw:
                return
            import numpy as np

            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if len(data) < want:
                data = np.pad(data, (0, want - len(data)))
            yield data


def _resolve_asset(raw: str) -> Path:
    """Resolve a server-side image path.

    Relative paths resolve against the project root rather than the process CWD,
    so ``?image=assets/sample.jpg`` works no matter where the server was started
    from. Absolute paths are honoured as given, since a caller may legitimately
    point at a file elsewhere on disk.
    """
    p = Path(raw).expanduser()
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def _guess_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }.get(path.suffix, "application/octet-stream")


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    eng = get_engine()
    print(f"\n  Snapdragoon engine : {eng.name}")
    print(f"  Inference device   : {eng.device_description}")
    print(f"  Neural accelerated : {eng.is_neural_accelerated}")
    print(f"\n  Serving on         : http://{config.HOST}:{config.PORT}\n")
    eng.warmup()
    create_app(eng).run(host=config.HOST, port=config.PORT, threaded=True)


if __name__ == "__main__":  # pragma: no cover
    main()
