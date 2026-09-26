"""Scene description from a camera or a still image.

Two capture modes, one interface:

* :func:`camera_frames` -- live webcam stream, for the live demo.
* :func:`load_image`     -- a file on disk, used by the CLI and by tests so the
  vision path is demonstrable with no camera present.

On Windows-on-Snapdragon, ``cv2``'s default ``CAP_DSHOW`` backend is reliable with
integrated webcams; if it fights you, switch to ``CAP_MSMF``. We try a short
list rather than hard-coding one, because the right backend is hardware-specific.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

_BACKENDS = ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY")


def open_camera(index: int = 0):
    """Open the first camera backend that works."""
    import cv2

    last: Exception | None = None
    for name in _BACKENDS:
        backend = getattr(cv2, name, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        last = RuntimeError(f"{name} could not open camera {index}")
        cap.release()
    raise RuntimeError(
        f"No camera could be opened (tried {', '.join(_BACKENDS)}). "
        "Grant camera access in System Settings > Privacy & Security > Camera."
    ) from last


def camera_frames(
    camera_index: int = 0, width: int = 640, height: int = 480
) -> Iterator[np.ndarray]:
    """Yield successive RGB frames from the webcam."""
    import cv2

    cap = open_camera(camera_index)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 array."""
    # Check existence before importing cv2: a missing file should report
    # "not found" whether or not OpenCV is installed.
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    import cv2

    data = np.fromfile(p, dtype=np.uint8)  # fromfile is Unicode-safe on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, frame: np.ndarray) -> Path:
    """Write an RGB frame to disk, creating parent directories."""
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(p.suffix or ".png", frame[:, :, ::-1])
    if not ok:
        raise ValueError(f"Could not encode image for {p}")
    encoded.tofile(p)
    return p


def describe_frame(frame: np.ndarray, engine, limit: int = 3) -> str:
    """Run the engine and return a spoken-friendly sentence."""
    from ..engines.base import format_detections

    return format_detections(engine.classify(frame), limit=limit)
