from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None  # type: ignore


@dataclass(slots=True)
class CaptureFrame:
    frame_bgr: np.ndarray
    left: int
    top: int


class ScreenCapture:
    def __init__(self) -> None:
        if mss is None:
            raise RuntimeError("mss is not installed")
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        shot = self.grab_with_offset()
        return shot.frame_bgr

    def grab_with_offset(self) -> CaptureFrame:
        monitor = self._sct.monitors[1]
        shot = self._sct.grab(monitor)
        frame = np.array(shot)[:, :, :3]
        return CaptureFrame(
            frame_bgr=frame,
            left=int(monitor["left"]),
            top=int(monitor["top"]),
        )
