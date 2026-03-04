from __future__ import annotations

import numpy as np

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None  # type: ignore


class ScreenCapture:
    def __init__(self) -> None:
        if mss is None:
            raise RuntimeError("mss is not installed")
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        monitor = self._sct.monitors[1]
        shot = self._sct.grab(monitor)
        frame = np.array(shot)
        return frame[:, :, :3]

