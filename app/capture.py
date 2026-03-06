from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.platform_win import ensure_dpi_aware

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
        ensure_dpi_aware()
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        shot = self.grab_with_offset()
        return shot.frame_bgr

    def grab_with_offset(
        self,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> CaptureFrame:
        monitor = _select_monitor(self._sct.monitors, preferred_x=preferred_x, preferred_y=preferred_y)
        shot = self._sct.grab(monitor)
        frame = np.array(shot)[:, :, :3]
        return CaptureFrame(
            frame_bgr=frame,
            left=int(monitor["left"]),
            top=int(monitor["top"]),
        )


def _contains_point(monitor: dict[str, int], x: int, y: int) -> bool:
    left = int(monitor["left"])
    top = int(monitor["top"])
    width = int(monitor["width"])
    height = int(monitor["height"])
    return left <= x < left + width and top <= y < top + height


def _select_monitor(
    monitors: list[dict[str, int]],
    preferred_x: int | None = None,
    preferred_y: int | None = None,
) -> dict[str, int]:
    if not monitors:
        raise RuntimeError("No monitors available for capture")

    if preferred_x is not None and preferred_y is not None:
        for monitor in monitors[1:]:
            if _contains_point(monitor, preferred_x, preferred_y):
                return monitor

    # mss.monitors[0] is the full virtual desktop and avoids primary-monitor-only bugs.
    return monitors[0]
