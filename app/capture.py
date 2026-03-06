from __future__ import annotations

import ctypes
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


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


_GA_ROOT = 2


class ScreenCapture:
    def __init__(self) -> None:
        if mss is None:
            raise RuntimeError("mss is not installed")
        ensure_dpi_aware()
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        shot = self.grab_with_offset()
        return shot.frame_bgr

    def grab_window_with_offset(
        self,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> CaptureFrame:
        window = _select_window_rect(preferred_x=preferred_x, preferred_y=preferred_y)
        if window is None:
            return self.grab_with_offset(preferred_x=preferred_x, preferred_y=preferred_y)

        shot = self._sct.grab(window)
        frame = np.array(shot)[:, :, :3]
        return CaptureFrame(
            frame_bgr=frame,
            left=int(window["left"]),
            top=int(window["top"]),
        )

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


def _select_window_rect(
    preferred_x: int | None = None,
    preferred_y: int | None = None,
) -> dict[str, int] | None:
    if preferred_x is None or preferred_y is None:
        return None
    return _client_rect_from_point(preferred_x, preferred_y)


def _client_rect_from_point(x: int, y: int) -> dict[str, int] | None:
    if not hasattr(ctypes, "windll"):
        return None

    user32 = ctypes.windll.user32
    point = _POINT(int(x), int(y))
    hwnd = user32.WindowFromPoint(point)
    if not hwnd:
        return None

    root = user32.GetAncestor(hwnd, _GA_ROOT)
    if root:
        hwnd = root

    rect = _RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    top_left = _POINT(rect.left, rect.top)
    bottom_right = _POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None

    width = int(bottom_right.x - top_left.x)
    height = int(bottom_right.y - top_left.y)
    if width < 16 or height < 16:
        return None

    return {
        "left": int(top_left.x),
        "top": int(top_left.y),
        "width": width,
        "height": height,
    }
