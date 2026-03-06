from __future__ import annotations

import ctypes


_DPI_AWARE_SET = False


def ensure_dpi_aware() -> None:
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET:
        return

    if not hasattr(ctypes, "windll"):
        return

    user32 = ctypes.windll.user32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            return
    _DPI_AWARE_SET = True
