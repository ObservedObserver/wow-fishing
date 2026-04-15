from __future__ import annotations

import ctypes
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto

from app.platform_win import ensure_dpi_aware

if sys.platform == "win32":
    user32 = ctypes.windll.user32
else:
    user32 = None  # type: ignore[assignment]

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040


class MouseButton(Enum):
    LEFT = auto()
    RIGHT = auto()
    MIDDLE = auto()


_BUTTON_DOWN_FLAGS: dict[MouseButton, int] = {
    MouseButton.LEFT: _MOUSEEVENTF_LEFTDOWN,
    MouseButton.RIGHT: _MOUSEEVENTF_RIGHTDOWN,
    MouseButton.MIDDLE: _MOUSEEVENTF_MIDDLEDOWN,
}
_BUTTON_UP_FLAGS: dict[MouseButton, int] = {
    MouseButton.LEFT: _MOUSEEVENTF_LEFTUP,
    MouseButton.RIGHT: _MOUSEEVENTF_RIGHTUP,
    MouseButton.MIDDLE: _MOUSEEVENTF_MIDDLEUP,
}


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]


def _send_input_mouse(flags: int, dx: int = 0, dy: int = 0) -> None:
    assert user32 is not None
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = _MOUSEINPUT(dx, dy, 0, flags, 0, None)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _send_input_key(vk: int, key_up: bool) -> None:
    assert user32 is not None
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    flags = KEYEVENTF_KEYUP if key_up else 0
    inp.union.ki = _KEYBDINPUT(vk, 0, flags, 0, None)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


class InputBackend(ABC):
    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None: ...

    @abstractmethod
    def mouse_move_relative(self, dx: int, dy: int) -> None: ...

    @abstractmethod
    def mouse_down(self, button: MouseButton) -> None: ...

    @abstractmethod
    def mouse_up(self, button: MouseButton) -> None: ...

    @abstractmethod
    def key_down(self, vk_code: int) -> None: ...

    @abstractmethod
    def key_up(self, vk_code: int) -> None: ...


class SendInputBackend(InputBackend):
    """SendInput for keys and mouse buttons; absolute moves use SetCursorPos."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("SendInputBackend requires Windows")
        ensure_dpi_aware()

    def move_cursor(self, x: int, y: int) -> None:
        assert user32 is not None
        user32.SetCursorPos(int(x), int(y))

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        _send_input_mouse(_MOUSEEVENTF_MOVE, int(dx), int(dy))

    def mouse_down(self, button: MouseButton) -> None:
        _send_input_mouse(_BUTTON_DOWN_FLAGS[button])

    def mouse_up(self, button: MouseButton) -> None:
        _send_input_mouse(_BUTTON_UP_FLAGS[button])

    def key_down(self, vk_code: int) -> None:
        _send_input_key(int(vk_code), key_up=False)

    def key_up(self, vk_code: int) -> None:
        _send_input_key(int(vk_code), key_up=True)


_LEGACY_MOVE = 0x0001
_LEGACY_LEFTDOWN = 0x0002
_LEGACY_LEFTUP = 0x0004
_LEGACY_RIGHTDOWN = 0x0008
_LEGACY_RIGHTUP = 0x0010
_LEGACY_MIDDLEDOWN = 0x0020
_LEGACY_MIDDLEUP = 0x0040

_LEGACY_BUTTON_DOWN = {
    MouseButton.LEFT: _LEGACY_LEFTDOWN,
    MouseButton.RIGHT: _LEGACY_RIGHTDOWN,
    MouseButton.MIDDLE: _LEGACY_MIDDLEDOWN,
}
_LEGACY_BUTTON_UP = {
    MouseButton.LEFT: _LEGACY_LEFTUP,
    MouseButton.RIGHT: _LEGACY_RIGHTUP,
    MouseButton.MIDDLE: _LEGACY_MIDDLEUP,
}


class LegacyBackend(InputBackend):
    """Legacy mouse_event / keybd_event (backward-compatible)."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("LegacyBackend requires Windows")
        ensure_dpi_aware()
        self.user32 = ctypes.windll.user32

    def move_cursor(self, x: int, y: int) -> None:
        self.user32.SetCursorPos(int(x), int(y))

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        self.user32.mouse_event(_LEGACY_MOVE, int(dx), int(dy), 0, 0)

    def mouse_down(self, button: MouseButton) -> None:
        self.user32.mouse_event(_LEGACY_BUTTON_DOWN[button], 0, 0, 0, 0)

    def mouse_up(self, button: MouseButton) -> None:
        self.user32.mouse_event(_LEGACY_BUTTON_UP[button], 0, 0, 0, 0)

    def key_down(self, vk_code: int) -> None:
        self.user32.keybd_event(int(vk_code), 0, 0, 0)

    def key_up(self, vk_code: int) -> None:
        self.user32.keybd_event(int(vk_code), 0, KEYEVENTF_KEYUP, 0)


def create_backend(name: str) -> InputBackend:
    if sys.platform != "win32":
        return _NonWindowsInputBackend()
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"legacy", "mouse_event", "keybd_event"}:
        return LegacyBackend()
    if normalized in {"send_input", "sendinput", "default"}:
        return SendInputBackend()
    raise ValueError(f"unknown input.backend: {name!r}; expected legacy or send_input")


class _NonWindowsInputBackend(InputBackend):
    """No-op backend for imports/tests on non-Windows hosts."""

    def move_cursor(self, x: int, y: int) -> None:
        return None

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        return None

    def mouse_down(self, button: MouseButton) -> None:
        return None

    def mouse_up(self, button: MouseButton) -> None:
        return None

    def key_down(self, vk_code: int) -> None:
        return None

    def key_up(self, vk_code: int) -> None:
        return None
