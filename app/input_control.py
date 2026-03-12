from __future__ import annotations

import ctypes
import random
import time

from app.config import ControlConfig
from app.platform_win import ensure_dpi_aware

_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_KEYEVENTF_KEYUP = 0x0002
_VK_1 = 0x31
_VK_SPACE = 0x20
_MOVE_VERIFY_TOLERANCE_PX = 3
_RIGHT_CLICK_RESET_DELAY_S = 0.01
_RIGHT_CLICK_HOLD_S = 0.03
_FUNCTION_KEY_BASE = 0x70
_FUNCTION_KEY_MAX = 24
_SPECIAL_KEYS: dict[str, int] = {
    "ENTER": 0x0D,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "F12": 0x7B,
    "SPACE": 0x20,
    "TAB": 0x09,
}


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MouseController:
    def __init__(self, cfg: ControlConfig) -> None:
        self.cfg = cfg
        ensure_dpi_aware()
        self.user32 = ctypes.windll.user32

    def get_position(self) -> tuple[int, int]:
        point = _POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def move_to(self, x: int, y: int) -> tuple[int, int]:
        jitter = self.cfg.jitter_px
        x = x + random.randint(-jitter, jitter)
        y = y + random.randint(-jitter, jitter)
        return self._move_smooth(x, y)

    def move_and_right_click(self, x: int, y: int) -> tuple[int, int]:
        final_x, final_y = self.move_to(x, y)
        self.right_click()
        return final_x, final_y

    def right_click(self) -> None:
        self._mouse_event(_MOUSEEVENTF_RIGHTUP, 0, 0)
        time.sleep(_RIGHT_CLICK_RESET_DELAY_S)
        self._mouse_event(_MOUSEEVENTF_RIGHTDOWN, 0, 0)
        time.sleep(_RIGHT_CLICK_HOLD_S)
        self._mouse_event(_MOUSEEVENTF_RIGHTUP, 0, 0)
        time.sleep(_RIGHT_CLICK_RESET_DELAY_S)
        self._mouse_event(_MOUSEEVENTF_RIGHTUP, 0, 0)

    def press_key_1(self) -> None:
        self._press_vk(_VK_1)

    def press_space(self) -> None:
        self._press_vk(_VK_SPACE)

    def press_interaction_key(self) -> None:
        self._press_vk(_virtual_key_from_name(self.cfg.interaction_key))

    def _move_smooth(self, target_x: int, target_y: int) -> tuple[int, int]:
        start_x, start_y = self.get_position()
        current_x = start_x
        current_y = start_y

        duration_s = max(0.02, self.cfg.move_duration_ms / 1000.0)
        steps = max(8, int(duration_s / 0.008))
        step_sleep = duration_s / steps

        for i in range(1, steps + 1):
            goal_x = int(round(start_x + (target_x - start_x) * (i / steps)))
            goal_y = int(round(start_y + (target_y - start_y) * (i / steps)))
            if goal_x != current_x or goal_y != current_y:
                self.user32.SetCursorPos(int(goal_x), int(goal_y))
                current_x, current_y = self.get_position()
            time.sleep(step_sleep)
        actual_x, actual_y = self.get_position()
        if (
            abs(actual_x - target_x) > _MOVE_VERIFY_TOLERANCE_PX
            or abs(actual_y - target_y) > _MOVE_VERIFY_TOLERANCE_PX
        ):
            self.user32.SetCursorPos(int(target_x), int(target_y))
            time.sleep(0.005)
            actual_x, actual_y = self.get_position()
        return actual_x, actual_y

    def _mouse_event(self, flags: int, dx: int, dy: int) -> None:
        self.user32.mouse_event(flags, dx, dy, 0, 0)

    def _press_vk(self, vk_code: int) -> None:
        self.user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.03)
        self.user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


def _virtual_key_from_name(key_name: str) -> int:
    normalized = str(key_name).strip().upper()
    if not normalized:
        raise ValueError("interaction_key cannot be empty")

    special = _SPECIAL_KEYS.get(normalized)
    if special is not None:
        return special

    if len(normalized) == 1 and normalized.isalnum():
        return ord(normalized)

    if normalized.startswith("F") and normalized[1:].isdigit():
        fn_index = int(normalized[1:])
        if 1 <= fn_index <= _FUNCTION_KEY_MAX:
            return _FUNCTION_KEY_BASE + fn_index - 1

    raise ValueError(f"unsupported interaction_key: {key_name!r}")
