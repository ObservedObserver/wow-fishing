from __future__ import annotations

import ctypes
import random
import sys
import time

from app.config import ControlConfig, HumanizeConfig, InputConfig, MousePathConfig
from app.humanize import HumanDelay
from app.input_backend import MouseButton, create_backend, InputBackend
from app.mouse_path import generate_path
from app.platform_win import ensure_dpi_aware

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
    def __init__(
        self,
        cfg: ControlConfig,
        *,
        input_backend: InputBackend | None = None,
        input_cfg: InputConfig | None = None,
        mouse_path_cfg: MousePathConfig | None = None,
        humanize_cfg: HumanizeConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.mouse_path_cfg = mouse_path_cfg
        self._humanize_cfg = humanize_cfg
        ensure_dpi_aware()
        if input_backend is not None:
            self.backend = input_backend
        elif input_cfg is not None:
            self.backend = create_backend(input_cfg.backend)
        else:
            self.backend = create_backend("send_input")
        if sys.platform == "win32":
            self.user32 = ctypes.windll.user32
        else:
            self.user32 = None  # type: ignore[assignment]
        self._randomize_holds = bool(
            (humanize_cfg is not None and humanize_cfg.enabled) or cfg.randomize_key_hold
        )
        self._randomize_mouse = bool(
            (humanize_cfg is not None and humanize_cfg.enabled) or cfg.randomize_mouse_click_timing
        )

    def get_position(self) -> tuple[int, int]:
        if self.user32 is None:
            return 0, 0
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
        if self._randomize_mouse:
            reset1 = _RIGHT_CLICK_RESET_DELAY_S * random.uniform(0.7, 1.4)
            hold_s = _RIGHT_CLICK_HOLD_S * random.uniform(0.6, 1.5)
            reset2 = _RIGHT_CLICK_RESET_DELAY_S * random.uniform(0.7, 1.4)
        else:
            reset1 = _RIGHT_CLICK_RESET_DELAY_S
            hold_s = _RIGHT_CLICK_HOLD_S
            reset2 = _RIGHT_CLICK_RESET_DELAY_S
        self.backend.mouse_up(MouseButton.RIGHT)
        time.sleep(reset1)
        self.backend.mouse_down(MouseButton.RIGHT)
        time.sleep(hold_s)
        self.backend.mouse_up(MouseButton.RIGHT)
        time.sleep(reset2)
        self.backend.mouse_up(MouseButton.RIGHT)

    def press_key_1(self) -> None:
        self._press_vk(0x31, hold_median_ms=self.cfg.key_press_hold_ms, slot2=False)

    def press_key_2(self) -> None:
        self._press_vk(0x32, hold_median_ms=self.cfg.slot2_key_press_hold_ms, slot2=True)

    def press_key_4(self) -> None:
        self._press_vk(0x34, hold_median_ms=self.cfg.key_press_hold_ms, slot2=False)

    def press_space(self) -> None:
        self._press_vk(0x20, hold_median_ms=self.cfg.key_press_hold_ms, slot2=False)

    def press_interaction_key(self) -> None:
        self._press_vk(
            _virtual_key_from_name(self.cfg.interaction_key),
            hold_median_ms=self.cfg.key_press_hold_ms,
            slot2=False,
        )

    def press_return_key(self) -> None:
        self._press_vk(
            _virtual_key_from_name(self.cfg.return_key),
            hold_median_ms=self.cfg.key_press_hold_ms,
            slot2=False,
        )

    def _sample_hold_ms(self, median_ms: int, slot2: bool) -> int:
        if not self._randomize_holds:
            return median_ms
        if slot2:
            return HumanDelay.key_hold(
                median_ms=self.cfg.slot2_key_press_hold_ms,
                sigma=self.cfg.slot2_key_hold_sigma,
                min_ms=self.cfg.slot2_key_hold_min_ms,
                max_ms=self.cfg.slot2_key_hold_max_ms,
            ).sample_ms()
        return HumanDelay.key_hold(
            median_ms=self.cfg.key_press_hold_ms,
            sigma=self.cfg.key_hold_sigma,
            min_ms=self.cfg.key_hold_min_ms,
            max_ms=self.cfg.key_hold_max_ms,
        ).sample_ms()

    def _press_vk(self, vk_code: int, hold_median_ms: int, slot2: bool) -> None:
        hold_ms = self._sample_hold_ms(hold_median_ms, slot2=slot2)
        self.backend.key_down(vk_code)
        time.sleep(max(0, hold_ms) / 1000.0)
        self.backend.key_up(vk_code)

    def _move_smooth(self, target_x: int, target_y: int) -> tuple[int, int]:
        start_x, start_y = self.get_position()
        current_x = start_x
        current_y = start_y

        mp = self.mouse_path_cfg
        if mp is not None and mp.enabled:
            waypoints = generate_path(
                (start_x, start_y),
                (target_x, target_y),
                mp,
            )
            for wx, wy, sleep_s in waypoints:
                self.backend.move_cursor(int(wx), int(wy))
                time.sleep(sleep_s)
        else:
            duration_s = max(0.02, self.cfg.move_duration_ms / 1000.0)
            steps = max(8, int(duration_s / 0.008))
            step_sleep = duration_s / steps

            for i in range(1, steps + 1):
                goal_x = int(round(start_x + (target_x - start_x) * (i / steps)))
                goal_y = int(round(start_y + (target_y - start_y) * (i / steps)))
                if goal_x != current_x or goal_y != current_y:
                    self.backend.move_cursor(int(goal_x), int(goal_y))
                    current_x, current_y = self.get_position()
                time.sleep(step_sleep)

        actual_x, actual_y = self.get_position()
        if (
            abs(actual_x - target_x) > _MOVE_VERIFY_TOLERANCE_PX
            or abs(actual_y - target_y) > _MOVE_VERIFY_TOLERANCE_PX
        ):
            self.backend.move_cursor(int(target_x), int(target_y))
            time.sleep(0.005)
            actual_x, actual_y = self.get_position()
        return actual_x, actual_y


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
