from app.config import AppConfig, ControlConfig
from main import (
    _delay_next_cast_for_slot2,
    _normalize_bite_action_mode,
    _perform_bite_action,
    _roll_slot2_interval_ms,
)


class _FakeMouse:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def right_click(self) -> None:
        self.actions.append("right_click")

    def press_interaction_key(self) -> None:
        self.actions.append("interaction_key")


def test_normalize_bite_action_mode_accepts_aliases() -> None:
    assert _normalize_bite_action_mode("mouse") == "mouse"
    assert _normalize_bite_action_mode("right-click") == "mouse"
    assert _normalize_bite_action_mode("interaction_key") == "interact_hotkey"


def test_perform_bite_action_uses_mouse_right_click() -> None:
    mouse = _FakeMouse()
    action = _perform_bite_action(
        mouse=mouse,  # type: ignore[arg-type]
        cfg=ControlConfig(bite_action_mode="mouse"),
    )

    assert action == "mouse_right_click"
    assert mouse.actions == ["right_click"]


def test_perform_bite_action_uses_interaction_hotkey() -> None:
    mouse = _FakeMouse()
    action = _perform_bite_action(
        mouse=mouse,  # type: ignore[arg-type]
        cfg=ControlConfig(bite_action_mode="interact_hotkey", interaction_key="F12"),
    )

    assert action == "interaction_hotkey:F12"
    assert mouse.actions == ["interaction_key"]


def test_roll_slot2_interval_uses_base_plus_jitter(monkeypatch) -> None:
    monkeypatch.setattr("main.random.randint", lambda low, high: 20_000)
    cfg = AppConfig.default()

    out = _roll_slot2_interval_ms(cfg)

    assert out == 620_000


def test_delay_next_cast_for_slot2_enforces_post_wait() -> None:
    assert _delay_next_cast_for_slot2(5_000, now_ms=4_000, post_use_wait_ms=8_000) == 12_000
    assert _delay_next_cast_for_slot2(20_000, now_ms=4_000, post_use_wait_ms=8_000) == 20_000
    assert _delay_next_cast_for_slot2(None, now_ms=4_000, post_use_wait_ms=8_000) is None
