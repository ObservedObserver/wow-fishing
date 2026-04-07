from app.config import AppConfig, ControlConfig
from main import (
    _normalize_bite_action_mode,
    _perform_bite_action,
    _prime_slot2_after_reel,
    _prime_slot2_timer,
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


def test_prime_slot2_timer_sets_next_slot2_and_post_wait(monkeypatch) -> None:
    monkeypatch.setattr("main.random.randint", lambda low, high: 20_000)
    cfg = AppConfig.default()

    next_slot2_at_ms, next_cast_at_ms = _prime_slot2_timer(now_ms=4_000, cfg=cfg)

    assert next_slot2_at_ms == 624_000
    assert next_cast_at_ms == 12_000


def test_prime_slot2_after_reel_uses_settle_delay() -> None:
    cfg = AppConfig.default()

    next_slot2_at_ms = _prime_slot2_after_reel(now_ms=4_000, cfg=cfg)

    assert next_slot2_at_ms == 5_000
