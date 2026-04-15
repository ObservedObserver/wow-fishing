from app.config import AppConfig, ControlConfig, SessionConfig
from app.session import SessionScheduler
from main import (
    _bag_timer_conflicts_with_slot2,
    _max_runtime_ms,
    _normalize_bite_action_mode,
    _perform_bite_action,
    _prime_bag_timer,
    _prime_slot2_after_reel,
    _prime_slot2_timer,
    _roll_bag_interval_ms,
    _roll_bag_open_duration_ms,
    _resume_after_session_break,
    _roll_slot2_interval_ms,
    _runtime_limit_reached,
    _schedule_resume_cast_at_ms,
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

    out = _roll_slot2_interval_ms(cfg, fatigue=None)

    assert out == 620_000


def test_prime_slot2_timer_sets_next_slot2_and_post_wait(monkeypatch) -> None:
    monkeypatch.setattr("main.random.randint", lambda low, high: 20_000)
    cfg = AppConfig.default()

    next_slot2_at_ms, next_cast_at_ms = _prime_slot2_timer(
        now_ms=4_000, cfg=cfg, fatigue=None
    )

    assert next_slot2_at_ms == 624_000
    assert next_cast_at_ms == 12_000


def test_prime_slot2_after_reel_uses_settle_delay() -> None:
    cfg = AppConfig.default()

    next_slot2_at_ms = _prime_slot2_after_reel(now_ms=4_000, cfg=cfg)

    assert next_slot2_at_ms == 5_000


def test_roll_bag_interval_uses_base_plus_jitter(monkeypatch) -> None:
    monkeypatch.setattr("main.random.randint", lambda low, high: 45_000)
    cfg = AppConfig.default()

    out = _roll_bag_interval_ms(cfg)

    assert out == 225_000


def test_prime_bag_timer_sets_next_bag_action(monkeypatch) -> None:
    monkeypatch.setattr("main.random.randint", lambda low, high: 30_000)
    cfg = AppConfig.default()

    next_bag_at_ms = _prime_bag_timer(now_ms=4_000, cfg=cfg)

    assert next_bag_at_ms == 214_000


def test_roll_bag_open_duration_within_configured_bounds(monkeypatch) -> None:
    monkeypatch.setattr("main.random.randint", lambda low, high: 4_200)
    cfg = AppConfig.default()

    out = _roll_bag_open_duration_ms(cfg)

    assert out == 4_200


def test_bag_timer_conflict_detects_nearby_slot2() -> None:
    cfg = AppConfig.default()

    assert _bag_timer_conflicts_with_slot2(
        now_ms=100_000,
        cfg=cfg,
        slot2_next_at_ms=105_500,
        slot2_pending_use_at_ms=None,
    ) is True


def test_bag_timer_conflict_ignores_distant_slot2() -> None:
    cfg = AppConfig.default()

    assert _bag_timer_conflicts_with_slot2(
        now_ms=100_000,
        cfg=cfg,
        slot2_next_at_ms=120_000,
        slot2_pending_use_at_ms=None,
    ) is False


def test_schedule_resume_cast_uses_initial_delay() -> None:
    cfg = AppConfig.default()

    next_cast_at_ms = _schedule_resume_cast_at_ms(now_ms=4_000, cfg=cfg)

    assert next_cast_at_ms == 4_500


def test_max_runtime_ms_uses_hours_from_config() -> None:
    cfg = AppConfig.default()

    assert _max_runtime_ms(cfg) == 9_000_000


def test_runtime_limit_reached_only_after_threshold() -> None:
    cfg = AppConfig.default()

    assert _runtime_limit_reached(8_999_999, 0, cfg) is False
    assert _runtime_limit_reached(9_000_000, 0, cfg) is True
    assert _runtime_limit_reached(9_000_000, None, cfg) is False


def test_resume_after_session_break_rearms_auto_cast_when_needed() -> None:
    cfg = AppConfig.default()
    session = SessionScheduler(SessionConfig(enabled=True))
    session.snapshot_before_break_auto(True)

    auto_enabled, next_cast_at_ms = _resume_after_session_break(
        now_ms=4_000,
        cfg=cfg,
        session=session,
    )

    assert auto_enabled is True
    assert next_cast_at_ms == 4_500
    assert session.should_resume_auto() is False


def test_resume_after_session_break_stays_idle_when_auto_was_off() -> None:
    cfg = AppConfig.default()
    session = SessionScheduler(SessionConfig(enabled=True))
    session.snapshot_before_break_auto(False)

    auto_enabled, next_cast_at_ms = _resume_after_session_break(
        now_ms=4_000,
        cfg=cfg,
        session=session,
    )

    assert auto_enabled is False
    assert next_cast_at_ms is None
