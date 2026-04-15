from app.audio import AudioEvent
from app.config import TimingConfig
from app.state_machine import FishingStateMachine


def test_state_machine_schedules_locate_and_tracks_success() -> None:
    cfg = TimingConfig(
        ignore_after_cast_ms=100,
        bite_window_start_ms=0,
        key_detect_delay_ms=150,
        key_retry_interval_ms=75,
        key_retry_max_attempts=3,
    )
    machine = FishingStateMachine(cfg, action_lock_ms=250)

    cast_id = machine.on_cast(0)

    assert cast_id == 1
    assert not machine.should_attempt_locate(149)
    assert machine.should_attempt_locate(150)

    machine.on_locate_success()
    decision = machine.update(now_ms=180, audio_event=None)

    assert decision.reason == "waiting_audio"
    assert decision.cast_id == 1


def test_state_machine_retries_then_recasts_after_locate_failures() -> None:
    cfg = TimingConfig(key_detect_delay_ms=100, key_retry_interval_ms=50, key_retry_max_attempts=2)
    machine = FishingStateMachine(cfg)
    machine.on_cast(0)

    retry = machine.on_locate_failure(100)
    assert retry.reason == "locate_retry"
    assert machine.locate_attempt == 2
    assert machine.locate_due_ts_ms == 150

    failed = machine.on_locate_failure(150)
    assert failed.reason == "locate_failed"
    assert failed.should_recast
    assert machine.cast_id is None


def test_state_machine_reels_on_audio() -> None:
    cfg = TimingConfig(
        ignore_after_cast_ms=100,
        bite_window_start_ms=200,
        bite_window_end_ms=500,
        max_cast_lifetime_ms=5000,
    )
    machine = FishingStateMachine(cfg, action_lock_ms=500)
    machine.on_cast(0)
    machine.on_locate_success()

    ev = AudioEvent(ts_ms=300, energy=0.2, threshold=0.1)
    decision = machine.update(now_ms=300, audio_event=ev)

    assert decision.should_reel
    assert decision.reason == "audio_bite"


def test_state_machine_blocks_audio_before_bite_window() -> None:
    cfg = TimingConfig(
        ignore_after_cast_ms=100,
        bite_window_start_ms=200,
        bite_window_end_ms=500,
        max_cast_lifetime_ms=5000,
    )
    machine = FishingStateMachine(cfg)
    machine.on_cast(0)
    machine.on_locate_success()

    ev = AudioEvent(ts_ms=150, energy=0.2, threshold=0.1)
    decision = machine.update(now_ms=150, audio_event=ev)

    assert not decision.should_reel
    assert decision.reason == "before_bite_window"


def test_state_machine_blocks_audio_after_bite_window() -> None:
    cfg = TimingConfig(
        ignore_after_cast_ms=100,
        bite_window_start_ms=200,
        bite_window_end_ms=500,
        max_cast_lifetime_ms=5000,
    )
    machine = FishingStateMachine(cfg)
    machine.on_cast(0)
    machine.on_locate_success()

    ev = AudioEvent(ts_ms=600, energy=0.2, threshold=0.1)
    decision = machine.update(now_ms=600, audio_event=ev)

    assert not decision.should_reel
    assert decision.reason == "after_bite_window"


def test_state_machine_requests_recast_after_timeout() -> None:
    cfg = TimingConfig(ignore_after_cast_ms=100, max_cast_lifetime_ms=300)
    machine = FishingStateMachine(cfg)
    machine.on_cast(0)
    machine.on_locate_success()

    decision = machine.update(now_ms=300, audio_event=None)

    assert decision.should_recast
    assert decision.reason == "cast_timeout"
