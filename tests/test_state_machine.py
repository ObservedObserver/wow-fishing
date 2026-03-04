from app.audio import AudioEvent
from app.config import TimingConfig
from app.state_machine import FishingStateMachine
from app.vision import Detection


def test_state_machine_clicks_after_audio_and_2_vision_hits() -> None:
    cfg = TimingConfig(
        ignore_after_cast_ms=100,
        bite_window_start_ms=200,
        bite_window_end_ms=5000,
        click_cooldown_ms=200,
    )
    machine = FishingStateMachine(cfg)
    machine.on_cast(0)

    # Move into WAIT_BITE.
    _ = machine.update(now_ms=120, audio_event=None, visual_detection=None)
    # Audio bite trigger inside bite window.
    ev = AudioEvent(ts_ms=350, energy=0.2, threshold=0.1)
    _ = machine.update(now_ms=350, audio_event=ev, visual_detection=None)

    # Two visual confirmations required.
    d1 = Detection(x=100, y=200, conf=0.7)
    r1 = machine.update(now_ms=360, audio_event=None, visual_detection=d1)
    assert not r1.should_click

    d2 = Detection(x=101, y=201, conf=0.8)
    r2 = machine.update(now_ms=370, audio_event=None, visual_detection=d2)
    assert r2.should_click
    assert r2.click_x == 101
    assert r2.click_y == 201

