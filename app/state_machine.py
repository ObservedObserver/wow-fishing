from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from app.config import TimingConfig
from app.audio import AudioEvent


class BotState(Enum):
    IDLE = auto()
    PENDING_LOCATE = auto()
    TRACKING = auto()


@dataclass(slots=True)
class BotDecision:
    cast_id: int | None = None
    should_reel: bool = False
    should_recast: bool = False
    reason: str = ""


class FishingStateMachine:
    def __init__(self, cfg: TimingConfig, action_lock_ms: int | None = None) -> None:
        self.cfg = cfg
        self.action_lock_ms = max(0, action_lock_ms or cfg.click_cooldown_ms)
        self._cast_sequence = 0
        self.state = BotState.IDLE
        self.cast_id: int | None = None
        self.cast_ts_ms: int | None = None
        self.locate_due_ts_ms: int | None = None
        self.locate_attempt = 0
        self.last_action_ms = -10_000

    def reset(self) -> None:
        self.state = BotState.IDLE
        self.cast_id = None
        self.cast_ts_ms = None
        self.locate_due_ts_ms = None
        self.locate_attempt = 0

    def on_cast(self, now_ms: int) -> int:
        self._cast_sequence += 1
        self.cast_id = self._cast_sequence
        self.state = BotState.PENDING_LOCATE
        self.cast_ts_ms = now_ms
        self.locate_due_ts_ms = now_ms + self.cfg.key_detect_delay_ms
        self.locate_attempt = 1
        return self.cast_id

    def should_attempt_locate(self, now_ms: int) -> bool:
        return (
            self.state == BotState.PENDING_LOCATE
            and self.locate_due_ts_ms is not None
            and now_ms >= self.locate_due_ts_ms
        )

    def on_locate_success(self) -> None:
        self.state = BotState.TRACKING
        self.locate_due_ts_ms = None
        self.locate_attempt = 0

    def on_locate_failure(self, now_ms: int) -> BotDecision:
        cast_id = self.cast_id
        if self.state != BotState.PENDING_LOCATE:
            return BotDecision(cast_id=cast_id, reason="locate_not_pending")

        if self.locate_attempt < max(1, self.cfg.key_retry_max_attempts):
            self.locate_attempt += 1
            self.locate_due_ts_ms = now_ms + self.cfg.key_retry_interval_ms
            return BotDecision(cast_id=cast_id, reason="locate_retry")

        self.reset()
        return BotDecision(
            cast_id=cast_id,
            should_recast=self.cfg.recast_on_miss,
            reason="locate_failed",
        )

    def on_reel(self, now_ms: int) -> None:
        self.last_action_ms = now_ms
        self.reset()

    def update(self, now_ms: int, audio_event: AudioEvent | None) -> BotDecision:
        cast_id = self.cast_id
        if self.state == BotState.IDLE or self.cast_ts_ms is None:
            return BotDecision(cast_id=cast_id, reason="idle")

        elapsed = now_ms - self.cast_ts_ms
        if self.state == BotState.PENDING_LOCATE:
            return BotDecision(cast_id=cast_id, reason="pending_locate")

        if elapsed >= max(0, self.cfg.max_cast_lifetime_ms):
            self.reset()
            return BotDecision(cast_id=cast_id, should_recast=True, reason="cast_timeout")

        if elapsed < max(0, self.cfg.ignore_after_cast_ms):
            return BotDecision(cast_id=cast_id, reason="ignore_after_cast")

        if elapsed < max(0, self.cfg.bite_window_start_ms):
            return BotDecision(cast_id=cast_id, reason="before_bite_window")

        if elapsed > max(0, self.cfg.bite_window_end_ms):
            return BotDecision(cast_id=cast_id, reason="after_bite_window")

        if audio_event is None:
            return BotDecision(cast_id=cast_id, reason="waiting_audio")

        if (now_ms - self.last_action_ms) < self.action_lock_ms:
            return BotDecision(cast_id=cast_id, reason="action_lock")

        return BotDecision(cast_id=cast_id, should_reel=True, reason="audio_bite")
