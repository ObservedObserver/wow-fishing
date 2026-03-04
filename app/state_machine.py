from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from app.audio import AudioEvent
from app.config import TimingConfig
from app.vision import Detection


class BotState(Enum):
    IDLE = auto()
    CASTED = auto()
    WAIT_BITE = auto()
    BITE_CANDIDATE = auto()
    CLICKED = auto()


@dataclass(slots=True)
class BotDecision:
    should_click: bool = False
    click_x: int = 0
    click_y: int = 0
    reason: str = ""


class FishingStateMachine:
    def __init__(self, cfg: TimingConfig) -> None:
        self.cfg = cfg
        self.state = BotState.IDLE
        self.cast_ts_ms = 0
        self.last_click_ms = -10_000
        self._vision_hits = 0
        self._vision_needed = 2
        self._candidate_ts_ms = 0

    def on_cast(self, now_ms: int) -> None:
        self.state = BotState.CASTED
        self.cast_ts_ms = now_ms
        self._vision_hits = 0

    def update(
        self,
        now_ms: int,
        audio_event: AudioEvent | None,
        visual_detection: Detection | None,
    ) -> BotDecision:
        if (now_ms - self.last_click_ms) < self.cfg.click_cooldown_ms:
            return BotDecision(reason="cooldown")

        if self.state == BotState.IDLE:
            return BotDecision(reason="idle")

        elapsed = now_ms - self.cast_ts_ms
        if self.state == BotState.CASTED and elapsed >= self.cfg.ignore_after_cast_ms:
            self.state = BotState.WAIT_BITE

        if self.state == BotState.WAIT_BITE:
            in_window = self.cfg.bite_window_start_ms <= elapsed <= self.cfg.bite_window_end_ms
            if audio_event is not None and in_window:
                self.state = BotState.BITE_CANDIDATE
                self._candidate_ts_ms = now_ms
                self._vision_hits = 0
            elif elapsed > self.cfg.bite_window_end_ms:
                self.state = BotState.IDLE
            return BotDecision(reason="wait_bite")

        if self.state == BotState.BITE_CANDIDATE:
            if now_ms - self._candidate_ts_ms > 1_500:
                self.state = BotState.WAIT_BITE
                self._vision_hits = 0
                return BotDecision(reason="candidate_timeout")

            if visual_detection is None:
                return BotDecision(reason="no_visual")

            self._vision_hits += 1
            if self._vision_hits < self._vision_needed:
                return BotDecision(reason="visual_confirming")

            self.state = BotState.CLICKED
            self.last_click_ms = now_ms
            return BotDecision(
                should_click=True,
                click_x=visual_detection.x,
                click_y=visual_detection.y,
                reason="bite_confirmed",
            )

        if self.state == BotState.CLICKED:
            self.state = BotState.IDLE
            return BotDecision(reason="clicked")

        return BotDecision(reason="noop")

