from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from enum import Enum, auto

from app.humanize import HumanDelay
from app.config import SessionConfig


class SessionPhase(Enum):
    FISHING = auto()
    MICRO_BREAK = auto()
    SHORT_BREAK = auto()
    LONG_BREAK = auto()
    SESSION_END = auto()


class SessionScheduler:
    """Work/break macro-session: fishing segments, breaks, hard cap on total fishing time."""

    def __init__(
        self,
        cfg: SessionConfig,
        clock_ms: Callable[[], int] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._clock_ms = clock_ms or (lambda: int(time.monotonic() * 1000))
        self._rng = rng or random
        self._session_started = False
        self._active = False
        self._phase = SessionPhase.SESSION_END
        self._phase_start_ms = 0
        self._phase_end_ms = 0
        self._fishing_segments_completed = 0
        self._total_fishing_ms = 0
        self._had_auto_before_break = False

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def active(self) -> bool:
        return self._active

    @property
    def session_started(self) -> bool:
        return self._session_started

    @property
    def total_fishing_ms(self) -> int:
        return self._total_fishing_ms

    def start(self, now_ms: int) -> None:
        self._session_started = True
        self._active = True
        self._total_fishing_ms = 0
        self._fishing_segments_completed = 0
        self._had_auto_before_break = False
        self._enter_fishing(now_ms)

    def reset(self) -> None:
        self._session_started = False
        self._active = False
        self._phase = SessionPhase.SESSION_END
        self._phase_start_ms = 0
        self._phase_end_ms = 0

    def snapshot_before_break_auto(self, had_auto: bool) -> None:
        self._had_auto_before_break = had_auto

    def should_resume_auto(self) -> bool:
        return self._had_auto_before_break

    def check(self, now_ms: int) -> SessionPhase:
        if not self._cfg.enabled:
            return SessionPhase.FISHING
        if not self._session_started:
            return SessionPhase.FISHING
        if not self._active:
            return SessionPhase.SESSION_END
        if self._phase == SessionPhase.SESSION_END:
            return SessionPhase.SESSION_END
        if now_ms < self._phase_end_ms:
            return self._phase
        return self._transition(now_ms)

    def _fishing_segment_duration_ms(self) -> int:
        med = max(1, self._cfg.fishing_segment_median_min) * 60_000
        lo = max(1, self._cfg.fishing_segment_min_min) * 60_000
        hi = max(lo, self._cfg.fishing_segment_max_min * 60_000)
        delay = HumanDelay(
            mu=math.log(med),
            sigma=0.25,
            min_ms=lo,
            max_ms=hi,
        )
        return delay.sample_ms()

    def _break_duration_ms(self, phase: SessionPhase) -> int:
        if phase == SessionPhase.MICRO_BREAK:
            lo = self._cfg.micro_break_min_s * 1000
            hi = max(lo, self._cfg.micro_break_max_s * 1000)
        elif phase == SessionPhase.SHORT_BREAK:
            lo = self._cfg.short_break_min_s * 1000
            hi = max(lo, self._cfg.short_break_max_s * 1000)
        else:
            lo = self._cfg.long_break_min_s * 1000
            hi = max(lo, self._cfg.long_break_max_s * 1000)
        return self._rng.randint(lo, hi)

    def _enter_fishing(self, now_ms: int) -> None:
        self._phase = SessionPhase.FISHING
        self._phase_start_ms = now_ms
        self._phase_end_ms = now_ms + self._fishing_segment_duration_ms()

    def _enter_break(self, now_ms: int, phase: SessionPhase) -> None:
        self._phase = phase
        self._phase_start_ms = now_ms
        self._phase_end_ms = now_ms + self._break_duration_ms(phase)

    def _transition(self, now_ms: int) -> SessionPhase:
        if self._phase == SessionPhase.FISHING:
            seg_ms = max(0, now_ms - self._phase_start_ms)
            self._total_fishing_ms += seg_ms
            self._fishing_segments_completed += 1

            max_total = max(0, self._cfg.max_session_min) * 60_000
            if max_total > 0 and self._total_fishing_ms >= max_total:
                self._phase = SessionPhase.SESSION_END
                self._active = False
                return self._phase

            every = max(1, self._cfg.long_break_every)
            if self._fishing_segments_completed % every == 0:
                self._enter_break(now_ms, SessionPhase.LONG_BREAK)
                return self._phase

            if self._rng.random() < self._cfg.short_break_probability:
                self._enter_break(now_ms, SessionPhase.SHORT_BREAK)
            else:
                self._enter_break(now_ms, SessionPhase.MICRO_BREAK)
            return self._phase

        self._enter_fishing(now_ms)
        return self._phase
