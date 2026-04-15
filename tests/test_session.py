from __future__ import annotations

import random

from app.config import SessionConfig
from app.session import SessionPhase, SessionScheduler


def test_session_disabled_check_returns_fishing() -> None:
    cfg = SessionConfig(enabled=False)
    s = SessionScheduler(cfg, clock_ms=lambda: 1000)
    assert s.check(1000) == SessionPhase.FISHING


def test_session_start_then_fishing_phase_until_end() -> None:
    cfg = SessionConfig(
        enabled=True,
        fishing_segment_median_min=1,
        fishing_segment_min_min=1,
        fishing_segment_max_min=1,
        max_session_min=500,
        long_break_every=99,
    )
    rng = random.Random(42)
    t = [0]

    s = SessionScheduler(cfg, clock_ms=lambda: t[0], rng=rng)

    s.start(0)
    assert s.check(0) == SessionPhase.FISHING
    t[0] = 59_000
    assert s.check(59_000) == SessionPhase.FISHING
    t[0] = 60_001
    phase = s.check(60_001)
    assert phase in (
        SessionPhase.MICRO_BREAK,
        SessionPhase.SHORT_BREAK,
        SessionPhase.LONG_BREAK,
    )
