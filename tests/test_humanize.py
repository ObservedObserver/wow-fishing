from __future__ import annotations

import math
import statistics

from app.config import HumanizeConfig
from app.humanize import FatigueModel, HumanDelay


def test_human_delay_sample_ms_within_bounds() -> None:
    cfg = HumanizeConfig(
        bite_reaction_median_ms=500,
        bite_reaction_sigma=0.35,
        bite_reaction_min_ms=100,
        bite_reaction_max_ms=800,
    )
    dist = HumanDelay.bite_reaction(cfg)
    for _ in range(500):
        s = dist.sample_ms()
        assert 100 <= s <= 800


def test_human_delay_log_normal_is_right_skewed_large_sample() -> None:
    cfg = HumanizeConfig()
    dist = HumanDelay.bite_reaction(cfg)
    samples = [dist.sample_ms() for _ in range(2000)]
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    assert mean >= median * 0.95


def test_fatigue_model_increases_expected_value_with_mock_clock() -> None:
    t = [0]

    def clock() -> int:
        return t[0]

    cfg = HumanizeConfig(
        fatigue_drift_per_hour=1.0,
        fatigue_max_drift=2.0,
        fatigue_spike_prob=0.0,
    )
    base = HumanDelay(
        mu=math.log(400),
        sigma=0.2,
        min_ms=200,
        max_ms=900,
    )
    fm = FatigueModel(cfg, clock_ms=clock)
    early = [fm.adjusted_sample(base) for _ in range(300)]
    t[0] = int(3.6e6)
    late = [fm.adjusted_sample(base) for _ in range(300)]
    assert statistics.mean(late) >= statistics.mean(early) * 0.98
