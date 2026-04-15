from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from app.config import HumanizeConfig


class HumanDelay:
    """Generates human-like delay values using log-normal distribution."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        min_ms: int,
        max_ms: int,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.min_ms = min_ms if min_ms <= max_ms else max_ms
        self.max_ms = max_ms if max_ms >= min_ms else min_ms

    def sample_ms(self) -> int:
        if self.min_ms >= self.max_ms:
            return self.min_ms
        raw = random.lognormvariate(self.mu, self.sigma)
        return max(self.min_ms, min(self.max_ms, int(round(raw))))

    @staticmethod
    def bite_reaction(cfg: HumanizeConfig) -> HumanDelay:
        return HumanDelay(
            mu=math.log(max(1, cfg.bite_reaction_median_ms)),
            sigma=cfg.bite_reaction_sigma,
            min_ms=cfg.bite_reaction_min_ms,
            max_ms=cfg.bite_reaction_max_ms,
        )

    @staticmethod
    def cast_interval(cfg: HumanizeConfig) -> HumanDelay:
        return HumanDelay(
            mu=math.log(max(1, cfg.cast_interval_median_ms)),
            sigma=cfg.cast_interval_sigma,
            min_ms=cfg.cast_interval_min_ms,
            max_ms=cfg.cast_interval_max_ms,
        )

    @staticmethod
    def slot2_jitter(cfg: HumanizeConfig) -> HumanDelay:
        return HumanDelay(
            mu=math.log(max(1, cfg.slot2_jitter_median_ms)),
            sigma=cfg.slot2_jitter_sigma,
            min_ms=cfg.slot2_jitter_min_ms,
            max_ms=cfg.slot2_jitter_max_ms,
        )

    @staticmethod
    def key_hold(
        median_ms: int,
        sigma: float,
        min_ms: int,
        max_ms: int,
    ) -> HumanDelay:
        return HumanDelay(
            mu=math.log(max(1, median_ms)),
            sigma=sigma,
            min_ms=min_ms,
            max_ms=max_ms,
        )


class FatigueModel:
    """Shifts delay distributions over time to simulate fatigue."""

    def __init__(
        self,
        cfg: HumanizeConfig,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._cfg = cfg
        self._clock_ms = clock_ms or (lambda: int(time.monotonic() * 1000))
        self.start_ms = self._clock_ms()
        self.spike_multiplier_range = (
            cfg.fatigue_spike_multiplier_min,
            cfg.fatigue_spike_multiplier_max,
        )

    def now_ms(self) -> int:
        return self._clock_ms()

    def reset(self) -> None:
        self.start_ms = self._clock_ms()

    def adjusted_sample(self, base: HumanDelay) -> int:
        hours_elapsed = (self._clock_ms() - self.start_ms) / 3_600_000.0
        drift = min(
            self._cfg.fatigue_max_drift,
            self._cfg.fatigue_drift_per_hour * max(0.0, hours_elapsed),
        )
        shifted = HumanDelay(
            mu=base.mu + drift,
            sigma=base.sigma,
            min_ms=base.min_ms,
            max_ms=base.max_ms,
        )
        val = shifted.sample_ms()
        if random.random() < self._cfg.fatigue_spike_prob:
            mult = random.uniform(self.spike_multiplier_range[0], self.spike_multiplier_range[1])
            val = min(shifted.max_ms, int(val * mult))
        return val
