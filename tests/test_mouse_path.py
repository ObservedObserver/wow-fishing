from __future__ import annotations

import math
import random

from app.config import MousePathConfig
from app.mouse_path import fitts_duration_ms, generate_path


def test_fitts_duration_increases_with_distance() -> None:
    a = fitts_duration_ms(50.0, 20.0, 50.0, 150.0)
    b = fitts_duration_ms(400.0, 20.0, 50.0, 150.0)
    assert b > a


def test_generate_path_curved_longer_than_straight() -> None:
    cfg = MousePathConfig(
        enabled=True,
        overshoot_prob=0.0,
        min_move_duration_ms=50,
        max_move_duration_ms=2000,
    )
    path = generate_path((100, 100), (300, 120), cfg, rng=random.Random(1))
    straight = math.hypot(200, 20)
    length = sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )
    assert length >= straight * 0.99
