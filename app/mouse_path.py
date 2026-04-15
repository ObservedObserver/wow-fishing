from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import MousePathConfig


def fitts_duration_ms(
    distance_px: float,
    target_width_px: float,
    a: float,
    b: float,
) -> float:
    """Fitts' Law: MT = a + b * log2(2D/W)."""
    if distance_px < 1:
        return a
    id_bits = math.log2(max(1.0, 2.0 * distance_px / max(1.0, target_width_px)))
    return a + b * id_bits


def _cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    a = u * u * u
    b = 3 * u * u * t
    c = 3 * u * t * t
    d = t * t * t
    x = a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0]
    y = a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]
    return x, y


def generate_path(
    start: tuple[int, int],
    end: tuple[int, int],
    cfg: MousePathConfig,
    rng: random.Random | None = None,
) -> list[tuple[int, int, float]]:
    """Returns list of (x, y, sleep_s) waypoints for cursor movement."""
    r = rng or random
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx = ex - sx
    dy = ey - sy
    dist = math.hypot(dx, dy)
    target_w = float(max(1, cfg.target_width_px))

    base_mt = fitts_duration_ms(dist, target_w, cfg.fitts_a_ms, cfg.fitts_b_ms)
    mt_ms = base_mt * r.uniform(0.85, 1.15)
    mt_ms = max(cfg.min_move_duration_ms, min(cfg.max_move_duration_ms, mt_ms))

    p0 = (sx, sy)
    p3 = (ex, ey)
    perp_scale = r.gauss(0, cfg.arc_sigma) * max(1.0, dist)
    nx = -dy / max(1.0, dist)
    ny = dx / max(1.0, dist)
    mid_x = (sx + ex) / 2.0 + perp_scale * nx
    mid_y = (sy + ey) / 2.0 + perp_scale * ny
    p1 = (
        mid_x + r.gauss(0, dist * 0.05),
        mid_y + r.gauss(0, dist * 0.05),
    )
    p2 = (
        ex + r.gauss(0, dist * 0.02),
        ey + r.gauss(0, dist * 0.02),
    )

    steps = max(12, int(mt_ms / 8))
    raw_points: list[tuple[int, int]] = []
    for t in range(steps + 1):
        u = t / steps
        px, py = _cubic_bezier(p0, p1, p2, p3, u)
        raw_points.append((int(round(px)), int(round(py))))

    speed_profile = [
        math.sin(math.pi * t / steps) ** cfg.speed_gamma for t in range(steps + 1)
    ]
    total = sum(speed_profile) or 1.0
    sleeps = [(s / total) * (mt_ms / 1000.0) for s in speed_profile]

    for i, (px, py) in enumerate(raw_points):
        jx = r.gauss(0, cfg.tremor_amplitude_px)
        jy = r.gauss(0, cfg.tremor_amplitude_px)
        raw_points[i] = (int(px + jx), int(py + jy))

    if dist > cfg.overshoot_min_dist_px and r.random() < cfg.overshoot_prob:
        overshoot_dist = r.uniform(3.0, max(5.0, dist * 0.08))
        angle = math.atan2(dy, dx)
        os_x = int(ex + overshoot_dist * math.cos(angle))
        os_y = int(ey + overshoot_dist * math.sin(angle))
        raw_points.append((os_x, os_y))
        sleeps.append(r.uniform(0.02, 0.06))
        raw_points.append(
            (
                int(ex + r.randint(-1, 1)),
                int(ey + r.randint(-1, 1)),
            )
        )
        sleeps.append(r.uniform(0.015, 0.04))

    return [(p[0], p[1], s) for p, s in zip(raw_points, sleeps, strict=True)]
