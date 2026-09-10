from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np


TimingCenter = Union[float, str]


def parse_timing_center(value: str | float | int | None) -> TimingCenter:
    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"auto", "peak"}:
        return "auto"
    return float(text)


def estimate_timing_peak_center(
    rows: Iterable[dict[str, float]],
    *,
    lo: float = -3.5,
    hi: float = 3.5,
    bins: int = 140,
    refine_half_width: float = 0.25,
) -> float:
    values: list[float] = []
    for row in rows:
        for key in ("t1", "t2"):
            if key not in row:
                continue
            value = float(row[key])
            if math.isfinite(value) and lo <= value <= hi:
                values.append(value)
    if not values:
        return 0.0

    arr = np.asarray(values, dtype=float)
    hist, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    if hist.size == 0 or int(hist.max()) <= 0:
        return 0.0

    peak_bin = int(np.argmax(hist))
    center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    near = arr[np.abs(arr - center) <= refine_half_width]
    if near.size:
        center = float(np.mean(near))
    return float(center)


def resolve_timing_center(rows: Iterable[dict[str, float]], timing_center: TimingCenter) -> float:
    if timing_center == "auto":
        return estimate_timing_peak_center(rows)
    return float(timing_center)


def passes_timing_window(row: dict[str, float], timing_window: float | None, timing_center: float) -> bool:
    if timing_window is None:
        return True
    if "t1" not in row or "t2" not in row:
        return False
    return abs(float(row["t1"]) - timing_center) <= timing_window and abs(float(row["t2"]) - timing_center) <= timing_window
