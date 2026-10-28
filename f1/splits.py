"""Stable benchmark-level development/holdout assignment (not an official split)."""

import hashlib
import math


def benchmark_split(benchmark_id: str, holdout_fraction: float = 0.2) -> str:
    if not math.isfinite(holdout_fraction) or not 0 <= holdout_fraction <= 1:
        raise ValueError("holdout_fraction must be finite and in [0, 1]")
    value = int.from_bytes(hashlib.sha256(benchmark_id.encode()).digest(), "big")
    # Compare integer hashes to an exact rational threshold, including endpoints.
    numerator, denominator = holdout_fraction.as_integer_ratio()
    return "holdout" if value * denominator < numerator * 2**256 else "tune"
