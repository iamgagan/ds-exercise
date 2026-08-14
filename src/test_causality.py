"""
Regression test for the property that matters most operationally: the detector
must be *causal*. A score assigned to month t may not depend on any data from
month t+1 onward, or the offline evaluation is measuring a model that cannot
exist in production.

Run: python src/test_causality.py   (or: pytest src/test_causality.py)

This is a real end-to-end check on the actual scoring path, not a unit test of
a helper: it scores the full panel, scores a truncated panel, and requires the
surviving months to be bit-identical. An earlier version of `build_scores`
pooled the seasonal index and the peer sigma over the whole panel and failed
this test on ~66% of rows (max drift 7.3 z, against an alert threshold of ~1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generator import generate_dataset
from anomaly_detection import build_scores

SCORE_COLS = ["baseline", "residual", "sigma", "sudden_z", "gradual_z", "alert_score"]
KEYS = ["brand", "channel", "month"]


def check_no_lookahead(cut_months=(14, 20, 25), tol=1e-9) -> bool:
    panel = generate_dataset()["panel"]
    full = build_scores(panel)
    ok = True

    for cut in cut_months:
        trunc = build_scores(panel[panel.month_idx <= cut].copy())
        merged = full.merge(trunc, on=KEYS, suffixes=("_full", "_trunc"))
        merged = merged[merged["month_idx_full"] <= cut]

        worst_col, worst_diff, worst_n = None, 0.0, 0
        for col in SCORE_COLS:
            a, b = merged[f"{col}_full"], merged[f"{col}_trunc"]
            both_nan = a.isna() & b.isna()
            diff = (a - b).abs().where(~both_nan)
            # a value present in one run but absent in the other is also a violation
            mismatched_nan = (a.isna() ^ b.isna()).sum()
            n_bad = int((diff > tol).sum()) + int(mismatched_nan)
            if n_bad and (diff.max() or 0) >= worst_diff:
                worst_col, worst_diff, worst_n = col, float(diff.max() or 0), n_bad

        if worst_col is None:
            print(f"  cut@{cut:>3}: PASS  ({len(merged)} surviving rows identical)")
        else:
            ok = False
            print(f"  cut@{cut:>3}: FAIL  {worst_n} rows differ; worst column "
                  f"'{worst_col}' max |diff|={worst_diff:.4f}")
    return ok


def check_direction_gating() -> bool:
    """`direction='up'` must never let a downward move outscore the threshold band."""
    panel = generate_dataset()["panel"]
    up = build_scores(panel, direction="up").dropna(subset=["alert_score"])
    leaked = up[(up["alert_score"] > 0.5) & (up["residual"] < 0) & (up["gradual_z"].isna() | (up["gradual_z"] < 0))]
    if len(leaked) == 0:
        print(f"  direction='up': PASS  (no downward-only month scores above 0.5)")
        return True
    print(f"  direction='up': FAIL  {len(leaked)} downward months scored > 0.5")
    return False


if __name__ == "__main__":
    print("Causality regression tests")
    print("-" * 60)
    results = [check_no_lookahead(), check_direction_gating()]
    print("-" * 60)
    if all(results):
        print("ALL PASS - scores for month t depend only on data before t")
        sys.exit(0)
    print("FAILURES - the detector is not deployable as scored")
    sys.exit(1)
