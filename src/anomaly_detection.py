"""
Anomaly detector for monthly brand x channel media metrics.

Baseline (deliberately NOT mean +/- 2 sigma):

    expected_t = own_trailing_trend_t (shrunk toward peer-cohort trend for
                 short-history brand-channels) + channel-level seasonal index
    residual_t = log(metric_t) - expected_t
    sigma_t    = trailing robust MAD of residual (shrunk toward peer MAD)
    robust_z_t = residual_t / sigma_t

STRICT CAUSALITY: every quantity used to score month t is estimated from
data strictly before t - own trend, own sigma, the pooled channel seasonal
index, the peer trend, the peer sigma, and every fallback. Nothing is fit on
the full panel. This is enforced by aggregating each pooled quantity to the
(group, month_idx) level first and then shifting by one *month* (not one
row, which would leak across brands sharing a month), and is regression-
tested in `test_causality.py`: truncating the panel must leave every
surviving month's score bit-identical.

The cost of that discipline is real and is not hidden: the seasonal index
for a given (channel, calendar-month) needs at least one prior occurrence of
that calendar month, so it is zero for the first 12 months of the panel and
thin in year 2. That weakens year-1 detection. An earlier version sidestepped
this by pooling the seasonal index over the whole panel, which inflated
measured performance via lookahead - the honest, deployable numbers are the
ones reported now.

Two parallel scores are computed from the same residual series:
  - sudden_z: single-month robust z-score (catches 1-month step spikes)
  - gradual_z: 3-month trailing mean of robust z, only credited when the
    three months broadly agree in sign (catches slow 3-month ramps without
    over-crediting a single noisy month)

DIRECTION: the business problem is *spike* investigation, and every cause
type in the taxonomy raises the headline metric. The detector is therefore
one-sided by default (`direction="up"`): a ROAS collapse is a real
operational event but it is not what this system diagnoses, and scoring it
on |z| spent half the analyst review budget on months that no cause in the
taxonomy could ever explain. `direction="both"` restores the two-sided
behavior for callers that want it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # scales MAD to be comparable to std under normality


def _robust_mad(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    med = x.median()
    return MAD_SCALE * (x - med).abs().median()


def _causal_pooled_mad(df: pd.DataFrame, group_cols: list[str], min_obs: int) -> pd.Series:
    """
    Robust MAD of `residual` pooled over `group_cols`, using only months strictly
    earlier than the row's own month. Returns NaN where fewer than `min_obs`
    prior observations exist, so the caller can fall back to a wider pool.

    Deliberately a loop over (group x month): the panel is ~1k rows, and a
    vectorized expanding-MAD would be materially harder to verify as causal,
    which is the whole point of this function.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("month_idx")
        for t in g["month_idx"].unique():
            prior = g.loc[g["month_idx"] < t, "residual"].dropna()
            if len(prior) >= min_obs:
                out.loc[g.index[g["month_idx"] == t]] = _robust_mad(prior)
    return out


def build_scores(panel: pd.DataFrame, metric: str = "roas", min_own_history: int = 3,
                  direction: str = "up") -> pd.DataFrame:
    if direction not in ("up", "down", "both"):
        raise ValueError(f"direction must be 'up', 'down' or 'both', got {direction!r}")

    df = panel.copy()
    df["cal_month"] = pd.PeriodIndex(df["month"], freq="M").month
    df["log_metric"] = np.log(df[metric].clip(lower=1e-6))
    df = df.sort_values("month_idx")

    # --- own trailing trend: robust (median) local level, causal ---
    df["own_n_hist"] = df.groupby(["brand", "channel"]).cumcount()
    df["own_trailing_trend"] = (
        df.groupby(["brand", "channel"])["log_metric"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=min_own_history).median())
    )
    df["detrended"] = df["log_metric"] - df["own_trailing_trend"]

    # --- channel seasonal index, pooled across brands but STRICTLY causal: for calendar month M
    # at month_idx t, the median of brand-averaged detrended values from *earlier occurrences* of
    # M only. Aggregating to (channel, cal_month, month_idx) before shifting is what stops brands
    # sharing a month from leaking into each other. Median (not mean) so injected spikes don't
    # drag the typical seasonal shape. Zero for the first occurrence - see module docstring.
    occ = (
        df.dropna(subset=["detrended"])
        .groupby(["channel", "cal_month", "month_idx"], as_index=False)["detrended"].mean()
        .sort_values("month_idx")
    )
    occ["seasonal_index"] = (
        occ.groupby(["channel", "cal_month"])["detrended"]
        .transform(lambda s: s.shift(1).expanding().median())
    )
    df = df.merge(occ[["channel", "cal_month", "month_idx", "seasonal_index"]],
                  on=["channel", "cal_month", "month_idx"], how="left")
    df["seasonal_index"] = df["seasonal_index"].fillna(0.0)

    # --- peer (same channel, other brands) trailing trend: fallback for cold-start
    # brand-channels. Aggregated to (channel, month_idx) before shift(1) so the shift is by one
    # *month*, not one row - shifting rows would let a brand see another brand's same-month value.
    pm = (
        df.dropna(subset=["log_metric"])
        .groupby(["channel", "month_idx"], as_index=False)["log_metric"].mean()
        .sort_values("month_idx")
    )
    pm["peer_trailing_trend"] = (
        pm.groupby("channel")["log_metric"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    )
    df = df.merge(pm[["channel", "month_idx", "peer_trailing_trend"]],
                  on=["channel", "month_idx"], how="left")

    # --- last-resort level fallback, also causal (expanding mean of prior months, all channels) ---
    gm = df.groupby("month_idx", as_index=False)["log_metric"].mean().sort_values("month_idx")
    gm["global_level"] = gm["log_metric"].shift(1).expanding().mean()
    df = df.merge(gm[["month_idx", "global_level"]], on="month_idx", how="left")

    has_own = df["own_n_hist"] >= min_own_history
    level = np.where(has_own, df["own_trailing_trend"], df["peer_trailing_trend"])
    level = pd.Series(level, index=df.index).fillna(df["peer_trailing_trend"]).fillna(df["global_level"])
    df["baseline"] = level + df["seasonal_index"]
    df["residual"] = df["log_metric"] - df["baseline"]

    # --- robust trailing sigma: partial-pooling blend of own trailing MAD and a pooled MAD.
    # A pure own-history MAD over <12 samples is itself noisy (large standard error at small n),
    # which lets sigma collapse toward the floor and inflate z on ordinary months. Shrinking
    # toward the larger-sample pooled MAD stabilizes it, with the weight growing as own history
    # accumulates. Both the pooled channel MAD and its global fallback are causal.
    df["own_sigma"] = (
        df.sort_values("month_idx").groupby(["brand", "channel"])["residual"]
        .transform(lambda s: s.shift(1).rolling(12, min_periods=4).apply(_robust_mad, raw=False))
    )
    df["_all"] = 1
    df["peer_sigma"] = _causal_pooled_mad(df, ["channel"], min_obs=8)
    df["global_sigma"] = _causal_pooled_mad(df, ["_all"], min_obs=8)
    df = df.drop(columns=["_all"])
    pooled_sigma = df["peer_sigma"].fillna(df["global_sigma"]).fillna(0.15)

    sigma_shrink_w = df["own_n_hist"] / (df["own_n_hist"] + 10.0)
    df["sigma"] = sigma_shrink_w * df["own_sigma"].fillna(pooled_sigma) + (1 - sigma_shrink_w) * pooled_sigma
    df["sigma"] = df["sigma"].clip(lower=0.08)  # floor: avoid divide-by-near-zero blowing up z on flat series

    df["sudden_z"] = df["residual"] / df["sigma"]

    def _gradual(s: pd.Series) -> pd.Series:
        roll = s.rolling(3, min_periods=3)
        mean3 = roll.mean()
        # sign agreement: fraction of the 3 months sharing the majority sign
        agree = roll.apply(lambda w: (np.sign(w) == np.sign(w.sum())).mean() if w.sum() != 0 else 0.0, raw=False)
        return mean3 * agree

    df["gradual_z"] = (
        df.sort_values("month_idx").groupby(["brand", "channel"])["sudden_z"].transform(_gradual)
    )

    # Direction gating (see module docstring): "up" scores only upward departures, so a metric
    # collapse cannot consume an alert slot that no cause in the taxonomy could explain.
    if direction == "up":
        signed = df[["sudden_z", "gradual_z"]]
    elif direction == "down":
        signed = -df[["sudden_z", "gradual_z"]]
    else:
        signed = df[["sudden_z", "gradual_z"]].abs()
    df["alert_score"] = signed.max(axis=1, skipna=True)
    # which of the two detectors produced the alert, so spike_type reflects the firing signal
    # rather than whichever score happens to be larger in absolute value
    df["alert_source"] = np.where(
        signed["sudden_z"].fillna(-np.inf) >= signed["gradual_z"].fillna(-np.inf), "sudden", "gradual"
    )
    df["direction"] = direction
    return df


def label_ground_truth(scored: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    gt = labels_df[["brand", "channel", "month"]].drop_duplicates().assign(is_true_spike=True)
    out = scored.merge(gt, on=["brand", "channel", "month"], how="left")
    out["is_true_spike"] = out["is_true_spike"] == True  # noqa: E712 (NaN -> False, avoids fillna dtype warning)
    return out


def select_threshold(scored_with_gt: pd.DataFrame, cost_ratio_fn_over_fp: float = 5.0) -> dict:
    """
    Sweep alert_score thresholds and pick the one minimizing expected cost
    = FP_count * 1 + FN_count * cost_ratio_fn_over_fp, i.e. the standard
    cost-weighted-confusion-matrix threshold rule (NOT rate-normalized:
    normalizing FP by the negative class size and FN by the much smaller
    positive class size would implicitly re-inflate the cost ratio by
    n_neg/n_pos and push the optimum to a degenerate corner).

    Rationale for cost_ratio=5: a missed spike (false negative) means the
    old manual, days-long process is the only backstop and trust in the
    system erodes silently; a false positive costs an analyst a few minutes
    to dismiss a pre-populated review card. We treat a missed real spike as
    roughly 5x as costly as an unnecessary alert. This is a business
    judgment call, not a statistical one, and should be revisited with the
    actual analyst team once real review-time data exists.
    """
    d = scored_with_gt.dropna(subset=["alert_score"])
    thresholds = np.linspace(0.2, 5.0, 97)
    rows = []
    n_pos = d["is_true_spike"].sum()
    n_neg = (~d["is_true_spike"]).sum()
    for t in thresholds:
        pred_pos = d["alert_score"] >= t
        tp = (pred_pos & d["is_true_spike"]).sum()
        fp = (pred_pos & ~d["is_true_spike"]).sum()
        fn = ((~pred_pos) & d["is_true_spike"]).sum()
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        cost = fp * 1.0 + fn * cost_ratio_fn_over_fp
        rows.append(dict(threshold=t, precision=precision, recall=recall,
                          fp=int(fp), fn=int(fn), cost=cost))
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["cost"].idxmin()]
    return dict(curve=curve, best_threshold=float(best["threshold"]),
                precision=float(best["precision"]), recall=float(best["recall"]),
                n_pos=int(n_pos), n_neg=int(n_neg))


def evaluate_at_threshold(scored_with_gt: pd.DataFrame, threshold: float) -> dict:
    """Evaluate one fixed operating threshold without tuning it on these rows."""
    d = scored_with_gt.dropna(subset=["alert_score"])
    pred_pos = d["alert_score"] >= threshold
    tp = int((pred_pos & d["is_true_spike"]).sum())
    fp = int((pred_pos & ~d["is_true_spike"]).sum())
    fn = int(((~pred_pos) & d["is_true_spike"]).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    return dict(
        threshold=float(threshold), precision=float(precision), recall=float(recall),
        tp=tp, fp=fp, fn=fn, n_rows=int(len(d)),
    )


def cross_validate_threshold_by_brand(scored_with_gt: pd.DataFrame, n_splits: int = 4,
                                      seed: int = 19) -> dict:
    """Tune on calibration brands and report predictions only on unseen brands."""
    brands = np.array(sorted(scored_with_gt["brand"].dropna().unique()), dtype=object)
    if not 2 <= n_splits <= len(brands):
        raise ValueError("n_splits must be between 2 and the number of brands")
    rng = np.random.default_rng(seed)
    rng.shuffle(brands)

    folds = []
    totals = dict(tp=0, fp=0, fn=0, n_rows=0)
    for fold_idx, holdout in enumerate(np.array_split(brands, n_splits), start=1):
        holdout_brands = sorted(holdout.tolist())
        calibration_brands = sorted(set(brands.tolist()) - set(holdout_brands))
        calibration = scored_with_gt[scored_with_gt["brand"].isin(calibration_brands)]
        test = scored_with_gt[scored_with_gt["brand"].isin(holdout_brands)]
        selected = select_threshold(calibration)
        metrics = evaluate_at_threshold(test, selected["best_threshold"])
        for key in totals:
            totals[key] += metrics[key]
        folds.append(
            dict(
                fold=fold_idx,
                calibration_brands=calibration_brands,
                holdout_brands=holdout_brands,
                calibration_threshold=selected["best_threshold"],
                calibration_precision=selected["precision"],
                calibration_recall=selected["recall"],
                holdout_precision=metrics["precision"],
                holdout_recall=metrics["recall"],
                tp=metrics["tp"], fp=metrics["fp"], fn=metrics["fn"], n_rows=metrics["n_rows"],
            )
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else np.nan
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else np.nan
    return dict(
        precision=float(precision), recall=float(recall), folds=folds,
        tp=totals["tp"], fp=totals["fp"], fn=totals["fn"], n_rows=totals["n_rows"],
    )


def detect(panel: pd.DataFrame, labels_df: pd.DataFrame, metric: str = "roas",
            direction: str = "up") -> dict:
    scored = build_scores(panel, metric=metric, direction=direction)
    scored_gt = label_ground_truth(scored, labels_df)
    cv_evaluation = cross_validate_threshold_by_brand(scored_gt)
    # The deployment threshold is calibrated on the full synthetic panel only after the
    # brand-held-out estimate above has been produced. It is not the reported evaluation score.
    sel = select_threshold(scored_gt)
    scored_gt["flagged"] = scored_gt["alert_score"] >= sel["best_threshold"]
    scored_gt["spike_type"] = np.where(scored_gt["flagged"], scored_gt["alert_source"], None)
    return dict(scored=scored_gt, threshold_selection=sel, cv_evaluation=cv_evaluation)


if __name__ == "__main__":
    from data_generator import generate_dataset

    data = generate_dataset()
    result = detect(data["panel"], data["labels"])
    sel = result["threshold_selection"]
    cv = result["cv_evaluation"]
    print(f"deployment threshold={sel['best_threshold']:.2f} "
          f"brand-held-out precision={cv['precision']:.2f} recall={cv['recall']:.2f}")
    print(result["scored"]["flagged"].sum(), "months flagged out of", len(result["scored"]))
