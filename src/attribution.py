"""
Attribution model: given a flagged (brand, channel, month) anomaly, output calibrated
marginal likelihoods for all six cause types, not a single label or normalized softmax.

Framing (see design_proposal.md "Causal honesty"): this model is a
differential-diagnosis tool, not a causal-inference engine. It is trained to
recognize the *statistical signature* each cause type leaves on observable
metrics (a pattern-matching / associational task). It does not, and cannot
from observational data alone, prove that a given cause actually produced
the spike - that would require a designed experiment (geo holdout, PSA-style
incrementality test) or a validated marketing-mix model. iROAS itself, even
in this synthetic dataset, is only partially causal: it is generated from a
true incrementality factor but deliberately confounded a bit by external
demand shocks, mirroring the fact that real incrementality measurement is
rarely perfectly clean either.

Label strategy: ground truth cause labels do not exist in the real world.
Here we bootstrap with synthetic labels from the data-generating process
(a documented, inspectable set of domain priors about how each cause type
should move each metric). In production this synthetic-label model would
be the *cold start*, replaced over time by weak supervision from analyst
confirmations on real flagged anomalies (human-in-the-loop active learning
- see design_proposal.md "Productionisation").
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline

from anomaly_detection import build_scores
from data_generator import CAUSE_TYPES

PRESENCE_THRESHOLD = 0.20  # a cause's synthetic label weight must clear this to count as "present"
N_FOLDS = 5
CO_MOVEMENT_Z = 1.0  # bar for "this month moved" when measuring brand/market co-movement
PROB_EPS = 1e-6


def _new_estimator(seed: int):
    """One fitted preprocessing/model unit; missing-value policy lives inside it."""
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        LGBMClassifier(n_estimators=60, num_leaves=7, min_child_samples=5,
                       learning_rate=0.08, random_state=seed, verbosity=-1),
    )


def _logit_feature(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), PROB_EPS, 1 - PROB_EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _fit_platt(raw_probabilities: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=17)
    calibrator.fit(_logit_feature(raw_probabilities), y)
    return calibrator


def _apply_calibrator(calibrator: LogisticRegression | None,
                      raw_probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(raw_probabilities, dtype=float)
    return calibrator.predict_proba(_logit_feature(raw_probabilities))[:, 1]


class CalibratedCauseModel:
    """Average group-held-out fold models, each with its own nested Platt calibrator."""

    def __init__(self, fold_models: list[tuple[object, LogisticRegression | None]]):
        self.fold_models = fold_models

    def predict_proba(self, x) -> np.ndarray:
        calibrated = []
        for estimator, calibrator in self.fold_models:
            raw = estimator.predict_proba(x)[:, 1]
            calibrated.append(_apply_calibrator(calibrator, raw))
        positive = np.mean(np.vstack(calibrated), axis=0)
        return np.column_stack([1 - positive, positive])


def _z_col(panel: pd.DataFrame, metric: str, colname: str) -> pd.DataFrame:
    s = build_scores(panel, metric=metric)
    return s[["brand", "channel", "month", "sudden_z"]].rename(columns={"sudden_z": colname})


def build_feature_table(panel: pd.DataFrame, detector_scored: pd.DataFrame) -> pd.DataFrame:
    keys = ["brand", "channel", "month"]

    feat = detector_scored[keys + ["month_idx", "sudden_z", "gradual_z", "alert_score", "flagged"]].rename(
        columns={"sudden_z": "roas_sudden_z", "gradual_z": "roas_gradual_z", "alert_score": "roas_alert_score"}
    )
    for metric, col in [("iroas", "iroas_z"), ("rroi", "rroi_z"), ("spend", "spend_z"),
                        ("ctr", "ctr_z"), ("cpc", "cpc_z"), ("picr", "picr_z"),
                        ("impression_share", "impr_share_z")]:
        feat = feat.merge(_z_col(panel, metric, col), on=keys, how="left")

    feat["roas_iroas_divergence"] = feat["roas_sudden_z"] - feat["iroas_z"]
    feat["roas_rroi_divergence"] = feat["roas_sudden_z"] - feat["rroi_z"]
    feat["cpc_z_signed_good"] = -feat["cpc_z"]  # falling CPC (auction improving) is the "good" direction
    feat["shape_sudden"] = (feat["roas_sudden_z"].abs() >= feat["roas_gradual_z"].abs()).astype(int)

    # channel's share of brand total spend, and how anomalous that share is vs its own trailing level
    brand_month_spend = panel.groupby(["brand", "month"])["spend"].sum().rename("brand_spend").reset_index()
    p2 = panel.merge(brand_month_spend, on=["brand", "month"])
    p2["channel_share"] = (p2["spend"] / p2["brand_spend"]).clip(lower=1e-4)
    share_scored = build_scores(p2, metric="channel_share")[keys + ["sudden_z"]].rename(
        columns={"sudden_z": "mix_share_z"}
    )
    feat = feat.merge(share_scored, on=keys, how="left")

    # Brand-level total spend, scored on the same machinery: a spend cut REMOVES budget (channel
    # and brand total both fall) where a mix shift only REALLOCATES it (channel share moves,
    # brand total flat).
    #
    # What this actually buys, measured over 5 CV seeds with both classes correctly restricted to
    # flagged months: spend_reduction 2.81 -> 3.18 lift (beyond seed noise), external_demand
    # 3.79 -> 4.18 (within noise), and mix_shift 1.00 -> 0.87 - i.e. it helps the "budget removed"
    # signature and does NOT rescue mix shift, which stays at chance either way. An earlier
    # version of this comment claimed it fixed mix shift; that claim came from a leaked
    # evaluation and does not survive a correct one.
    # Two subtleties, both of which bit an earlier version:
    #   min_count=1 - the panel has reporting gaps, and a plain sum() treats a missing channel as
    #   zero spend. That fabricates exactly the "budget removed" signature this feature exists to
    #   detect (brand-months with a missing channel averaged -0.8z vs +0.1z for clean ones), and
    #   an all-missing month summed to 0.0 and became a log-clipped -13.8 residual. Propagate NaN.
    #   Per-brand channel key - build_scores falls back to a same-"channel" peer mean while a
    #   series has <3 own observations. With one shared key that peer is the mean across brands
    #   whose totals span 27x, so small or late-launching brands scored |z| of 5-7 for being
    #   small rather than for changing. Keying per brand confines the fallback to that brand.
    brand_total = panel.groupby(["brand", "month"], as_index=False)["spend"].sum(min_count=1)
    brand_total = brand_total.merge(
        panel[["brand", "month", "month_idx"]].drop_duplicates(), on=["brand", "month"]
    )
    brand_total["channel"] = "_TOTAL_" + brand_total["brand"]
    brand_z = build_scores(brand_total, metric="spend")[["brand", "month", "sudden_z"]].rename(
        columns={"sudden_z": "brand_spend_z"}
    )
    feat = feat.merge(brand_z, on=["brand", "month"], how="left")
    feat["spend_vs_brand"] = feat["spend_z"] - feat["brand_spend_z"]  # channel move net of brand move

    # Co-movement: is this brand-channel's move part of a broader pattern? Deliberately uses a
    # fixed 1z bar rather than the detector's tuned operating threshold - this measures "did
    # things move together", which should not shift every time the alert threshold is retuned
    # for a different FP/FN cost. Kept as a named constant so it is a choice, not a stray literal.
    alerty = detector_scored.assign(is_alert=detector_scored["alert_score"] >= CO_MOVEMENT_Z)
    brand_month_alert_rate = (
        alerty.groupby(["brand", "month"])
        .apply(lambda g: pd.Series({"n_alert": g["is_alert"].sum(), "n_total": len(g)}), include_groups=False)
        .reset_index()
    )
    market_month_alert_rate = (
        alerty.groupby("month")
        .apply(lambda g: pd.Series({"m_alert": g["is_alert"].sum(), "m_total": len(g)}), include_groups=False)
        .reset_index()
    )
    feat = feat.merge(alerty[keys + ["is_alert"]], on=keys, how="left")
    feat = feat.merge(brand_month_alert_rate, on=["brand", "month"], how="left")
    feat = feat.merge(market_month_alert_rate, on="month", how="left")
    # exclude self from both rates
    feat["brand_coincidence"] = ((feat["n_alert"] - feat["is_alert"]) / (feat["n_total"] - 1).clip(lower=1))
    other_brand_total = feat["m_total"] - feat["n_total"]
    other_brand_alert = feat["m_alert"] - feat["n_alert"]
    feat["market_coincidence"] = (other_brand_alert / other_brand_total.clip(lower=1))
    feat = feat.drop(columns=["is_alert", "n_alert", "n_total", "m_alert", "m_total"])

    return feat


FEATURE_COLS = [
    "roas_sudden_z", "roas_gradual_z", "iroas_z", "rroi_z", "spend_z",
    "ctr_z", "cpc_z_signed_good", "picr_z", "impr_share_z",
    "roas_iroas_divergence", "roas_rroi_divergence", "mix_share_z",
    "brand_spend_z", "spend_vs_brand",
    "shape_sudden", "brand_coincidence", "market_coincidence",
]


def build_training_set(feat: pd.DataFrame, labels_df: pd.DataFrame, panel: pd.DataFrame,
                        rng: np.random.Generator) -> pd.DataFrame:
    """
    Training population = the serving population = months the detector flagged. **Both** classes
    are restricted; restricting only one is a label leak, not a fix.

    An earlier version restricted negatives to flagged months but kept every labeled positive,
    reasoning that discarding ~37% of a tiny label set was too expensive. That was wrong. It left
    37.5% of positives unflagged against 0% of negatives, and since `roas_sudden_z` and
    `roas_gradual_z` are features and `flagged` is a threshold on them, a tree could read
    "below the detector threshold" as "guaranteed positive". The shortcut alone - predicting from
    unflagged-ness and nothing else - scored 2.33x lift on mix_shift, i.e. the whole of that
    version's headline 2.21x. Restricting both classes returns mix_shift to ~1.2x, which is the
    honest number: it remains at chance, and is reported that way.
    """
    keys = ["brand", "channel", "month"]
    flagged_keys = set(map(tuple, feat.loc[feat["flagged"] == True, keys].values))  # noqa: E712
    positives = feat.merge(labels_df, on=keys, how="inner")
    positives = positives[positives.apply(lambda r: (r["brand"], r["channel"], r["month"]) in flagged_keys, axis=1)]
    for c in CAUSE_TYPES:
        positives[c] = positives[c].fillna(0.0)

    labeled_keys = set(map(tuple, labels_df[keys].values))
    all_keys = feat[keys].drop_duplicates()
    as_tuples = all_keys.apply(tuple, axis=1)
    bg_pool = all_keys[~as_tuples.isin(labeled_keys) & as_tuples.isin(flagged_keys)]
    if bg_pool.empty:  # detector flagged nothing unlabeled; fall back to the wider panel
        bg_pool = all_keys[~as_tuples.isin(labeled_keys)]
    n_bg = min(len(bg_pool), max(len(positives) * 2, 120))
    bg_sample = bg_pool.sample(n=n_bg, random_state=int(rng.integers(0, 1_000_000)))
    negatives = feat.merge(bg_sample, on=keys, how="inner")
    for c in CAUSE_TYPES:
        negatives[c] = 0.0
    # background months belong to no event, so each is its own CV group (they are genuinely
    # independent samples, unlike the multi-month/multi-channel rows an event produces)
    negatives["group_id"] = ["bg_" + "_".join(map(str, k)) for k in negatives[keys].to_numpy()]

    train = pd.concat([positives, negatives], ignore_index=True)
    train = train.dropna(subset=FEATURE_COLS, how="all")
    if train["group_id"].isna().any():
        raise ValueError("every training row needs a group_id for leakage-free CV")
    return train


def train_and_evaluate(train: pd.DataFrame, seed: int = 11) -> dict:
    """
    Nested out-of-fold evaluation with StratifiedGroupKFold keyed on `group_id` (one injected
    event = one group), median imputation fitted inside every fold, and Platt calibration fitted
    on inner group-held-out predictions.

    This is not a cosmetic choice. A single event spans up to 3 months x several channels,
    so its rows are near-duplicates in feature space; plain StratifiedKFold splits on rows
    and puts the same shock in train and test. Grouped CV is the honest estimate of how the model
    behaves on an event it has never seen. Calibration is nested so the event being evaluated
    influences neither its base prediction nor the raw-score-to-likelihood mapping.
    """
    X = train[FEATURE_COLS]
    groups = train["group_id"].to_numpy()
    models: Dict[str, CalibratedCauseModel] = {}
    oof_probs = {c: np.zeros(len(train)) for c in CAUSE_TYPES}
    metrics_rows = []

    for cause in CAUSE_TYPES:
        y = (train[cause] >= PRESENCE_THRESHOLD).astype(int).to_numpy()
        n_pos_groups = len(np.unique(groups[y == 1]))
        if y.sum() < N_FOLDS or y.sum() == len(y) or n_pos_groups < N_FOLDS:
            # too few positives for this cause to cross-validate meaningfully; fit on all data,
            # report as "insufficient synthetic examples" rather than a fabricated score
            model = _new_estimator(seed)
            model.fit(X, y)
            models[cause] = CalibratedCauseModel([(model, None)])
            metrics_rows.append(dict(cause=cause, n_pos=int(y.sum()), n_pos_events=int(n_pos_groups),
                                      pr_auc=np.nan, base_rate=np.nan, lift_over_base=np.nan,
                                      brier=np.nan, brier_raw=np.nan,
                                      note="too few independent events for reliable grouped CV"))
            oof_probs[cause] = model.predict_proba(X)[:, 1]
            continue

        raw_oof = np.zeros(len(train))
        calibrated_oof = np.zeros(len(train))
        serving_folds = []
        outer_cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups)):
            X_train, y_train, groups_train = X.iloc[tr_idx], y[tr_idx], groups[tr_idx]

            # Nested group-held-out raw predictions fit the calibrator without allowing an
            # event's labels into either its base prediction or its calibration mapping.
            n_inner_pos_groups = len(np.unique(groups_train[y_train == 1]))
            n_inner_neg_groups = len(np.unique(groups_train[y_train == 0]))
            n_inner = min(4, n_inner_pos_groups, n_inner_neg_groups)
            inner_raw = np.zeros(len(tr_idx))
            if n_inner >= 2:
                inner_cv = StratifiedGroupKFold(
                    n_splits=n_inner, shuffle=True, random_state=seed + 100 + outer_fold
                )
                for inner_tr, inner_te in inner_cv.split(X_train, y_train, groups_train):
                    inner_model = _new_estimator(seed + 200 + outer_fold)
                    inner_model.fit(X_train.iloc[inner_tr], y_train[inner_tr])
                    inner_raw[inner_te] = inner_model.predict_proba(X_train.iloc[inner_te])[:, 1]
                calibrator = _fit_platt(inner_raw, y_train)
            else:
                calibrator = None

            outer_model = _new_estimator(seed + outer_fold)
            outer_model.fit(X_train, y_train)
            raw_test = outer_model.predict_proba(X.iloc[te_idx])[:, 1]
            raw_oof[te_idx] = raw_test
            calibrated_oof[te_idx] = _apply_calibrator(calibrator, raw_test)
            serving_folds.append((outer_model, calibrator))

        models[cause] = CalibratedCauseModel(serving_folds)
        oof_probs[cause] = calibrated_oof
        pr_auc = average_precision_score(y, calibrated_oof)
        brier = brier_score_loss(y, calibrated_oof)
        brier_raw = brier_score_loss(y, raw_oof)
        # PR-AUC is not comparable across causes with different prevalence: a no-skill model
        # scores the base rate. lift = pr_auc / base_rate is the honest cross-cause read, and
        # lift <= 1 means the model has learned nothing for that cause.
        base_rate = float(y.mean())
        lift = pr_auc / base_rate if base_rate > 0 else np.nan
        metrics_rows.append(dict(cause=cause, n_pos=int(y.sum()), n_pos_events=int(n_pos_groups),
                                  pr_auc=pr_auc, base_rate=base_rate, lift_over_base=lift,
                                  brier=brier, brier_raw=brier_raw,
                                  note="" if lift > 1.5 else "WEAK: at/near chance"))

    oof_df = pd.DataFrame(oof_probs)
    event_mask = train[CAUSE_TYPES].sum(axis=1) > 0  # only rows that actually had an injected cause
    if event_mask.sum() > 0:
        true_argmax = train.loc[event_mask, CAUSE_TYPES].to_numpy().argmax(axis=1)
        pred_argmax = oof_df.loc[event_mask, CAUSE_TYPES].to_numpy().argmax(axis=1)
        top1_hit_rate = float((true_argmax == pred_argmax).mean())
    else:
        top1_hit_rate = np.nan

    metrics = pd.DataFrame(metrics_rows)
    return dict(models=models, metrics=metrics, oof_probs=oof_df, top1_hit_rate=top1_hit_rate)


def predict_distribution(models: Dict[str, CalibratedCauseModel], feat_row: pd.Series) -> pd.DataFrame:
    # Preserve NaNs: each fitted pipeline owns the exact median imputation learned on its fold.
    x = feat_row[FEATURE_COLS].astype(float).to_frame().T
    raw = {c: float(models[c].predict_proba(x)[0, 1]) for c in CAUSE_TYPES}
    return pd.DataFrame({"cause": CAUSE_TYPES,
                          "probability": [raw[c] for c in CAUSE_TYPES]}).sort_values(
        "probability", ascending=False
    ).reset_index(drop=True)


NARRATIVE_TEMPLATES = {
    "genuine_efficiency_gain": "leading indicators improved broadly (CTR/PICR up, spend roughly flat) - consistent with a genuine efficiency gain",
    "spend_reduction_artifact": "spend fell while leading indicators were flat - the ROAS move looks like it tracks a budget cut more than a quality change",
    "mix_shift_artifact": "this channel's share of the brand's total budget moved sharply - the metric change may reflect reallocation, not improvement",
    "survivorship_bias": "impression share dropped alongside the metric spike - underperforming placements may have simply stopped reporting",
    "external_demand_spike": "other channels/brands moved similarly in the same month - a category-wide demand shift is a plausible shared driver",
    "creative_refresh": "CTR/PICR jumped sharply and suddenly (1-month pattern) - consistent with a creative refresh rather than a durable gain",
}


LOW_CONFIDENCE_NOTE = (
    "No cause clears a confident threshold. Treat this as 'flagged, unexplained' - the driver "
    "may sit outside the six-cause taxonomy (competitor action, macro shift, a reporting change). "
    "Worth an analyst's eyes before it goes in the monthly narrative."
)


def narrate(row: pd.Series, dist: pd.DataFrame, top_k: int = 2, confident_at: float = 0.25) -> str:
    """
    Analyst-facing explanation. Two deliberate constraints:

    1. Direction. Every template describes an *upward* driver, so narrating a downward move
       with them ("spend fell ... consistent with a budget cut" attached to a ROAS collapse)
       is incoherent. The detector is one-sided by default, but this guards the case where a
       caller scores with direction="both" and pipes the result through here anyway.
    2. Honest uncertainty. If no cause clears `confident_at`, the output says so rather than
       ranking six weak guesses as if the top one meant something.

    The displayed shape comes from the detector's `alert_source` - the signal that actually
    fired - not from the `shape_sudden` model feature, which compares |sudden_z| to |gradual_z|
    and can disagree with the firing signal. Telling an analyst
    "sudden, single-month" when the alert came from the 3-month ramp sends them to the wrong
    part of the chart.
    """
    source = row.get("alert_source")
    if source not in ("sudden", "gradual"):  # fall back for callers without detector output
        source = "sudden" if row.get("shape_sudden", 1) == 1 else "gradual"
    score_col = "roas_sudden_z" if source == "sudden" else "roas_gradual_z"
    z = float(row.get(score_col, np.nan))
    shape = "sudden, single-month" if source == "sudden" else "gradual, multi-month"
    header = (f"{row['brand']} / {row['channel']} / {row['month']}: alert score {z:+.1f} "
              f"robust std. devs vs. expected ({shape} pattern).")

    if np.isfinite(z) and z < 0:
        return (header + "\n  Downward move - the cause taxonomy in this model covers drivers of "
                         "metric *increases* only, so no attribution is offered. Route to the "
                         "standard performance-decline review instead.")

    ranked = dist.sort_values("probability", ascending=False)
    top = ranked[ranked["probability"] >= confident_at].head(top_k)
    if len(top) == 0:
        best = "" if len(ranked) == 0 else (
            f"\n  Closest match was {ranked.iloc[0]['cause'].replace('_', ' ')} at "
            f"{ranked.iloc[0]['probability']:.0%}, below the {confident_at:.0%} confidence bar."
        )
        return header + "\n  " + LOW_CONFIDENCE_NOTE + best

    parts = [
        f"{r['cause'].replace('_', ' ')} ({r['probability']:.0%} calibrated likelihood): "
        f"{NARRATIVE_TEMPLATES[r['cause']]}"
        for _, r in top.iterrows()
    ]
    return header + "\n  Most likely drivers -\n  - " + "\n  - ".join(parts)


if __name__ == "__main__":
    from data_generator import generate_dataset
    from anomaly_detection import detect

    data = generate_dataset()
    det = detect(data["panel"], data["labels"])
    feat = build_feature_table(data["panel"], det["scored"])
    rng = np.random.default_rng(3)
    train = build_training_set(feat, data["labels"], data["panel"], rng)
    result = train_and_evaluate(train)
    print(result["metrics"].to_string(index=False))
    print("top1_hit_rate:", result["top1_hit_rate"])

    flagged = det["scored"][det["scored"]["flagged"]].merge(
        feat, on=["brand", "channel", "month", "month_idx"], how="left", suffixes=("", "_f")
    )
    if len(flagged):
        example = flagged.iloc[0]
        dist = predict_distribution(result["models"], example)
        print()
        print(narrate(example, dist))
