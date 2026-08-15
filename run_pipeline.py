#!/usr/bin/env python
"""
End-to-end pipeline: generate synthetic data -> detect anomalies -> attribute
causes -> write all outputs to ./output/.

Single-command usage:
    python run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import generate_dataset, CAUSE_TYPES, MONTHS
from anomaly_detection import detect
from attribution import (
    build_feature_table, build_training_set, train_and_evaluate,
    predict_distribution, narrate, FEATURE_COLS,
)

OUT = ROOT / "output"
PLOTS = OUT / "plots"


def plot_example_series(panel: pd.DataFrame, scored: pd.DataFrame, brand: str, channel: str, path: Path):
    s = scored[(scored.brand == brand) & (scored.channel == channel)].sort_values("month_idx")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(s["month_idx"], s["roas"], marker="o", ms=3, lw=1, label="ROAS")
    ax.plot(s["month_idx"], np.exp(s["baseline"]), lw=1, ls="--", color="gray", label="baseline (expected)")
    flagged = s[s["flagged"] == True]  # noqa: E712
    if len(flagged):
        ax.scatter(flagged["month_idx"], flagged["roas"], color="red", zorder=5, label="flagged anomaly")
    ax.set_title(f"{brand} / {channel}")
    ax.set_xlabel("month index")
    ax.set_ylabel("ROAS")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curve(curve: pd.DataFrame, best_threshold: float, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(curve["recall"], curve["precision"], lw=2)
    best_row = curve.iloc[(curve["threshold"] - best_threshold).abs().idxmin()]
    ax.scatter([best_row["recall"]], [best_row["precision"]], color="red", zorder=5,
               label=f"deployment threshold={best_threshold:.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Anomaly detector: full-panel threshold calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_attribution_metrics(metrics: pd.DataFrame, path: Path):
    m = metrics.dropna(subset=["pr_auc"]).sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(m["cause"].str.replace("_", " "), m["pr_auc"], color="#3b6fa0")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("PR-AUC (nested group-held-out, calibrated)")
    ax.set_title("Attribution model: per-cause discrimination")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    print("[1/4] Generating synthetic dataset...")
    data = generate_dataset()
    panel, events, labels = data["panel"], data["events"], data["labels"]
    panel.to_csv(OUT / "panel.csv", index=False)
    events.to_csv(OUT / "events_ground_truth.csv", index=False)
    labels.to_csv(OUT / "labels_ground_truth.csv", index=False)
    print(f"    panel: {panel.shape}, events: {events.shape}, labeled anomaly-months: {labels.shape}")

    print("[2/4] Running anomaly detector...")
    det = detect(panel, labels)
    scored, sel = det["scored"], det["threshold_selection"]
    detector_cv = det["cv_evaluation"]
    scored.to_csv(OUT / "detector_scored.csv", index=False)
    sel["curve"].to_csv(OUT / "detector_pr_curve.csv", index=False)
    cv_table = pd.DataFrame(detector_cv["folds"])
    for col in ["calibration_brands", "holdout_brands"]:
        cv_table[col] = cv_table[col].map(lambda values: ";".join(values))
    overall = pd.DataFrame([dict(
        fold="overall", calibration_brands="", holdout_brands="all brands (held out once)",
        calibration_threshold=np.nan, calibration_precision=np.nan, calibration_recall=np.nan,
        holdout_precision=detector_cv["precision"], holdout_recall=detector_cv["recall"],
        tp=detector_cv["tp"], fp=detector_cv["fp"], fn=detector_cv["fn"], n_rows=detector_cv["n_rows"],
    )])
    pd.concat([cv_table, overall], ignore_index=True).to_csv(
        OUT / "detector_brand_cv_metrics.csv", index=False
    )
    print(f"    deployment threshold={sel['best_threshold']:.2f}  "
          f"brand-held-out precision={detector_cv['precision']:.2f}  "
          f"recall={detector_cv['recall']:.2f}  "
          f"({scored['flagged'].sum()} of {len(scored)} months flagged at deployment threshold)")
    plot_pr_curve(sel["curve"], sel["best_threshold"], PLOTS / "detector_pr_curve.png")

    example_pairs = [
        ("Solstice Foods", "digital_video"),
        ("Nimbus Home", "social"),
        ("Crestline Beverages", "retail_media"),
    ]
    for brand, channel in example_pairs:
        if ((panel.brand == brand) & (panel.channel == channel)).any():
            safe = f"{brand}_{channel}".replace(" ", "_").replace(".", "")
            plot_example_series(panel, scored, brand, channel, PLOTS / f"series_{safe}.png")

    print("[3/4] Training attribution model...")
    feat = build_feature_table(panel, scored)
    feat.to_csv(OUT / "attribution_features.csv", index=False)
    rng = np.random.default_rng(3)
    train = build_training_set(feat, labels, panel, rng)
    result = train_and_evaluate(train)
    result["metrics"].to_csv(OUT / "attribution_metrics.csv", index=False)
    print(result["metrics"].to_string(index=False))
    print(f"    top-1 cause hit rate (OOF, vs. synthetic ground truth): {result['top1_hit_rate']:.2f}")
    plot_attribution_metrics(result["metrics"], PLOTS / "attribution_pr_auc.png")

    print("[4/4] Writing narrative explanations for flagged anomalies...")
    flagged = scored[scored["flagged"] == True].merge(  # noqa: E712
        feat, on=["brand", "channel", "month", "month_idx"], how="left", suffixes=("", "_f")
    )
    narratives = []
    sample = flagged.sample(n=min(15, len(flagged)), random_state=1).sort_values(["brand", "channel", "month"])
    for _, row in sample.iterrows():
        dist = predict_distribution(result["models"], row)
        narratives.append(narrate(row, dist))
    (OUT / "sample_narratives.txt").write_text("\n\n".join(narratives))

    print(f"\nDone. Outputs written to {OUT}/")


if __name__ == "__main__":
    main()
