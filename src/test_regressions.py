"""Regression tests for analyst-facing and evaluation correctness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anomaly_detection import (
    build_scores,
    cross_validate_threshold_by_brand,
    label_ground_truth,
)
from attribution import FEATURE_COLS, narrate, predict_distribution
from data_generator import CAUSE_TYPES, generate_dataset


class _RecordingModel:
    def __init__(self, probability: float):
        self.probability = probability
        self.saw_missing = False

    def predict_proba(self, x):
        values = np.asarray(x, dtype=float)
        self.saw_missing = bool(np.isnan(values).any())
        return np.array([[1.0 - self.probability, self.probability]])


class NarrativeTests(unittest.TestCase):
    def test_gradual_alert_uses_gradual_score_for_direction(self):
        row = pd.Series(
            {
                "brand": "Example Brand",
                "channel": "social",
                "month": "2025-03",
                "roas_sudden_z": -0.4,
                "roas_gradual_z": 1.3,
                "alert_source": "gradual",
            }
        )
        dist = pd.DataFrame(
            {
                "cause": ["genuine_efficiency_gain"],
                "probability": [0.8],
            }
        )

        text = narrate(row, dist)

        self.assertIn("+1.3", text)
        self.assertIn("gradual, multi-month", text)
        self.assertNotIn("Downward move", text)

    def test_narrative_omits_causes_below_confidence_threshold(self):
        row = pd.Series(
            {
                "brand": "Example Brand",
                "channel": "social",
                "month": "2025-03",
                "roas_sudden_z": 1.4,
                "roas_gradual_z": 0.7,
                "alert_source": "sudden",
            }
        )
        dist = pd.DataFrame(
            {
                "cause": ["genuine_efficiency_gain", "mix_shift_artifact"],
                "probability": [0.8, 0.04],
            }
        )

        text = narrate(row, dist, confident_at=0.25)

        self.assertIn("genuine efficiency gain", text)
        self.assertNotIn("mix shift artifact", text)
        self.assertNotIn("explained signal", text)


class ProbabilityTests(unittest.TestCase):
    def test_prediction_preserves_missing_values_for_fitted_preprocessor(self):
        models = {cause: _RecordingModel(0.4) for cause in CAUSE_TYPES}
        row = pd.Series({col: 1.0 for col in FEATURE_COLS})
        row["cpc_z_signed_good"] = np.nan

        dist = predict_distribution(models, row)

        self.assertTrue(all(model.saw_missing for model in models.values()))
        self.assertNotIn("relative_share", dist.columns)
        self.assertAlmostEqual(float(dist["probability"].sum()), 2.4)


class GeneratorTests(unittest.TestCase):
    def test_external_demand_events_span_multiple_brands(self):
        data = generate_dataset()
        events = data["events"]
        labels = data["labels"]
        category_events = events[events["scope"] == "category_wide"]

        self.assertGreaterEqual(len(category_events), 5)
        for event_key in category_events["event_key"]:
            touched = labels[labels["event_keys"].str.split(";").apply(lambda xs: event_key in xs)]
            self.assertGreaterEqual(touched["brand"].nunique(), 2)


class DetectorEvaluationTests(unittest.TestCase):
    def test_brand_grouped_threshold_cv_holds_out_every_brand_once(self):
        data = generate_dataset()
        scored = label_ground_truth(build_scores(data["panel"]), data["labels"])

        result = cross_validate_threshold_by_brand(scored, n_splits=4)

        held_out = []
        for fold in result["folds"]:
            train = set(fold["calibration_brands"])
            test = set(fold["holdout_brands"])
            self.assertTrue(train.isdisjoint(test))
            held_out.extend(test)
        self.assertCountEqual(held_out, sorted(scored["brand"].unique()))
        self.assertTrue(np.isfinite(result["precision"]))
        self.assertTrue(np.isfinite(result["recall"]))


if __name__ == "__main__":
    unittest.main()
