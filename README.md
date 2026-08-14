# Spike Investigation: Anomaly Detection + Attribution

Take-home exercise submission. See `design_proposal.md` for the primary deliverable.

## Run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

Runs end to end with no arguments: generates the synthetic dataset, runs the anomaly detector,
trains and evaluates the attribution model, and writes everything to `output/` (scored panel,
ground truth, metrics, PR curves, example series plots, sample analyst narratives).

For an annotated walkthrough with inline plots, open `notebook.ipynb` (pre-executed; re-run cells
to regenerate).

To verify the detector is causal (no month-*t* score depends on later data):

```bash
python src/test_causality.py
```

## Layout

- `src/data_generator.py` - synthetic CPG media dataset (8 brands, 9-channel universe, 36 months)
- `src/anomaly_detection.py` - baseline model + robust z-score detector + threshold selection
- `src/attribution.py` - feature engineering + per-cause probabilistic attribution model
- `src/test_causality.py` - regression test: no month-*t* score may depend on data after *t*
- `run_pipeline.py` - single-command end-to-end run
- `notebook.ipynb` - exploratory walkthrough with visuals
- `design_proposal.md` - the design proposal (primary deliverable)

## Time spent

Roughly 8 hours.
