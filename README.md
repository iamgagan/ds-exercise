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
ground truth, brand-held-out detector metrics, calibrated attribution metrics, PR curves, example
series plots, and sample analyst narratives).

For an annotated walkthrough with inline plots, open `notebook.ipynb` (pre-executed; re-run cells
to regenerate).

To run the regression suite and verify the detector is causal (no month-*t* score depends on later
data):

```bash
python -m unittest src/test_regressions.py
python src/test_causality.py
```

## Layout

- `src/data_generator.py` - synthetic CPG media dataset with shared category-demand events
- `src/anomaly_detection.py` - robust detector + brand-held-out threshold evaluation
- `src/attribution.py` - feature engineering + nested calibrated per-cause likelihood models
- `src/test_causality.py` - regression test: no month-*t* score may depend on data after *t*
- `src/test_regressions.py` - category-shock, narrative, probability, and evaluation regressions
- `run_pipeline.py` - single-command end-to-end run
- `notebook.ipynb` - exploratory walkthrough with visuals
- `design_proposal.md` - the design proposal (primary deliverable)

## Time spent

Roughly 8 hours.
