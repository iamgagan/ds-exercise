# Design Proposal: Spike Investigation - Anomaly Detection & Attribution

**Scope:** deep on two of the three components - **anomaly detection** and **attribution**.
Forecasting is de-scoped: it isn't in the brief's evaluation criteria, and a shallow third pass
would dilute the two that matter. All numbers come from `output/` after `python run_pipeline.py`.

## 1. Problem framing

Two distinct ML problems, chained, not merged into one model:

1. **Anomaly detection** - unsupervised time-series outlier detection. No reliable label exists
   at deployment time, only a business definition of "meaningful," so the output must be a
   calibrated departure-from-expectation score, not a classifier.
2. **Attribution** - weakly supervised multi-label classification with unverifiable ground truth.
   "Why" is never observed, only inferred, and causes co-occur, so a single-label classifier
   would misrepresent the system's confidence.

Keeping them decoupled matters: different failure modes, different consumers, and merging them
hides which half is wrong when the system misfires.

## 2. Data strategy

`src/data_generator.py`: 8 brands x 6-8 channels (9-channel universe) x 36 months, built as a
small structural model rather than noise with labels bolted on:

- **Revenue follows a diminishing-returns response curve** (`spend x elasticity(spend) x quality x demand x noise`), so a spend cut mechanically raises ROAS and a mix shift toward
  efficient channels mechanically raises blended ROAS - no special-cased "if artifact then boost
  ROAS" rule. The artifacts fall out of the economics that produces them in real media data.
- **iROAS comes from a latent quality factor independent of spend level**, so it doesn't move for
  spend/mix artifacts. That asymmetry is the single strongest attribution feature. It isn't perfectly
  clean either - deliberately confounded by demand shocks, mirroring real incrementality work.
- **RROI** is an EWMA shrinkage of the attributable-sales ratio toward its trailing 3-month
  average - "stickier" than ROAS, so a single-month move RROI doesn't confirm is a tell.
- **Leading indicators** (CTR, CPC, PICR, impression share) track the same quality factor as
  iROAS, not spend/mix - what tells genuine gains apart from budget artifacts.
- **Six cause types** injected as latent-factor shocks, sudden (1mo) or gradual (3mo ramp), often
  co-occurring with a Dirichlet-split weight. External-demand events are sampled once per category
  and applied to every brand in that category, so the intended "lifted all boats" signature is
  genuinely cross-brand. Local events remain brand/channel-specific.
- **Realism knobs:** asymmetric per-brand seasonality, ~2-3% missing periods, one brand launching
  mid-panel (cold-start case), CPC undefined for TV/OOH (bought on reach). Ground truth is stored
  **only** for evaluation, never as a feature.

**Limitations:** one draw from one generative model, encoding our assumptions about response
curves, seasonality, and confounding. Survivorship bias is the least faithful mechanism (a
reported-metric multiplier, not true placement dropout). Label density (~22%) is higher than a
real book of business, to get enough events per cause to evaluate at all.

## 3. Model design

**Anomaly detection.** Not mean +/- 2 sigma. Baseline = own trailing median (6mo) + a pooled
channel seasonal index, falling back to the same-channel peer trend for brand-channels with
fewer than 3 own observations. `sudden_z` (1-month) and `gradual_z` (3-month trailing mean, credited
only when the months agree in sign - a lightweight CUSUM) run in parallel off the same residual.
Threshold selection minimizes `FP_count + 5 x FN_count` (raw counts - normalizing by class size
pushes the optimum to a degenerate corner); 5:1 reflects that a missed spike leaves the manual
process as the only backstop, while a false positive costs minutes. The deployment threshold is
**0.50**; evaluation is brand-held-out rather than measured on the calibration rows (Section 5).

The detector is **one-sided** (`direction="up"`). Every cause in the taxonomy raises the headline
metric, so scoring on `|z|` spent half the review budget on *collapses* no cause could explain,
and produced incoherent narratives ("spend fell ... consistent with a budget cut" attached to a
ROAS drop). Collapses route to the separate performance-decline review. Analyst narratives report
the detector that actually fired: `sudden_z` for sudden alerts and `gradual_z` for gradual alerts.

**Rejected:** rolling mean/std (not robust to the anomaly pulling its own window); per-series STL
(most series too short - as low as 10 months post-launch); a global IsolationForest (loses channel
seasonality, nothing an analyst can check against the chart).

**Attribution.** Independent LightGBM binary classifiers, one per cause (not one softmax) - causes
co-occur, so outputs are calibrated marginal likelihoods and deliberately do **not** sum to one.
Each group-held-out fold owns its median imputer and small regularized model (60 trees, 7 leaves);
an inner event-grouped loop fits Platt calibration without seeing the evaluated event. The serving
model averages the calibrated fold ensemble. **Rejected:** normalizing marginals into an
"explained-signal share" (not probabilistically meaningful); multi-class softmax; a neural model.

## 4. Feature engineering

All features are z-scores/ratios against the same baseline machinery used for detection, re-used
for `iroas`, `rroi`, `spend`, `ctr`, `cpc`, `picr`, `impression_share`, and channel spend share:

| Feature | Separates |
|---|---|
| `roas_iroas_divergence`, `roas_rroi_divergence` | genuine gain (both move) vs. artifact (ROAS only); transient vs. sustained |
| `spend_z`, `mix_share_z`, `brand_spend_z`, `spend_vs_brand` | budget *reallocated* (brand total flat) vs. *removed* (total down) |
| `ctr_z`, `picr_z`, `cpc_z` (sign-flipped) | genuine efficiency gain / creative refresh |
| `impr_share_z` | survivorship bias (drops with the metric spike) |
| `brand_coincidence`, `market_coincidence` | external demand vs. brand-specific |
| `shape_sudden` | creative refresh/survivorship (sudden) vs. gain/mix shift (gradual) |

**Lag structure:** features are contemporaneous with the flagged month - this diagnoses an
already-detected spike, not a future one. The structural lag matters more: **every** quantity
scoring month *t* uses only data strictly before *t*, the pooled seasonal index and peer sigma
included. That is enforced, not asserted - `src/test_causality.py` re-scores a truncated panel
and requires bit-identical results. An earlier version failed it: pooling those two over the full
panel shifted 66% of month-*t* scores (max 7.3 z, against a ~1 z threshold); fixing it lowered
the reported numbers and made them deployable.

## 5. Evaluation

**Detector.** Four-fold brand-grouped evaluation tunes the threshold on six brands and scores it on
two unseen brands, rotating until every brand is held out once. Aggregated held-out performance is
**precision 0.46 / recall 0.76**. Only after that estimate is produced is the deployment threshold
fit on all eight brands; `output/detector_brand_cv_metrics.csv` contains every fold.

**Attribution, without clean labels in the real world.** Nested cross-validation is grouped by
*event*, not row. One event spans up to 3 months x several channels, so row-wise splitting leaks
near-duplicates. Median imputation is fitted inside each fold, and calibration uses inner
group-held-out predictions. **Effective sample size is events, not rows.**

Out-of-fold results (`output/attribution_metrics.csv`); lift is included because PR-AUC is not
comparable across causes of differing prevalence.

| cause | events | PR-AUC | base | lift |
|:---|---:|---:|---:|---:|
| Survivorship bias | 11 | 0.90 | 0.030 | 29.6x |
| Spend reduction artifact | 8 | 0.21 | 0.024 | 8.7x |
| Creative refresh | 21 | 0.65 | 0.079 | 8.2x |
| Genuine efficiency gain | 15 | 0.58 | 0.084 | 6.9x |
| Mix shift artifact | 6 | 0.30 | 0.056 | 5.3x |
| External demand spike | 6 | 0.88 | 0.258 | 3.4x |

Top-1 hit rate is **0.68**. PR-AUC is reported beside base rate because raw PR-AUC is not comparable
across causes. Nested Platt scaling improves held-out Brier score for five causes; external demand
worsens from 0.057 to 0.064 on only six events, a visible calibration-risk warning rather than a
claim of certainty.

The training population is the serving population - flagged months, with **both** classes
restricted. That symmetry is load-bearing. An intermediate version restricted only the negatives,
keeping every labeled positive because a tiny label set could ill afford losing 37% of it. It
looked excellent (mix-shift 2.21x, nothing at chance) and was an artifact: 37.5% of positives were
unflagged vs 0% of negatives, and since `roas_sudden_z` is a feature and `flagged` is a threshold
on it, "below the threshold" *is* "positive". Unflagged-ness alone scored 2.33x - the entire
headline. When a model runs downstream of a filter, both classes must come from the filtered
population, and a jump on your weakest class is evidence of a leak before it is evidence of a fix.

`brand_spend_z` / `spend_vs_brand` separate budget *removed* (brand total falls) from *reallocated*
(flat). The shared category-demand generator now makes cross-brand coincidence a real signal rather
than a narrative unsupported by the data-generating process.

The synthetic-label model is only a cold start; the production evaluation loop is
analyst-agreement rate over time (see *Productionisation*).

Predicted likelihoods are independent, not normalized contributions. Narratives include only
causes clearing the 25% confidence bar; a weak second-ranked cause is omitted rather than promoted.

## 6. Causal honesty

**Causal by construction:** the *true* incrementality factor behind iROAS - never fully observed
even in the synthetic data; the system sees a measured, demand-confounded proxy.
**Associational, and presented as such:** ROAS, RROI, and every attribution output. The model is
a differential-diagnosis tool, not a causal-inference engine - "consistent with a creative
refresh," never "the refresh caused this," a distinction that lives in the narrative templates
(`src/attribution.py`), not just here. **For a non-technical audience:** every output pairs a
calibrated marginal likelihood with its evidence - "creative refresh (28%): CTR/PICR jumped" - and
says "flagged, unexplained" when nothing is confident. A real causal claim needs a designed
experiment (geo holdout, PSA test) or a validated MMM - neither runs here.

## 7. Productionisation

**Retraining:** monthly, on a rolling 24-36 month window. **Monitoring:** analyst agreement rate
on attribution (the real accuracy proxy); feature drift on leading indicators (PSI/KS, monthly);
alert-volume drift; raw-vs-calibrated Brier and lift-over-base per cause, so poor calibration or a
cause degrading toward chance surfaces instead of looking authoritative. **Cold start:** exercised
here (one brand launches mid-panel) - new brand-channels borrow
the peer trend and sigma until a 3-month own-history floor. Next step, not built: cluster brands
into spend/mix archetypes so new brands borrow from the nearest archetype, not the full pool.
**Human-in-the-loop:** every analyst confirm/correct/reject is logged and feeds the next retrain -
how the synthetic-label cold start gets replaced by real signal.

## 8. Limits - where this breaks

The simulator's assumptions are the ceiling: if real response curves, seasonality, or confounding
differ materially, both models need re-validation first. Event counts remain tiny (6-21 per cause),
and external-demand calibration is unstable despite good discrimination. Short-history brands
underperform on peer priors; the causal seasonal index is blind for a panel's first 12 months.
Attribution has no "none of the above" - a driver outside the taxonomy (competitor stockout,
macro shock) yields uniformly low probabilities, surfaced as "flagged, unexplained." Forecasting
isn't built. Best next inputs: analyst-confirmed labels and placement-level spend/revenue.
