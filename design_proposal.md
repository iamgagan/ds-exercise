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
  co-occurring with a Dirichlet-split weight, calibrated to be investigation-worthy but not
  trivially separable. Injection rates balance *independent events per cause*, not rows - event
  count is what the evaluation can resolve.
- **Realism knobs:** asymmetric per-brand seasonality, ~2-3% missing periods, one brand launching
  mid-panel (cold-start case), CPC undefined for TV/OOH (bought on reach). Ground truth is stored
  **only** for evaluation, never as a feature.

**Limitations:** one draw from one generative model, encoding our assumptions about response
curves, seasonality, and confounding. Survivorship bias is the least faithful mechanism (a
reported-metric multiplier, not true placement dropout). Label density (~19%) is higher than a
real book of business, to get enough events per cause to evaluate at all.

## 3. Model design

**Anomaly detection.** Not mean +/- 2 sigma. Baseline = own trailing median (6mo) + a pooled
channel seasonal index, falling back to the same-channel peer trend for brand-channels with
fewer than 3 own observations. `sudden_z` (1-month) and `gradual_z` (3-month trailing mean, credited
only when the months agree in sign - a lightweight CUSUM) run in parallel off the same residual.
Threshold selection minimizes `FP_count + 5 x FN_count` (raw counts - normalizing by class size
pushes the optimum to a degenerate corner); 5:1 reflects that a missed spike leaves the manual
process as the only backstop, while a false positive costs minutes. Current calibration: **precision 0.36 / recall 0.66** - the full curve
(`output/detector_pr_curve.png`) is the deliverable, not this one point.

The detector is **one-sided** (`direction="up"`). Every cause in the taxonomy raises the headline
metric, so scoring on `|z|` spent half the review budget on *collapses* no cause could explain,
and produced incoherent narratives ("spend fell ... consistent with a budget cut" attached to a
ROAS drop). Gating on signed departure dominates two-sided on both axes: precision 0.36 vs 0.31,
recall 0.66 vs 0.51. Collapses route to the separate performance-decline review.

**Rejected:** rolling mean/std (not robust to the anomaly pulling its own window); per-series STL
(most series too short - as low as 10 months post-launch); a global IsolationForest (loses channel
seasonality, nothing an analyst can check against the chart).

**Attribution.** Independent LightGBM binary classifiers, one per cause (not one softmax) -
causes co-occur, and forcing probabilities to sum to 1 would misrepresent "70% likely this AND
55% likely that" as "60/40." Models are small and heavily regularized (60 trees, 7 leaves) - with
7-14 events per cause, anything larger fits noise. **Rejected:** multi-class softmax (contradicts
"distribution, not a label"); rule-based attribution (brittle, no calibrated confidence); a neural
multi-label model (unjustifiable at this label volume).

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

**Attribution, without clean labels in the real world.** Cross-validation is grouped by *event*,
not by row, using scikit-learn's `StratifiedGroupKFold`. One event spans up to 3 months x several
channels, so its rows are near-duplicates and row-wise splitting puts the same shock in train and
test - a leak worth up to 0.47 PR-AUC (external demand: 0.57 row-wise vs 0.10 event-wise). So
**effective sample size is events, not rows**: 60 rows from 8 events is an 8-sample problem.

Out-of-fold results (`output/attribution_metrics.csv`); lift is included because PR-AUC is not
comparable across causes of differing prevalence.

| cause | events | PR-AUC | base | lift |
|:---|---:|---:|---:|---:|
| Survivorship bias | 12 | 0.92 | 0.052 | 17.8x |
| Creative refresh | 13 | 0.59 | 0.050 | 11.8x |
| Genuine efficiency gain | 14 | 0.63 | 0.076 | 8.3x |
| External demand spike | 7 | 0.36 | 0.098 | 3.6x |
| Spend reduction artifact | 12 | 0.30 | 0.103 | 2.9x |
| **Mix shift artifact** | **8** | **0.10** | **0.111** | **0.89x - at chance** |

Top-1 hit rate is 0.61. Four causes carry strong signal, spend-reduction is modest, and
**mix-shift sits at chance** - reported, not smoothed over.

The training population is the serving population - flagged months, with **both** classes
restricted. That symmetry is load-bearing. An intermediate version restricted only the negatives,
keeping every labeled positive because a tiny label set could ill afford losing 37% of it. It
looked excellent (mix-shift 2.21x, nothing at chance) and was an artifact: 37.5% of positives were
unflagged vs 0% of negatives, and since `roas_sudden_z` is a feature and `flagged` is a threshold
on it, "below the threshold" *is* "positive". Unflagged-ness alone scored 2.33x - the entire
headline. When a model runs downstream of a filter, both classes must come from the filtered
population, and a jump on your weakest class is evidence of a leak before it is evidence of a fix.

`brand_spend_z` / `spend_vs_brand` separate budget *removed* (brand total falls) from
*reallocated* (flat). Over 5 CV seeds they are worth +0.37 lift on spend-reduction, beyond seed
noise, and do **not** rescue mix-shift (1.00 -> 0.87) - they earn their place on the cause they
were designed for, not the one first claimed for them.

The synthetic-label model is only a cold start; the production evaluation loop is
analyst-agreement rate over time (see *Productionisation*).

Predicted probabilities are mostly low - at precision 0.36 most flagged months have no injected
cause, and the model correctly stays unsure rather than confabulating.

## 6. Causal honesty

**Causal by construction:** the *true* incrementality factor behind iROAS - never fully observed
even in the synthetic data; the system sees a measured, demand-confounded proxy.
**Associational, and presented as such:** ROAS, RROI, and every attribution output. The model is
a differential-diagnosis tool, not a causal-inference engine - "consistent with a creative
refresh," never "the refresh caused this," a distinction that lives in the narrative templates
(`src/attribution.py`), not just here. **For a non-technical audience:** every output pairs a
probability with its evidence - "creative refresh (28% likely): CTR/PICR jumped sharply" - and
says "flagged, unexplained" when nothing is confident. A real causal claim needs a designed
experiment (geo holdout, PSA test) or a validated MMM - neither runs here.

## 7. Productionisation

**Retraining:** monthly, on a rolling 24-36 month window. **Monitoring:** analyst agreement rate
on attribution (the real accuracy proxy); feature drift on leading indicators (PSI/KS, monthly);
alert-volume drift; Brier and lift-over-base per cause, so a cause at chance surfaces instead of
looking authoritative - this is what keeps mix-shift's 0.89x visible rather than buried in an
average. **Cold start:** exercised here (one brand launches mid-panel) - new brand-channels borrow
the peer trend and sigma until a 3-month own-history floor. Next step, not built: cluster brands
into spend/mix archetypes so new brands borrow from the nearest archetype, not the full pool.
**Human-in-the-loop:** every analyst confirm/correct/reject is logged and feeds the next retrain -
how the synthetic-label cold start gets replaced by real signal.

## 8. Limits - where this breaks

The simulator's assumptions are the ceiling: if real response curves, seasonality, or confounding
differ materially, both models need re-validation first. **Mix-shift attribution does not work**
(0.89x lift, at chance, on 8 events) and spend-reduction is modest at 2.9x - the system separates
budget mechanics from genuine gains but cannot identify reallocation. Short-history brands
underperform on peer priors; the causal seasonal index is blind for a panel's first 12 months.
Attribution has no "none of the above" - a driver outside the taxonomy (competitor stockout,
macro shock) yields uniformly low probabilities, surfaced as "flagged, unexplained." Forecasting
isn't built. Best next inputs: analyst-confirmed labels and placement-level spend/revenue.
