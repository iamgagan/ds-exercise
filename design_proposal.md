# Design Proposal: Spike Investigation - Anomaly Detection & Attribution

**Scope:** deep on two of the three components - **anomaly detection** and **attribution**.
Forecasting is de-scoped (see *Limits*): it isn't in the brief's evaluation criteria, and a
shallow third pass would have diluted the two that matter. All numbers come from `output/` after
`python run_pipeline.py` (seed fixed).

## 1. Problem framing

Two distinct ML problems, chained, not merged into one model:

1. **Anomaly detection** - unsupervised time-series outlier detection. No reliable label exists
   at deployment time, only a business definition of "meaningful," so the output must be a
   calibrated departure-from-expectation score, not a classifier.
2. **Attribution** - weakly supervised multi-label classification with unverifiable ground truth.
   "Why" is never observed, only inferred, and several causes can be simultaneously true, so a
   single-label classifier would misrepresent the system's confidence.

Keeping them decoupled matters: different failure modes, different consumers, and merging them
makes it impossible to tell which half is wrong when the system misfires.

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
  average - "stickier" than ROAS, so a single-month ROAS move RROI doesn't confirm is a tell.
- **Leading indicators** (CTR, CPC, PICR, impression share) move with the same quality factor as
  iROAS, not with spend/mix - what lets genuine gains be told apart from budget artifacts.
- **Six cause types** injected as latent-factor shocks, sudden (1mo) or gradual (3mo ramp), often
  co-occurring with a Dirichlet-split weight, calibrated to be investigation-worthy without being
  trivially separable from noise. Injection rates are balanced on *independent events per cause*,
  not rows: brand-wide causes emit many correlated rows from one shock, and event count is what
  the evaluation can resolve.
- **Realism knobs:** asymmetric per-brand seasonality, ~2-3% missing periods, one brand launching
  mid-panel (cold-start case), CPC undefined for TV/OOH (bought on reach). Ground truth is stored
  **only** for evaluation, never as a feature.

**Limitations:** one draw from one generative model, encoding our assumptions about response
curves, seasonality, and confounding. Survivorship bias is the least faithful mechanism (a
reported-metric multiplier, not true placement dropout - the panel has no placement granularity).
Label density (~19%) is higher than a real book of business, to get enough events per cause to
evaluate at all.

## 3. Model design

**Anomaly detection.** Not mean +/- 2 sigma. Baseline = own trailing median (6mo) + a pooled
channel seasonal index, falling back to the same-channel peer trend for brand-channels with
fewer than 3 own observations. `sudden_z` (1-month) and `gradual_z` (3-month trailing mean,
credited only when the months agree in sign - a lightweight CUSUM) run in parallel off the same
robust residual. Threshold selection minimizes `FP_count + 5 x FN_count` (raw counts -
normalizing by class size re-inflates the ratio and pushes the optimum to a degenerate corner);
5:1 reflects that a missed spike leaves the manual process as the only backstop, while a false
positive costs minutes. Current calibration: **precision 0.36 / recall 0.66** - the full curve
(`output/detector_pr_curve.png`) is the deliverable, not this one point.

The detector is **one-sided** (`direction="up"`). Every cause in the taxonomy raises the headline
metric, so scoring on `|z|` spent half the review budget on metric *collapses* that no cause
could ever explain - and produced incoherent narratives ("spend fell ... consistent with a budget
cut" attached to a ROAS drop). Gating on signed departure dominates the two-sided version on both
axes: precision 0.36 vs 0.31, recall 0.66 vs 0.51. Collapses are a real concern with a different
taxonomy; they route to the separate performance-decline review.

**Rejected:** rolling mean/std (not robust to the anomaly pulling its own window); per-series STL
(most series too short - as low as 10 months post-launch); a global IsolationForest (loses channel
seasonality, nothing an analyst can check against the chart).

**Attribution.** Independent LightGBM binary classifiers, one per cause (not one softmax) -
causes co-occur, and forcing probabilities to sum to 1 would misrepresent "70% likely this AND
55% likely that" as "60/40." Models are small and heavily regularized (60 trees, 7 leaves, min 5
samples/leaf) - with 7-14 independent events per cause, anything larger would fit noise. **Rejected:** single
multi-class softmax (contradicts "distribution, not a label"); rule-based attribution (brittle at
the boundaries, no calibrated confidence to know when to override it); a neural multi-label model
(unjustifiable at this label volume).

## 4. Feature engineering

All features are z-scores/ratios against the same baseline machinery used for detection, re-used
for `iroas`, `rroi`, `spend`, `ctr`, `cpc`, `picr`, `impression_share`, and channel spend share:

| Feature | Separates |
|---|---|
| `roas_iroas_divergence` | genuine gain (both move) vs. spend/mix artifact (ROAS only) |
| `roas_rroi_divergence` | transient single-month noise vs. sustained shift |
| `spend_z`, `mix_share_z` | spend-reduction / mix-shift artifacts specifically |
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

**Detector:** precision/recall against injected ground truth (curve above).

**Attribution, without clean labels in the real world.** Cross-validation is grouped by *event*,
not by row - the key methodological choice here, and implemented with scikit-learn's
`StratifiedGroupKFold`. One event spans up to 3 months x several channels, so its rows are
near-duplicates in feature space, and row-wise splitting puts the same shock in train and test. That leak inflated PR-AUC
by up to 0.47 here: external demand scored 0.57 row-wise versus 0.10 event-wise, i.e. no skill
reported as skill. The corollary is that **effective sample size is events, not rows** - a cause
with 144 rows from 8 events is an 8-sample problem.

Out-of-fold results (`output/attribution_metrics.csv`); lift is included because PR-AUC is not
comparable across causes of differing prevalence:

| cause | events | PR-AUC | base | lift |
|:---|---:|---:|---:|---:|
| Survivorship bias | 12 | 0.84 | 0.034 | 24.5x |
| Creative refresh | 13 | 0.54 | 0.032 | 16.8x |
| Genuine efficiency gain | 14 | 0.56 | 0.053 | 10.5x |
| External demand spike | 7 | 0.36 | 0.076 | 4.7x |
| Spend reduction artifact | 12 | 0.21 | 0.123 | 1.7x |
| **Mix shift artifact** | **8** | **0.15** | **0.154** | **0.95x** |

Top-1 hit rate is 0.47. Four causes carry strong signal; spend-reduction is weak;
**mix-shift has learned nothing** - it sits at its own base rate. That is diagnostic, not noise:
the two weak causes are exactly the two that move budget without touching quality. The model
separates budget mechanics from genuine gains reliably, but cannot tell the two budget causes
apart. Fixing it needs more events, or a feature separating them directly - whether *other*
channels absorbed the budget, which mix shift implies and a pure cut does not.

The synthetic-label model is only a cold start; the production evaluation loop is
analyst-agreement rate over time (see *Productionisation*).

Predicted probabilities are mostly low - at precision 0.36 most flagged months have no injected
cause, and the model correctly stays unsure rather than confabulating.

## 6. Causal honesty

**Causal by construction:** the *true* incrementality factor behind iROAS - never fully observed
even in the synthetic data; what the system sees is a measured, demand-confounded proxy.
**Associational, and presented as such:** ROAS, RROI, and every attribution output. The model is
a differential-diagnosis tool, not a causal-inference engine - "this pattern is consistent with a
creative refresh," never "the refresh caused this," a distinction that lives in the narrative
templates (`src/attribution.py`), not just here. **For a non-technical audience:** every output
pairs a probability with its evidence - "creative refresh (28% likely): CTR/PICR jumped sharply
and suddenly" - so an analyst can judge the evidence rather than trust a bare score, and sees an
explicit "flagged, unexplained" when nothing is confident. A real causal claim needs a designed
experiment (geo holdout, PSA test) or a validated MMM counterfactual - neither runs here.

## 7. Productionisation

**Retraining:** monthly, aligned to the reporting cycle, on a rolling 24-36 month window.
**Monitoring:** analyst agreement rate on attribution (the real accuracy proxy); feature drift on
leading indicators (PSI/KS, monthly); alert-volume drift (a jump in flag rate is itself worth
investigating); rolling Brier score and lift-over-base per cause,
so a cause degrading to chance (as mix shift already has) surfaces automatically instead of
looking authoritative. **Cold start:** exercised here (one brand
launches mid-panel) - new brand-channels borrow the peer trend and sigma until a 3-month
own-history floor. Next step, not built: cluster brands into spend/mix archetypes so new brands
borrow from the nearest archetype rather than the full pool.
**Human-in-the-loop:** every analyst confirm/correct/reject is logged and feeds the next retrain -
how the synthetic-label cold start gets replaced by real signal.

## 8. Limits - where this breaks

The simulator's assumptions are the ceiling: if real response curves, seasonality, or confounding
differ materially, both models need re-validation first. The functional gap is the one in
*Evaluation* - the system cannot separate the two budget-mechanics causes from each other, only
from genuine gains. Short-history brands underperform on peer priors that may not fit, and the
causal seasonal index is blind for a panel's first 12 months. Attribution has no "none of the
above": a driver outside the taxonomy (competitor stockout, macro shock) yields uniformly low
probabilities, surfaced as "flagged, unexplained." Forecasting isn't built - it needs planned
spend separated from what isn't known in advance. Highest-value next inputs: analyst-confirmed
labels, placement-level spend/revenue, longer per-brand history.
