"""
Synthetic data generator for the CPG media-spike exercise.

Design intent (see design_proposal.md for the full rationale):

- ROAS is generated through a *diminishing-returns response curve*
  (Revenue = f(spend)), so a pure spend cut mechanically raises ROAS
  without any change in underlying quality. This is what makes the
  "spend reduction artifact" and "mix shift artifact" cause types
  emerge naturally from the generative process rather than being
  hand-scripted as label-specific hacks.
- iROAS is generated from a latent "efficiency" factor that is
  *independent of spend level*, so it does not move when spend is cut
  or reallocated. It is, however, partially confounded by external
  category demand shocks (a deliberate, documented imperfection -
  real-world incrementality measurement is rarely fully clean either).
  This asymmetry (ROAS reacts to spend/mix, iROAS mostly doesn't) is
  the core signal the attribution model learns to exploit.
- RROI is an EWMA/shrinkage blend of the attributable-sales ratio
  toward its own trailing 3-month average, i.e. a deliberately
  "stickier" metric. Divergence between RROI and ROAS in a given month
  is itself evidence that something transient (not a sustained
  efficiency change) is happening.
- Leading indicators (CTR, CPC, PICR, impression share) are driven by
  the *same* efficiency factor as iROAS, not by spend/mix, which is
  what lets an analyst (or model) tell a genuine efficiency gain apart
  from a spend/mix artifact even though both raise ROAS.

The generator also injects and records ground-truth event metadata
(cause types, weights, window, shape) purely for offline evaluation.
This ground truth is never exposed to the detector/attribution models
as a feature.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RNG_SEED = 7

CAUSE_TYPES = [
    "genuine_efficiency_gain",
    "spend_reduction_artifact",
    "mix_shift_artifact",
    "survivorship_bias",
    "external_demand_spike",
    "creative_refresh",
]

CHANNEL_UNIVERSE = {
    # channel: (base_roas_range, base_iroas_range, response_elasticity_p, base_ctr, base_cpc, base_picr, base_impr_share)
    # response_elasticity_p is intentionally more concave (further from 1.0) than a first pass:
    # p close to 1 makes ROAS almost insensitive to spend changes, which made spend-driven
    # artifacts (spend cuts, mix shift) statistically invisible - unrealistic given the brief's
    # own example of a 40% ROAS jump. These values keep diminishing returns economically
    # plausible while making budget-driven ROAS swings large enough to actually trigger review.
    "tv": ((1.4, 2.2), (1.1, 1.7), 0.60, 0.004, 0.0, 0.06, 0.55),
    "ooh": ((1.1, 1.8), (0.9, 1.4), 0.58, 0.003, 0.0, 0.04, 0.40),
    "paid_search": ((3.5, 5.5), (2.6, 4.0), 0.80, 0.045, 1.85, 0.14, 0.70),
    "social": ((2.4, 3.6), (1.8, 2.7), 0.74, 0.012, 0.95, 0.09, 0.60),
    "display": ((1.8, 2.8), (1.3, 2.0), 0.70, 0.006, 0.70, 0.05, 0.50),
    "digital_video": ((2.0, 3.2), (1.5, 2.3), 0.72, 0.008, 1.10, 0.07, 0.55),
    "audio": ((1.6, 2.5), (1.2, 1.9), 0.68, 0.005, 0.85, 0.05, 0.45),
    "retail_media": ((3.2, 4.8), (2.4, 3.6), 0.80, 0.020, 1.40, 0.16, 0.65),
    "affiliate": ((2.8, 4.2), (2.0, 3.1), 0.76, 0.030, 1.20, 0.12, 0.58),
}
CPC_HAS_NO_COST_CHANNELS = {"tv", "ooh"}  # bought on impressions/GRPs, not clicks -> CPC undefined

BRANDS = [
    # name, spend_tier (relative scale), channels (6-8 of the 9), season_profile, launch_month_index (0 = full history)
    dict(name="Solstice Foods", tier=3.0, category="food_beverage",
         channels=["tv", "paid_search", "social", "display", "digital_video", "retail_media", "affiliate"],
         season="holiday_q4", launch_month=0),
    dict(name="Nimbus Home", tier=1.0, category="home_personal_wellness",
         channels=["display", "paid_search", "social", "digital_video", "audio", "retail_media"],
         season="steady", launch_month=0),
    dict(name="Verdant Snacks", tier=0.5, category="food_beverage",
         channels=["social", "paid_search", "digital_video", "retail_media", "affiliate", "display"],
         season="summer_peak", launch_month=0),
    dict(name="Crestline Beverages", tier=2.2, category="food_beverage",
         channels=["tv", "ooh", "paid_search", "social", "digital_video", "retail_media", "audio", "affiliate"],
         season="summer_peak", launch_month=0),
    dict(name="Halcyon Personal Care", tier=0.7, category="home_personal_wellness",
         channels=["paid_search", "social", "display", "digital_video", "retail_media", "affiliate"],
         season="back_to_school", launch_month=0),
    dict(name="Ridgeway Grocery", tier=1.6, category="food_beverage",
         channels=["tv", "paid_search", "social", "display", "retail_media", "affiliate", "audio"],
         season="holiday_q4", launch_month=0),
    dict(name="Aster Wellness", tier=0.9, category="home_personal_wellness",
         channels=["social", "paid_search", "digital_video", "display", "retail_media", "audio"],
         season="back_to_school", launch_month=0),
    dict(name="Pinebrook Pet Co.", tier=0.35, category="home_personal_wellness",
         channels=["social", "paid_search", "display", "retail_media", "affiliate", "digital_video"],
         season="steady", launch_month=20),  # cold-start / short-history brand
]

# 36 months (the top of the brief's 24-36 range) for two concrete reasons: the strictly-causal
# seasonal index needs prior occurrences of each calendar month to exist at all, and grouped
# cross-validation is limited by the number of independent *events*, not rows - a longer panel
# across more brands is the only honest way to raise that count.
N_MONTHS = 36
START_PERIOD = pd.Period("2023-01", freq="M")
MONTHS = pd.period_range(START_PERIOD, periods=N_MONTHS, freq="M")


def _season_curve(profile: str, month_idx: np.ndarray) -> np.ndarray:
    """Asymmetric, non-sinusoidal seasonal multiplier by calendar month."""
    cal_month = ((month_idx) % 12) + 1
    if profile == "holiday_q4":
        # sharp ramp into Nov/Dec, slow bleed-off into Jan/Feb
        base = np.array([0.82, 0.80, 0.90, 0.95, 0.95, 0.92, 0.90, 0.95, 1.05, 1.20, 1.55, 1.75])
    elif profile == "summer_peak":
        base = np.array([0.85, 0.85, 0.95, 1.05, 1.20, 1.35, 1.40, 1.25, 1.05, 0.90, 0.80, 0.80])
    elif profile == "back_to_school":
        base = np.array([0.90, 0.90, 0.95, 0.95, 0.95, 0.95, 1.10, 1.45, 1.30, 1.00, 0.95, 1.05])
    elif profile == "steady":
        base = np.array([0.95, 0.95, 1.00, 1.00, 1.05, 1.05, 1.00, 1.00, 1.00, 1.00, 1.05, 1.10])
    else:
        raise ValueError(profile)
    return base[cal_month - 1]


def _ramp_shape(window_len: int, shape: str) -> np.ndarray:
    """Effect intensity across a spike window, in [0, 1]."""
    if shape == "sudden":
        # full effect in month 1, quick partial decay after (captures 1-month spikes)
        w = np.array([1.0] + [0.35] * (window_len - 1))
    else:  # gradual, 3-month build
        w = np.linspace(0.35, 1.0, window_len)
    return w


@dataclasses.dataclass
class SpikeEvent:
    event_id: int
    brand: str
    channels: Tuple[str, ...]  # empty tuple => brand-wide (mix shift)
    start_idx: int
    length: int
    shape: str  # "sudden" | "gradual"
    causes: Dict[str, float]  # cause_type -> weight (sums to 1 across causes in this event)


@dataclasses.dataclass
class CategoryDemandEvent:
    event_id: int
    category: str
    brands: Tuple[str, ...]
    start_idx: int
    length: int
    shape: str
    weight: float

    @property
    def event_key(self) -> str:
        return f"category:{self.category}#{self.event_id}"


def _sample_events(rng: np.random.Generator, brand_cfg: dict, brand_channels: List[str]) -> List[SpikeEvent]:
    """A brand gets 6-10 local spike events scattered across its history, some multi-causal."""
    # note: mix_shift events are brand-wide (touch every active channel for the event's duration),
    # so they contribute far more labeled (brand,channel,month) rows per
    # event than the single/dual-channel cause types. Their sampling weight is kept lower than a
    # naive "equal chance per cause type" would suggest, so the labeled panel stays a minority of
    # total brand-channel-months (an "anomaly" that touches a quarter of the panel isn't anomalous).
    # Sampling weights are balanced on *independent events per cause*, not rows. Grouped CV
    # (one event = one group) means a cause with 100 rows from 6 events is a 6-sample problem,
    # and estimates at that n are worthless - mix_shift previously scored below its own base
    # rate for exactly this reason. Brand-wide mix shifts generate many rows per event, so they
    # need a *lower* row share but a comparable event count. Category demand is sampled separately.
    n_events = rng.integers(6, 11)
    combos = [
        (("genuine_efficiency_gain",), "gradual", 0.14),
        (("genuine_efficiency_gain", "creative_refresh"), "sudden", 0.08),
        (("spend_reduction_artifact",), "sudden", 0.18),
        (("spend_reduction_artifact", "mix_shift_artifact"), "gradual", 0.06),
        (("mix_shift_artifact",), "gradual", 0.12),
        (("survivorship_bias",), "sudden", 0.15),
        (("creative_refresh",), "sudden", 0.27),
    ]
    weights = np.array([c[2] for c in combos])
    weights = weights / weights.sum()

    events = []
    launch = brand_cfg["launch_month"]
    usable_len = N_MONTHS - launch
    for i in range(n_events):
        combo_idx = rng.choice(len(combos), p=weights)
        causes, default_shape, _ = combos[combo_idx]
        shape = default_shape if rng.random() < 0.75 else ("sudden" if default_shape == "gradual" else "gradual")
        length = 1 if shape == "sudden" else 3
        if usable_len - length - 1 <= 0:
            continue
        start_idx = launch + int(rng.integers(1, usable_len - length))
        # mix shifts are brand-wide phenomena; other local causes hit 1-2 channels
        if "mix_shift_artifact" in causes:
            channels = tuple()
        else:
            k = min(len(brand_channels), rng.integers(1, 3))
            channels = tuple(rng.choice(brand_channels, size=k, replace=False))
        # split weight across co-occurring causes (not always even)
        raw = rng.dirichlet(np.ones(len(causes)) * 2.5)
        cause_weights = {c: float(w) for c, w in zip(causes, raw)}
        events.append(SpikeEvent(len(events), brand_cfg["name"], channels, start_idx, length, shape, cause_weights))
    return events


def _sample_category_demand_events(rng: np.random.Generator) -> List[CategoryDemandEvent]:
    """Sample shared shocks that lift every brand in the affected category."""
    categories: Dict[str, Tuple[str, ...]] = {}
    for category in sorted({b["category"] for b in BRANDS}):
        categories[category] = tuple(b["name"] for b in BRANDS if b["category"] == category)

    events: List[CategoryDemandEvent] = []
    for category, brands in categories.items():
        # Three independent events per category keeps grouped CV feasible without making
        # category-wide rows dominate the panel.
        starts = rng.choice(np.arange(2, N_MONTHS - 3), size=3, replace=False)
        for event_id, start_idx in enumerate(sorted(starts)):
            shape = "sudden" if rng.random() < 0.8 else "gradual"
            length = 1 if shape == "sudden" else 3
            events.append(
                CategoryDemandEvent(
                    event_id=event_id,
                    category=category,
                    brands=brands,
                    start_idx=int(start_idx),
                    length=length,
                    shape=shape,
                    weight=float(rng.uniform(0.75, 1.0)),
                )
            )
    return events


def generate_dataset(seed: int = RNG_SEED) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    panel_rows = []
    all_events: List[SpikeEvent] = []
    month_idx_arr = np.arange(N_MONTHS)

    # brand-level idiosyncrasy so two brands on the same channel aren't identical
    brand_idio = {b["name"]: rng.uniform(0.85, 1.15) for b in BRANDS}

    # Shared category-level demand timelines. Unlike brand-specific shocks, these events are
    # sampled once and applied to every brand in a category, so "lifted all boats" is observable.
    category_events = _sample_category_demand_events(rng)
    category_demand_shock = {b["name"]: np.ones(N_MONTHS) for b in BRANDS}
    for ev in category_events:
        idxs = np.arange(ev.start_idx, min(ev.start_idx + ev.length, N_MONTHS))
        ramp = _ramp_shape(ev.length, ev.shape)[: len(idxs)]
        for brand in ev.brands:
            category_demand_shock[brand][idxs] *= 1 + 0.38 * ev.weight * ramp

    per_brand_channel_params = {}
    for b in BRANDS:
        for ch in b["channels"]:
            roas_lo, roas_hi = CHANNEL_UNIVERSE[ch][0]
            iroas_lo, iroas_hi = CHANNEL_UNIVERSE[ch][1]
            base_roas = rng.uniform(roas_lo, roas_hi) * brand_idio[b["name"]]
            base_iroas = rng.uniform(iroas_lo, iroas_hi) * brand_idio[b["name"]]
            base_spend_share = rng.uniform(0.7, 1.3)
            per_brand_channel_params[(b["name"], ch)] = dict(
                base_roas=base_roas, base_iroas=base_iroas, base_spend_share=base_spend_share,
            )

    for b in BRANDS:
        brand_channels = b["channels"]
        events = _sample_events(rng, b, brand_channels)
        all_events.extend(events)

        # channel-effect timelines, keyed by channel -> array over N_MONTHS
        eff_factor = {ch: np.ones(N_MONTHS) for ch in brand_channels}       # genuine gain / creative refresh
        spend_shock = {ch: np.ones(N_MONTHS) for ch in brand_channels}      # spend reduction
        surv_effect = {ch: np.zeros(N_MONTHS) for ch in brand_channels}     # survivorship bias, in [0, ~0.4]
        mix_shift_active = np.zeros(N_MONTHS)                               # brand-wide intensity in [0,1]
        demand_shock = category_demand_shock[b["name"]].copy()              # category-wide multiplier

        for ev in events:
            w = _ramp_shape(ev.length, ev.shape)
            idxs = np.arange(ev.start_idx, ev.start_idx + ev.length)
            idxs = idxs[idxs < N_MONTHS]
            w = w[: len(idxs)]
            for cause, weight in ev.causes.items():
                mag = weight
                if cause in ("genuine_efficiency_gain", "creative_refresh"):
                    bump = (0.50 if cause == "genuine_efficiency_gain" else 0.60) * mag
                    targets = ev.channels if ev.channels else brand_channels
                    for ch in targets:
                        eff_factor[ch][idxs] += bump * w
                elif cause == "spend_reduction_artifact":
                    cut = 0.55 * mag
                    targets = ev.channels if ev.channels else brand_channels
                    for ch in targets:
                        spend_shock[ch][idxs] *= (1 - cut * w)
                elif cause == "survivorship_bias":
                    drop = 0.45 * mag
                    targets = ev.channels if ev.channels else brand_channels
                    for ch in targets:
                        surv_effect[ch][idxs] = np.maximum(surv_effect[ch][idxs], drop * w)
                elif cause == "mix_shift_artifact":
                    mix_shift_active[idxs] = np.maximum(mix_shift_active[idxs], mag * w)

        # ---- 1) base spend intent per channel-month ----
        base_spend = {}
        for ch in brand_channels:
            share = per_brand_channel_params[(b["name"], ch)]["base_spend_share"]
            trend = 1 + 0.004 * month_idx_arr * rng.uniform(-0.3, 1.0)  # mild brand-channel trend, mostly flat/up
            season = _season_curve(b["season"], month_idx_arr)
            noise = rng.lognormal(mean=0, sigma=0.06, size=N_MONTHS)
            level = 40_000 * b["tier"] * share
            base_spend[ch] = level * trend * season * noise * spend_shock[ch]

        # ---- 2) mix shift: reallocate spend toward higher base-ROAS channels ----
        if mix_shift_active.any():
            roas_rank = sorted(brand_channels, key=lambda c: per_brand_channel_params[(b["name"], c)]["base_roas"])
            n = len(roas_rank)
            low_channels = roas_rank[: max(1, n // 2)]
            high_channels = roas_rank[max(1, n // 2):]
            for t in range(N_MONTHS):
                intensity = mix_shift_active[t]
                if intensity <= 0:
                    continue
                pull = 0.40 * intensity
                total_low = sum(base_spend[c][t] for c in low_channels)
                moved = total_low * pull
                for c in low_channels:
                    frac = base_spend[c][t] / total_low if total_low > 0 else 0
                    base_spend[c][t] -= moved * frac
                for c in high_channels:
                    base_spend[c][t] += moved / len(high_channels)

        # ---- 3) revenue / roas / iroas / mas / rroi / leading indicators ----
        for ch in brand_channels:
            p = per_brand_channel_params[(b["name"], ch)]
            spend = np.maximum(base_spend[ch], 1.0)
            spend_ref = np.median(spend[:6]) if N_MONTHS >= 6 else np.median(spend)
            elasticity = CHANNEL_UNIVERSE[ch][2]
            diminishing = (spend / spend_ref) ** (elasticity - 1)

            surv_mult = 1 + 1.0 * surv_effect[ch]  # reported-metric inflation from dropped low performers
            rev_noise = rng.lognormal(mean=0, sigma=0.09, size=N_MONTHS)
            revenue = spend * p["base_roas"] * diminishing * eff_factor[ch] * demand_shock * surv_mult * rev_noise
            roas = revenue / spend

            iroas_noise = rng.lognormal(mean=0, sigma=0.07, size=N_MONTHS)
            demand_confound = 1 + 0.30 * (demand_shock - 1)  # imperfect incrementality measurement
            iroas_true = p["base_iroas"] * eff_factor[ch]
            iroas_measured = iroas_true * demand_confound * iroas_noise

            mas_noise = rng.lognormal(mean=0, sigma=0.08, size=N_MONTHS)
            mas = spend * iroas_measured * mas_noise

            attributable_ratio = mas / spend
            s = pd.Series(attributable_ratio)
            trailing = s.rolling(3, min_periods=1).mean().to_numpy()
            rroi = 0.4 * attributable_ratio + 0.6 * trailing

            ctr = CHANNEL_UNIVERSE[ch][3] * eff_factor[ch] ** 1.1 * rng.lognormal(0, 0.10, N_MONTHS)
            if ch in CPC_HAS_NO_COST_CHANNELS:
                cpc = np.full(N_MONTHS, np.nan)
            else:
                cpc = CHANNEL_UNIVERSE[ch][4] / (eff_factor[ch] ** 0.8) * rng.lognormal(0, 0.10, N_MONTHS)
            picr = CHANNEL_UNIVERSE[ch][5] * eff_factor[ch] ** 1.3 * rng.lognormal(0, 0.12, N_MONTHS)
            impression_share = np.clip(
                CHANNEL_UNIVERSE[ch][6] * (1 - surv_effect[ch]) * rng.lognormal(0, 0.05, N_MONTHS), 0.02, 0.98
            )

            for t in range(N_MONTHS):
                if t < b["launch_month"]:
                    continue
                panel_rows.append(dict(
                    brand=b["name"], channel=ch, month=str(MONTHS[t]),
                    month_idx=t, spend=spend[t], revenue=revenue[t],
                    roas=roas[t], iroas=iroas_measured[t], mas=mas[t], rroi=rroi[t],
                    ctr=ctr[t], cpc=cpc[t], picr=picr[t], impression_share=impression_share[t],
                ))

    panel = pd.DataFrame(panel_rows)

    # ---- missingness: reporting gaps ----
    miss_mask = rng.random(len(panel)) < 0.03
    for col in ["ctr", "cpc", "picr", "impression_share"]:
        col_mask = rng.random(len(panel)) < 0.02
        panel.loc[col_mask, col] = np.nan
    panel.loc[miss_mask, ["spend", "revenue", "roas", "iroas", "mas", "rroi"]] = np.nan

    panel = panel.sort_values(["brand", "channel", "month_idx"]).reset_index(drop=True)

    # ---- ground truth tables (evaluation-only; never fed to the models as features) ----
    event_rows = []
    label_rows = []
    for ev in all_events:
        b_cfg = next(b for b in BRANDS if b["name"] == ev.brand)
        targets = ev.channels if ev.channels else tuple(b_cfg["channels"])
        end_idx = min(ev.start_idx + ev.length - 1, N_MONTHS - 1)
        event_key = f"brand:{ev.brand}#{ev.event_id}"
        event_rows.append(dict(
            event_id=ev.event_id, event_key=event_key, brand=ev.brand, brands=ev.brand,
            category=b_cfg["category"], channels=",".join(targets),
            scope="channel" if ev.channels else "brand_wide",
            start_month=str(MONTHS[ev.start_idx]), end_month=str(MONTHS[end_idx]),
            shape=ev.shape, causes=";".join(f"{c}:{w:.2f}" for c, w in ev.causes.items()),
        ))
        for ch in targets:
            for t in range(ev.start_idx, end_idx + 1):
                row = dict(brand=ev.brand, channel=ch, month=str(MONTHS[t]),
                           event_id=ev.event_id, event_key=event_key, shape=ev.shape)
                for c in CAUSE_TYPES:
                    row[c] = ev.causes.get(c, 0.0)
                label_rows.append(row)

    for ev in category_events:
        end_idx = min(ev.start_idx + ev.length - 1, N_MONTHS - 1)
        targets = sorted({ch for b in BRANDS if b["name"] in ev.brands for ch in b["channels"]})
        event_rows.append(dict(
            event_id=ev.event_id, event_key=ev.event_key, brand="CATEGORY", brands=",".join(ev.brands),
            category=ev.category, channels=",".join(targets), scope="category_wide",
            start_month=str(MONTHS[ev.start_idx]), end_month=str(MONTHS[end_idx]),
            shape=ev.shape, causes=f"external_demand_spike:{ev.weight:.2f}",
        ))
        for brand in ev.brands:
            brand_cfg = next(b for b in BRANDS if b["name"] == brand)
            for ch in brand_cfg["channels"]:
                for t in range(max(ev.start_idx, brand_cfg["launch_month"]), end_idx + 1):
                    row = dict(brand=brand, channel=ch, month=str(MONTHS[t]),
                               event_id=ev.event_id, event_key=ev.event_key, shape=ev.shape)
                    for cause in CAUSE_TYPES:
                        row[cause] = ev.weight if cause == "external_demand_spike" else 0.0
                    label_rows.append(row)

    events_df = pd.DataFrame(event_rows)
    labels_df = pd.DataFrame(label_rows)
    if not labels_df.empty:
        # group_id for leakage-free cross-validation. One event spans several months and often
        # several channels, so its rows are near-duplicates in feature space; splitting on rows
        # would put the same shock in train and test. Rows are therefore grouped by event - and
        # because a (brand,channel,month) can be touched by two overlapping events, those events
        # must land in the same group too. That is exactly connected components over the
        # row<->event graph, resolved here with union-find.
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for _, grp in labels_df.groupby(["brand", "channel", "month"])["event_key"]:
            keys = grp.unique()
            for k in keys[1:]:
                union(keys[0], k)

        labels_df["group_id"] = labels_df["event_key"].map(find)

        # a (brand,channel,month) can be touched by >1 event; sum cause weights and renormalize
        agg = labels_df.groupby(["brand", "channel", "month"])[CAUSE_TYPES].sum().reset_index()
        totals = agg[CAUSE_TYPES].sum(axis=1)
        for c in CAUSE_TYPES:
            agg[c] = np.where(totals > 0, agg[c] / totals.replace(0, np.nan), 0.0)
        groups = labels_df.groupby(["brand", "channel", "month"])["group_id"].first().reset_index()
        event_keys = (
            labels_df.groupby(["brand", "channel", "month"])["event_key"]
            .agg(lambda s: ";".join(sorted(set(s))))
            .rename("event_keys")
            .reset_index()
        )
        labels_df = agg.merge(groups, on=["brand", "channel", "month"], how="left")
        labels_df = labels_df.merge(event_keys, on=["brand", "channel", "month"], how="left")

    return dict(panel=panel, events=events_df, labels=labels_df)


if __name__ == "__main__":
    out = generate_dataset()
    print(out["panel"].shape, out["events"].shape, out["labels"].shape)
    print(out["panel"].head())
