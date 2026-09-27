"""Quantitative estimators for three convective Lightning Launch Commit Criteria
(NASA-STD-4010) from a model forecast: LLCC-1 (lightning), LLCC-2/3 (attached/detached
anvil), LLCC-4 (debris cloud from a decayed storm).

Ordering convention: ``weather_codes_last3h`` is read oldest-first, last element = the
hour immediately before "now" -- the same convention app/rules_estimate.py uses
(``recent = [h["weather_code"][j] for j in range(max(0, i-3), i)]``).

CALIBRATION (LLCC-1, the only rule here with a real outcome dataset)
----------------------------------------------------------------------
Ground truth: GOES-GLM nearest-flash distance per past launch attempt,
``~/Documents/scrubline/artifacts/all_events.json`` field ``glm_nearest_nm``
(``None`` = no flash found within the ~60 nmi GLM search radius -- a right-censored
"far" reading, not a missing value).

Forecast features: the archived Open-Meteo hindcast at that attempt hour, joined by
``event_index`` from ``data/weather/train_table.csv`` + ``unexplained_table.csv`` into
``all_events.json`` (event_index is a direct 0-based index into that list -- verified by
matching mission name + timestamp). Forecast weather codes for the same site/hour came
from ``~/Documents/scrubline/cache/hist_forecast/days/*.json`` (``time``/``weather_code``,
matched on the table's own resolved ``site`` column [CCSFS/KSC] + the hour floor of
``time_utc``); this only covered 99 of 192 rows (the hist_forecast cache starts at
2024-12-31 and 93 archived attempts predate it), and none of the 99 matched hours carried
a 95/96/99 code, so the "thunder code now" term below is carried in the formula but is
untested by this fit -- it is applied as a deterministic override instead (see below).

Fit: 192 archived forecast hours have both a resolved event_index and a GLM search result.
Built an integer score = count of {forecast CAPE >= 1500 J/kg, lifted index <= -6 C,
precipitation > 0 mm/h} true (0-3), then took the empirical hit rate (flash found within
10 nmi) and the mean nearest-flash distance (censored at 60 nmi when no flash was found)
in each score bin:

    score  n    hit-rate(<=10nmi)   mean dist (60-capped)   IQR (25-75%)
    0      112  0.000 (0/112)       58.1 nmi                60.0 - 60.0
    1      41   0.049 (2/41)        48.3 nmi                44.1 - 60.0
    2      35   0.086 (3/35)        48.6 nmi                32.1 - 60.0
    3      4    0.250 (1/4)         31.7 nmi                24.2 - 40.9

Base rate over all 192 rows: 6/192 = 3.1%. A plain 3-feature logistic regression
(cape, lifted_index, precipitation -> flash<=10nmi) trained on the same 192 rows scores
AUC 0.79 in-sample / 0.75 under 5-fold stratified cross-validation (scikit-learn
LogisticRegression, sklearn.model_selection.cross_val_predict) -- real discrimination
for n=6 positives, but the bin table above (not the raw logistic weights) is what is
hard-coded here because it is the more interpretable, and n per bin is disclosed directly
so a caller can judge how thin the evidence is (n=4 in the top bin). Brier score of the
fitted logistic under 5-fold CV was 0.0306, essentially the same as always predicting the
0.031 base rate (0.0303) -- rare-event Brier score is dominated by the base rate at this
sample size; the bin table's hit-rate monotonicity (0% -> 4.9% -> 8.6% -> 25%) and AUC are
the more informative calibration signal here.

LLCC-2/3 and LLCC-4 have no labeled outcomes anywhere in all_events.json (its
``rules_fired`` column never contains "LLCC-2", "LLCC-3", or "LLCC-4" across all 1502
events -- only LLCC-1/5/6/10 ever fire), so both are physically-motivated estimates, not
fits; that is stated again in each function's own "calibration" field, with the numbers
that stand in for a real fit (echo-top climatology from the same all_events.json file).
"""
import math

THUNDER_CODES = (95, 96, 99)
ANVIL_TOP_C = -20.0          # NASA-STD-4010 glaciation/anvil-capable storm top; same
                              # threshold app/rules_estimate.py uses for LLCC-2/3.
DEBRIS_WINDOW_MIN = 180.0    # LLCC-4: rule applies up to 3 h after the parent storm ends.
DEBRIS_DECAY_TAU_MIN = 90.0  # order-of-magnitude charge-relaxation time constant, chosen so
                              # residual-charge probability ~0.13 at the 180 min rule boundary
                              # (not fit -- no labeled debris-cloud events exist to fit against).

# --- LLCC-1 bin table (see module docstring for how these numbers were produced) ---
_LLCC1_BINS = {
    #      n     hit_rate  dist_mean  dist_p25  dist_p75  confidence
    0: dict(n=112, hit_rate=0.000, dist_mean=58.1, dist_low=60.0, dist_high=60.0, confidence="high"),
    1: dict(n=41,  hit_rate=0.049, dist_mean=48.3, dist_low=44.1, dist_high=60.0, confidence="medium"),
    2: dict(n=35,  hit_rate=0.086, dist_mean=48.6, dist_low=32.1, dist_high=60.0, confidence="medium"),
    3: dict(n=4,   hit_rate=0.250, dist_mean=31.7, dist_low=24.2, dist_high=40.9, confidence="low"),
}
_GLM_SEARCH_RADIUS_NM = 60.0
# Order-of-magnitude Florida convective flash-rate range used only to turn a hit-rate into a
# flash-count-if-nearby estimate for the "basis" text (Orville & Huffines 2001, J. Appl.
# Meteor. -- Florida NLDN flash-density climatology; Bailey et al. 2014 GOES-16 GLM
# validation): ordinary Florida storm cells produce on the order of 1-4 flashes/min.
_FL_FLASH_RATE_LOW, _FL_FLASH_RATE_HIGH = 1.0, 4.0


def _thunder_recent(codes):
    return any(c in THUNDER_CODES for c in (codes or []) if c is not None)


def estimate_llcc1(ctx) -> dict:
    """LLCC-1: lightning within 10 nmi in the last 30 min."""
    surface = ctx.get("surface") or {}
    cape = surface.get("cape")
    li = surface.get("lifted_index")
    precip = surface.get("precipitation")
    wcode = ctx.get("weather_code")
    thunder_now = wcode in THUNDER_CODES

    if thunder_now:
        value, low, high = 2.0, 0.0, 5.0
        confidence = "high"
        flash_lo, flash_hi = round(15 * _FL_FLASH_RATE_LOW), round(30 * _FL_FLASH_RATE_HIGH)
        basis = (
            "current forecast weather code reports an active thunderstorm (WMO 95/96/99), "
            "taken as a direct proxy for lightning at/near the site -- a deterministic "
            "override of the fitted bin table below, not itself fit (0 of the 192 archived "
            "calibration hours carried a thunderstorm code, so this branch is untested by "
            f"the fit); flash count within 10 nmi/30 min taken as {flash_lo}-{flash_hi} from "
            "typical Florida convective flash rates of 1-4/min (Orville & Huffines 2001; "
            "Bailey et al. 2014 GOES-16 GLM validation), scaled to a 15-30 min active window"
        )
    else:
        score = int(cape is not None and cape >= 1500) + int(li is not None and li <= -6) + \
            int(precip is not None and precip > 0)
        row = _LLCC1_BINS[min(score, 3)]
        value, low, high, confidence = row["dist_mean"], row["dist_low"], row["dist_high"], row["confidence"]
        flash_lo = round(30 * row["hit_rate"] * _FL_FLASH_RATE_LOW, 1)
        flash_hi = round(30 * row["hit_rate"] * _FL_FLASH_RATE_HIGH, 1)
        basis = (
            f"score = count of {{forecast CAPE>=1500 J/kg, lifted index<=-6 C, "
            f"precipitation>0 mm/h}} true = {score} (capped at 3); distance = mean observed "
            f"GOES-GLM nearest-flash distance in that score's bin, right-censored at the "
            f"~{_GLM_SEARCH_RADIUS_NM:.0f} nmi GLM search radius when no flash was found "
            f"(n={row['n']}, hit-rate(flash<=10nmi)={row['hit_rate']:.1%}); flash count "
            f"within 10 nmi/30 min taken as hit-rate x (1-4 flashes/min x 30 min) = "
            f"{flash_lo:.1f}-{flash_hi:.1f}, using the same Florida flash-rate literature as above"
        )

    violated = bool(thunder_now)
    return {
        "id": "LLCC-1",
        "quantity": "estimated nearest GOES-GLM flash distance; flash count within 10 nmi in the last 30 min",
        "value": round(value, 1), "unit": "nmi", "low": round(low, 1), "high": round(high, 1),
        "threshold": 10.0, "threshold_desc": "any flash within 10 nmi in the last 30 min = violation",
        "violated": violated, "confidence": confidence, "basis": basis,
        "calibration": (
            "fit against 192 archived forecast hours with a resolved event_index and a GOES-GLM "
            "search result (all_events.json glm_nearest_nm via data/weather/*_table.csv event_index "
            "join): bin hit-rates 0%/4.9%/8.6%/25% for score 0/1/2/3 (n=112/41/35/4) against a 3.1% "
            "(6/192) base rate; the underlying 3-feature logistic (cape, lifted_index, precipitation) "
            "scores AUC 0.75 under 5-fold stratified CV, Brier 0.0306 vs 0.0303 for the base-rate-only "
            "predictor (rare-event Brier is base-rate-dominated at this n; AUC/monotonicity are the "
            "more informative numbers here). The thunderstorm-code override above has zero matching "
            "samples in this set and is not itself fit."
        ),
    }


def _upper_wind_kt(levels):
    winds = [lv.get("wind_kt") for lv in (levels or [])
             if lv and lv.get("hPa") in (250, 300) and lv.get("wind_kt") is not None]
    return sum(winds) / len(winds) if winds else None


def estimate_llcc23(ctx) -> dict:
    """LLCC-2/3: attached/detached anvil -- presence and downwind extent."""
    layers = ctx.get("layers") or []
    levels = ctx.get("levels") or []
    wcode = ctx.get("weather_code")
    codes_last3h = ctx.get("weather_codes_last3h") or []
    thunder_now = wcode in THUNDER_CODES
    thunder_recent = _thunder_recent(codes_last3h)
    deep_storm = any(l.get("top_c") is not None and l["top_c"] < ANVIL_TOP_C for l in layers)
    anvil_present = bool((thunder_now or thunder_recent) and deep_storm)
    upper_wind = _upper_wind_kt(levels)
    has_data = bool(layers) and wcode is not None

    if anvil_present and upper_wind is not None:
        value = round(upper_wind * 1.0, 1)     # nmi; 1 kt = 1 nmi/hr, ~1 h downwind transport
        low = round(upper_wind * 0.5, 1)       # 30 min of transport
        high = round(upper_wind * 3.0, 1)      # capped at the LLCC-4 3 h aging/dissipation window
        violated, confidence = True, "low"
        basis = (
            f"anvil present: thunderstorm code now or in the last 3 h AND a diagnosed cloud layer "
            f"topping colder than {ANVIL_TOP_C:.0f} C (glaciated, anvil-capable top -- same threshold "
            f"app/rules_estimate.py uses for LLCC-2/3). Downwind extent = mean 250/300 hPa wind speed "
            f"({upper_wind:.0f} kt) x transport time (1 kt = 1 nmi/hr); range covers 0.5-3 h of "
            f"transport/aging, the 3 h bound tied to the LLCC-4 debris-cloud window. Basis: anvils are "
            f"carried downwind at roughly the generating level's wind speed (45th Weather Squadron "
            f"Launch Weather Officer guidance; NASA-STD-4010 LLCC-2/3 discussion of anvil transport)."
        )
    elif anvil_present:
        value = low = high = None
        violated, confidence = True, "low"
        basis = (
            f"anvil present (thunderstorm now/recent + a layer colder than {ANVIL_TOP_C:.0f} C) but no "
            f"250/300 hPa wind report in ctx['levels'] to estimate downwind extent"
        )
    else:
        value = low = high = 0.0
        violated, confidence = False, ("medium" if has_data else "low")
        basis = (
            f"no anvil: {'no thunderstorm code now or in the last 3 h' if not (thunder_now or thunder_recent) else f'no layer colder than {ANVIL_TOP_C:.0f} C diagnosed'}"
        )

    return {
        "id": "LLCC-2/3", "quantity": "estimated anvil presence and downwind extent",
        "value": value, "unit": "nmi", "low": low, "high": high,
        "threshold": None,
        "threshold_desc": "do not fly through attached or detached anvil ice/cirrus from a storm reaching "
                           "the glaciation level (NASA-STD-4010 LLCC-2/3); clearance depends on distance and "
                           "time since generation, not a single number",
        "violated": violated, "confidence": confidence, "basis": basis,
        "calibration": (
            "no LLCC-2 or LLCC-3 outcome is labeled anywhere in all_events.json (rules_fired never "
            "contains them across all 1502 events), so this is not fit to outcomes -- cross-check only: "
            "echo_top_10nm_kft/echo_top_25nm_kft in the same file have a median of 3/8 kft and a max of "
            "54/56 kft (n=215 with a reading); Florida summer -20 C heights typically run ~25-30 kft, so "
            "only the upper tail of observed echo tops would reach an anvil-capable top, consistent with "
            "attached/detached-anvil violations being rare relative to the more common shallower-storm rules."
        ),
    }


def estimate_llcc4(ctx) -> dict:
    """LLCC-4: debris cloud -- minutes since a storm that ended in the last 3 h stopped, and
    the residual charged-cloud probability implied by that age."""
    wcode = ctx.get("weather_code")
    codes = list(ctx.get("weather_codes_last3h") or [])
    thunder_now = wcode in THUNDER_CODES

    last_thunder_hours_ago = None
    for j, c in enumerate(reversed(codes)):   # j=0 -> 1 h ago, j=1 -> 2 h ago, ...
        if c in THUNDER_CODES:
            last_thunder_hours_ago = j + 1
            break

    if thunder_now:
        value = low = high = None
        residual_p = None
        violated, confidence = False, "medium"
        basis = ("the parent storm is still active this hour (thunderstorm code now); LLCC-4 is a "
                 "post-storm rule for a cloud whose generating storm has already ended, so it does not "
                 "apply yet -- LLCC-1/2/3 govern the still-active storm instead")
    elif last_thunder_hours_ago is None:
        value = low = high = None
        residual_p = None
        violated, confidence = False, "medium"
        basis = "no thunderstorm code in the last 3 forecast hours: no recent parent storm for a debris cloud to persist from"
    else:
        low = (last_thunder_hours_ago - 1) * 60.0
        high = last_thunder_hours_ago * 60.0
        value = (low + high) / 2.0
        residual_p = math.exp(-value / DEBRIS_DECAY_TAU_MIN)
        violated = value <= DEBRIS_WINDOW_MIN
        confidence = "low"
        basis = (
            f"most recent thunderstorm-coded hour in weather_codes_last3h was {last_thunder_hours_ago} h "
            f"before now (hourly resolution only, so the storm's end time is bucketed to a "
            f"{low:.0f}-{high:.0f} min window; value is the bucket midpoint); residual-charge probability "
            f"~= exp(-minutes/{DEBRIS_DECAY_TAU_MIN:.0f}) = {residual_p:.2f}, an order-of-magnitude decay "
            f"chosen so it falls to ~0.13 by the 180 min/3 h rule boundary -- not independently fit"
        )

    return {
        "id": "LLCC-4", "quantity": "estimated minutes since the parent storm ended; residual charged-cloud probability",
        "value": None if value is None else round(value, 0), "unit": "min",
        "low": None if low is None else round(low, 0), "high": None if high is None else round(high, 0),
        "threshold": DEBRIS_WINDOW_MIN,
        "threshold_desc": f"debris cloud rule applies for up to {DEBRIS_WINDOW_MIN:.0f} min (3 h) after the "
                           f"parent storm ends (NASA-STD-4010 LLCC-4)",
        "violated": violated, "confidence": confidence, "basis": basis,
        "calibration": (
            "no LLCC-4 outcome is labeled anywhere in all_events.json (rules_fired never contains it "
            "across all 1502 events), so neither the time bucketing nor the residual-charge decay is fit "
            "to outcomes -- the decay constant is chosen only to respect the rule's own 3 h boundary. "
            "Time resolution is capped at 1 h because weather_codes_last3h only carries hourly codes."
        ),
    }


if __name__ == "__main__":
    def level(hpa, ft, t, rh, wind_kt=10, wind_dir=270):
        return {"hPa": hpa, "ft": ft, "temp_c": t, "rh": rh, "wind_kt": wind_kt, "wind_dir": wind_dir}

    def mkctx(levels, layers, cape, li, precip, wcode, codes_last3h):
        return {
            "levels": levels,
            "surface": {"cape": cape, "cin": 0, "lifted_index": li, "precipitation": precip,
                        "cloud_cover_low": 10, "cloud_cover_mid": 10, "cloud_cover_high": 10,
                        "visibility": 10000, "freezing_level_ft": 15000, "wind_kt": 10, "gust_kt": 15},
            "weather_code": wcode, "weather_codes_last3h": codes_last3h, "layers": layers,
        }

    base_levels = [level(1000, 0, 28, 40), level(925, 2500, 22, 35), level(850, 5000, 15, 30),
                   level(700, 10000, 4, 25), level(600, 14000, -4, 20), level(500, 18500, -14, 15),
                   level(400, 24500, -28, 10), level(300, 31500, -42, 10, wind_kt=35),
                   level(250, 34500, -48, 10, wind_kt=40), level(200, 38500, -52, 10)]

    # Case 1: clear -- no CAPE, no precip, no recent storm anywhere.
    clear = mkctx(base_levels, [], cape=50.0, li=4.0, precip=0.0, wcode=1, codes_last3h=[0, 0, 1])

    # Case 2: convective, but no thunderstorm code *now* -- a storm was coded 2-3 h ago and has
    # since ended (tests LLCC-4), and left a deep (< -20 C) layer behind (tests LLCC-2/3's
    # thunder-recent branch) while current CAPE/LI/precip are still elevated (tests LLCC-1's
    # non-override bin table).
    conv_layers = [{"base_ft": 4000, "top_ft": 42000, "top_c": -35, "thickness_ft": 38000}]
    convective = mkctx(base_levels, conv_layers, cape=2200.0, li=-7.0, precip=2.5,
                        wcode=3, codes_last3h=[95, 3, 3])

    # Case 3: active thunderstorm now.
    ts_levels = [level(1000, 0, 29, 85), level(925, 2500, 22, 92), level(850, 5000, 16, 95),
                 level(700, 10000, 5, 96), level(600, 14000, -3, 95), level(500, 18500, -12, 93),
                 level(400, 24500, -25, 88), level(300, 31500, -40, 80, wind_kt=60),
                 level(250, 34500, -48, 60, wind_kt=65), level(200, 38500, -54, 30)]
    ts_layers = [{"base_ft": 3000, "top_ft": 48000, "top_c": -60, "thickness_ft": 45000}]
    thunderstorm = mkctx(ts_levels, ts_layers, cape=3000.0, li=-9.0, precip=6.0,
                          wcode=95, codes_last3h=[3, 3, 95])

    r1_clear, r1_conv, r1_ts = estimate_llcc1(clear), estimate_llcc1(convective), estimate_llcc1(thunderstorm)
    print("LLCC-1  clear:", r1_clear["value"], r1_clear["low"], r1_clear["high"], r1_clear["violated"])
    print("LLCC-1  conv :", r1_conv["value"], r1_conv["low"], r1_conv["high"], r1_conv["violated"])
    print("LLCC-1  ts   :", r1_ts["value"], r1_ts["low"], r1_ts["high"], r1_ts["violated"])
    assert r1_clear["violated"] is False
    assert r1_conv["violated"] is False
    assert r1_ts["violated"] is True                       # thunderstorm must violate LLCC-1
    assert r1_ts["value"] < r1_conv["value"] < r1_clear["value"]   # distance shrinks as risk rises

    r23_clear, r23_conv, r23_ts = estimate_llcc23(clear), estimate_llcc23(convective), estimate_llcc23(thunderstorm)
    print("LLCC-2/3 clear:", r23_clear["value"], r23_clear["violated"])
    print("LLCC-2/3 conv :", r23_conv["value"], r23_conv["violated"])
    print("LLCC-2/3 ts   :", r23_ts["value"], r23_ts["violated"])
    assert r23_clear["violated"] is False and r23_clear["value"] == 0.0
    assert r23_conv["violated"] is True and r23_conv["value"] > 0.0     # recent storm left a deep layer
    assert r23_ts["violated"] is True and r23_ts["value"] > r23_conv["value"]  # stronger upper wind -> farther

    r4_clear, r4_conv, r4_ts = estimate_llcc4(clear), estimate_llcc4(convective), estimate_llcc4(thunderstorm)
    print("LLCC-4  clear:", r4_clear["value"], r4_clear["violated"])
    print("LLCC-4  conv :", r4_conv["value"], r4_conv["violated"])
    print("LLCC-4  ts   :", r4_ts["value"], r4_ts["violated"])
    assert r4_clear["value"] is None and r4_clear["violated"] is False   # no recent storm at all
    assert r4_conv["value"] is not None and r4_conv["violated"] is True  # storm ended ~2-3h ago
    assert r4_ts["value"] is None and r4_ts["violated"] is False         # storm still active now

    print("\nall self-checks passed")
