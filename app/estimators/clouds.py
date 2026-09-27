"""Quantitative estimators for four cloud-geometry Lightning Launch Commit Criteria
(NASA-STD-4010), from a model forecast sounding.

Each rule gets a real physical estimate (not just pass/fail), built from:
  - piecewise-linear interpolation of the RH profile between pressure levels, so cloud
    base/top can fall *between* the given levels instead of snapping to them, and
  - a pseudo-adiabatic parcel lift (Bolton 1980 LCL, Rogers & Yau pseudoadiabatic lapse
    rate) from the lowest level to its equilibrium level, as a second, independent
    estimate of convective cloud top for LLCC-10.

Thresholds are the rule's own numbers: 4500 ft (LLCC-6), +5 C (LLCC-10), -10 C (LLCC-9),
and the forecast freezing level (LLCC-5/6). Everything else here is an estimate with a
stated method and confidence, not a certified value.

CALIBRATION (LLCC-10 convective top): compared against 113 real CCSFS/KSC scrub-attempt hours
with an observed radar echo top (10 nmi ring, falling back to 25 nmi), pulled from
~/Documents/scrubline/artifacts/all_events.json (date/time/pad + echo_top_10nm_kft /
echo_top_25nm_kft), cross-referenced hour-by-hour with archived Open-Meteo historical-forecast
soundings (relative_humidity/temperature/geopotential_height on the same 10 pressure levels,
plus cape). 150 of 207 available (date, site) pairs were sampled (seed 42); 113 yielded both a
fittable parcel and an observed echo top. Measured:
    n = 113, bias (obs - raw_parcel_EL) = -19.4 kft, Pearson r = 0.18, RMSE after bias = 16.2 kft
The raw (undilute) parcel EL badly over-estimates real echo tops: mean observed top was
11.7 kft, mean raw parcel EL was 31.0 kft. Two real effects drive that, worth stating rather
than hiding: (1) an undilute parcel has no entrainment, so it always runs deeper than real,
entraining cumulus; (2) 61 of the 113 cases never found an equilibrium level within the given
1000-200 hPa sounding at all (still buoyant at 200 hPa, ~38-41 kft) and got clamped there,
which floods the sample with near-ceiling predictions regardless of how strong the real storm
was -- a real limitation of stopping the sounding at 200 hPa, not a tuning artifact.
Correlation is weak (r=0.18): the bias correction re-centers the estimate but does not make it
precise, so LLCC-10 confidence is capped at "low" (see CUMULUS_TOP_CORR / _confidence cap).
Raw calibration rows and script: see calibrate.py output referenced in this task's report.
"""
import math

RD = 287.05       # J/(kg K), dry air gas constant
CP = 1005.7       # J/(kg K), dry air specific heat at constant pressure
LV = 2.501e6      # J/kg, latent heat of vaporization at 0 C
EPS = 0.622       # Rd/Rv

THICK_LAYER_FT = 4500.0     # LLCC-6
CUMULUS_TOP_C = 5.0         # LLCC-10: violated if top colder than +5 C
TRIBO_C = -10.0             # LLCC-9: violated at/colder than -10 C
CAPE_FLOOR = 10.0           # J/kg; below this, "convective" parcel lift is not meaningful

# --- Calibration, fit against 113 real echo tops (see module docstring) ---
CUMULUS_TOP_BIAS_FT = -19400.0  # add to raw parcel-EL height: undilute parcels + the 200 hPa
                                 # sounding ceiling both push raw EL well above real echo tops
CUMULUS_TOP_RMSE_FT = 16200.0   # residual scatter after the bias correction; sets the low/high band
CUMULUS_TOP_CORR = 0.18         # weak; caps confidence at "low" regardless of margin
CUMULUS_TOP_N = 113
CALIBRATION_NOTE = (
    "parcel-EL top height calibrated against 113 CCSFS/KSC scrub-attempt hours with an "
    "observed 10/25 nmi radar echo top (all_events.json x Open-Meteo historical-forecast "
    "profiles): bias -19,400 ft (raw undilute pseudoadiabat, often clamped at the 200 hPa "
    "sounding ceiling, runs far deeper than real entraining cumulus), r=0.18, RMSE 16,200 ft "
    "after bias correction. r=0.18 is a weak fit -- a single grid-column sounding with no "
    "entrainment cannot resolve individual cumulus cells, and about half the sample never "
    "reached an equilibrium level within 1000-200 hPa at all -- so confidence on LLCC-10 tops "
    "is capped at 'low' regardless of margin. LLCC-5, LLCC-6 and LLCC-9 have no matching "
    "real-world dataset (no independent ceilometer/cloud-depth truth in all_events.json) and "
    "are NOT calibrated: their bands come "
    "from the forecast's own pressure-level spacing (linear-interpolation resolution), not a fit."
)


# ---------- shared level/layer helpers ----------

def _valid_levels(levels):
    out = [l for l in (levels or []) if l and l.get("ft") is not None and l.get("temp_c") is not None]
    out.sort(key=lambda l: l["ft"])
    return out


def _freezing_level_ft(ctx):
    sfc = (ctx.get("surface") or {})
    if sfc.get("freezing_level_ft") is not None:
        return float(sfc["freezing_level_ft"])
    lv = _valid_levels(ctx.get("levels"))
    for a, b in zip(lv, lv[1:]):
        if a["temp_c"] >= 0 and b["temp_c"] <= 0 and a["temp_c"] != b["temp_c"]:
            frac = a["temp_c"] / (a["temp_c"] - b["temp_c"])
            return a["ft"] + frac * (b["ft"] - a["ft"])
    if lv and lv[0]["temp_c"] < 0:
        return lv[0]["ft"]
    return None


def _bracket_gap_ft(lv, z):
    """Spacing (ft) of the two levels bracketing height z -- the method's own vertical
    resolution at that height, used as an honest (not invented) uncertainty band."""
    if not lv:
        return None
    if len(lv) == 1:
        return 0.0
    fts = [l["ft"] for l in lv]
    if z <= fts[0]:
        return fts[1] - fts[0]
    if z >= fts[-1]:
        return fts[-1] - fts[-2]
    for a, b in zip(fts, fts[1:]):
        if a <= z <= b:
            return b - a
    return fts[-1] - fts[-2]


def interpolated_layers(levels, rh_threshold=90.0):
    """Cloud layers from the RH profile, with base/top linearly interpolated between
    pressure levels (not snapped to the nearest one)."""
    lv = [l for l in _valid_levels(levels) if l.get("rh") is not None]
    if len(lv) < 1:
        return []
    out = []
    cur_base = lv[0]["ft"] if lv[0]["rh"] >= rh_threshold else None
    for a, b in zip(lv, lv[1:]):
        ra, rb = a["rh"], b["rh"]
        if (ra - rh_threshold) * (rb - rh_threshold) < 0:
            frac = (rh_threshold - ra) / (rb - ra)
            z_cross = a["ft"] + frac * (b["ft"] - a["ft"])
            if ra < rh_threshold <= rb:
                cur_base = z_cross
            else:
                t_cross = a["temp_c"] + frac * (b["temp_c"] - a["temp_c"])
                if cur_base is not None:
                    out.append({"base_ft": cur_base, "top_ft": z_cross, "top_c": t_cross})
                cur_base = None
    if cur_base is not None:
        out.append({"base_ft": cur_base, "top_ft": lv[-1]["ft"], "top_c": lv[-1]["temp_c"]})
    for l in out:
        l["thickness_ft"] = l["top_ft"] - l["base_ft"]
    return out


def _dewpoint_c(t_c, rh):
    if rh is None or rh <= 0 or t_c is None:
        return None
    rh = min(rh, 100.0)
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return (b * gamma) / (a - gamma)


def _es_hpa(t_c):
    return 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))


def _ws(t_c, p_hpa):
    es = _es_hpa(t_c)
    return EPS * es / max(p_hpa - es, 1e-6)


def _lcl(t_c, td_c, p_hpa):
    """Bolton (1980) eq. 15: LCL temperature and pressure from a dry-bulb/dewpoint pair."""
    T, Td = t_c + 273.15, td_c + 273.15
    if Td >= T:
        Td = T - 0.1
    t_lcl = 1.0 / (1.0 / (Td - 56.0) + math.log(T / Td) / 800.0) + 56.0
    p_lcl = p_hpa * (t_lcl / T) ** (CP / RD)
    return t_lcl - 273.15, p_lcl


def _moist_lapse_dTdp(t_c, p_hpa):
    """Saturated (pseudo-adiabatic) dT/dp, K/hPa (Rogers & Yau form)."""
    T = t_c + 273.15
    ws = _ws(t_c, p_hpa)
    num = RD * T + LV * ws
    den = p_hpa * (CP + (LV ** 2 * ws * EPS) / (RD * T ** 2))
    return num / den


def parcel_equilibrium_level(levels, cape):
    """Lift the lowest-level parcel dry-adiabatically to its LCL, then pseudo-adiabatically
    above, against the environment sounding (log-p interpolated). Returns the equilibrium
    level (where the parcel first cools back below the environment after being warmer) as
    {'el_ft','el_c','lcl_ft'}, or None if there's no usable low-level data or CAPE is below
    CAPE_FLOOR (no meaningful convection to lift)."""
    lv = [l for l in _valid_levels(levels) if l.get("rh") is not None and l.get("hPa")]
    if len(lv) < 3 or cape is None or cape <= CAPE_FLOOR:
        return None
    sfc = lv[0]
    td0 = _dewpoint_c(sfc["temp_c"], sfc["rh"])
    if td0 is None:
        return None
    t_lcl, p_lcl = _lcl(sfc["temp_c"], td0, sfc["hPa"])

    ps = sorted(set(l["hPa"] for l in lv))
    if len(ps) < 3 or p_lcl <= min(ps):
        return None
    ln_p = [math.log(l["hPa"]) for l in lv]
    order = sorted(range(len(lv)), key=lambda i: ln_p[i])
    ln_p_s = [ln_p[i] for i in order]
    t_s = [lv[i]["temp_c"] for i in order]
    z_s = [lv[i]["ft"] for i in order]

    def env_at(p):
        lp = math.log(p)
        return _interp(lp, ln_p_s, t_s), _interp(lp, ln_p_s, z_s)

    p_top = min(ps)
    p, T = p_lcl, t_lcl
    lcl_env_t, lcl_z = env_at(p_lcl)
    buoyant = T > lcl_env_t
    dp = -5.0
    el_p = None
    reached_positive = buoyant
    while p + dp > p_top:
        Tm = T + _moist_lapse_dTdp(T, p) * (dp / 2.0)
        pm = p + dp / 2.0
        T_new = T + _moist_lapse_dTdp(Tm, pm) * dp
        p_new = p + dp
        env_t_new, _ = env_at(p_new)
        buoyant_new = T_new > env_t_new
        if buoyant_new:
            reached_positive = True
        if reached_positive and not buoyant_new:
            el_p = p_new
            break
        T, p = T_new, p_new
    if el_p is None:
        if reached_positive:
            el_p = p_top  # parcel still buoyant at the top of the given sounding
        else:
            return None
    el_t, el_z = env_at(el_p)
    return {"el_ft": el_z, "el_c": el_t, "lcl_ft": lcl_z}


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            f = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]


def _confidence(value, low, high, threshold, colder_is_violation):
    """Confidence attaches to the violated/clear call, not the exact number: 'high' when the
    uncertainty band doesn't reach the threshold and is tight relative to the margin, 'low'
    when the band straddles the threshold (we can't tell which side of the rule we're on)."""
    if value is None:
        return "low"
    band = (high - low) if (low is not None and high is not None) else None
    if band is None:
        return "low"
    crosses = (low <= threshold <= high) if low <= high else (high <= threshold <= low)
    if crosses:
        return "low"
    margin = (threshold - value) if colder_is_violation else (value - threshold)
    if band == 0:
        return "high"
    return "high" if abs(margin) > 2 * band else "medium"


def _result(rid, quantity, value, unit, low, high, threshold, threshold_desc, violated,
            confidence, basis, calibration):
    return {"id": rid, "quantity": quantity, "value": value, "unit": unit, "low": low, "high": high,
            "threshold": threshold, "threshold_desc": threshold_desc, "violated": violated,
            "confidence": confidence, "basis": basis, "calibration": calibration}


# ---------- the four estimators ----------

def estimate_llcc5(ctx) -> dict:
    """Disturbed weather: precipitation with cloud extending through the freezing level."""
    levels = ctx.get("levels") or []
    sfc = ctx.get("surface") or {}
    frz = _freezing_level_ft(ctx)
    lv = _valid_levels(levels)
    layers = interpolated_layers(levels) or (ctx.get("layers") or [])
    precip = sfc.get("precipitation")
    through = [l for l in layers if frz is not None and l["base_ft"] <= frz <= l["top_ft"]]
    depth_above = max((l["top_ft"] - frz for l in through), default=0.0) if frz is not None else None

    if frz is None or not lv:
        return _result("LLCC-5", "cloud depth above the freezing level (ft); precip rate (mm/h)",
                        None, "ft", None, None, 0.0,
                        "any depth of cloud through the freezing level, with precipitation present",
                        None, "low", "no freezing level in the forecast sounding", CALIBRATION_NOTE)

    gap = _bracket_gap_ft(lv, frz) or 0.0
    value = depth_above
    low = max(0.0, value - gap)
    high = value + gap
    violated = bool(precip and precip > 0 and depth_above > 0)
    conf = _confidence(value, low, high, 0.0, colder_is_violation=False) if precip else "medium"
    basis = (f"RH>=90% layers interpolated between pressure levels (not snapped to them); "
             f"freezing level {frz:.0f} ft; precip rate {precip if precip is not None else 0:.2f} mm/h "
             f"from the surface forecast; depth = cloud-layer top minus freezing level for any layer "
             f"spanning it")
    return _result("LLCC-5", "cloud depth above the freezing level (ft); precip rate (mm/h)",
                    round(value, 0), "ft", round(low, 0), round(high, 0), 0.0,
                    "any depth of cloud through the freezing level, with precipitation present",
                    violated, conf, basis, CALIBRATION_NOTE)


def estimate_llcc6(ctx) -> dict:
    """Thick cloud layer: layer > 4500 ft thick spanning the freezing level."""
    levels = ctx.get("levels") or []
    frz = _freezing_level_ft(ctx)
    lv = _valid_levels(levels)
    layers = interpolated_layers(levels) or (ctx.get("layers") or [])
    through = [l for l in layers if frz is not None and l["base_ft"] <= frz <= l["top_ft"]]

    if frz is None or not through:
        return _result("LLCC-6", "thickness of the thickest layer through the freezing level (ft)",
                        0.0 if frz is not None else None, "ft", 0.0 if frz is not None else None,
                        0.0 if frz is not None else None, THICK_LAYER_FT,
                        f"> {THICK_LAYER_FT:.0f} ft thick, spanning the freezing level",
                        False if frz is not None else None,
                        "medium" if frz is not None else "low",
                        "no forecast cloud layer spans the freezing level" if frz is not None
                        else "no freezing level in the forecast sounding", CALIBRATION_NOTE)

    thick = max(through, key=lambda l: l["thickness_ft"])
    value = thick["thickness_ft"]
    gap_base = _bracket_gap_ft(lv, thick["base_ft"]) or 0.0
    gap_top = _bracket_gap_ft(lv, thick["top_ft"]) or 0.0
    band = math.hypot(gap_base, gap_top) / 2.0
    low, high = max(0.0, value - band), value + band
    violated = value > THICK_LAYER_FT
    conf = _confidence(value, low, high, THICK_LAYER_FT, colder_is_violation=False)
    basis = (f"thickest RH>=90% layer spanning the freezing level ({frz:.0f} ft), base/top "
             f"linearly interpolated between pressure levels; band is the bracketing "
             f"pressure-level spacing at the base and top, not a fitted error")
    return _result("LLCC-6", "thickness of the thickest layer through the freezing level (ft)",
                    round(value, 0), "ft", round(low, 0), round(high, 0), THICK_LAYER_FT,
                    f"> {THICK_LAYER_FT:.0f} ft thick, spanning the freezing level",
                    violated, conf, basis, CALIBRATION_NOTE)


def estimate_llcc9(ctx) -> dict:
    """Triboelectrification: flight through visible moisture/cloud at or colder than -10 C."""
    levels = ctx.get("levels") or []
    lv = _valid_levels(levels)
    layers = interpolated_layers(levels) or (ctx.get("layers") or [])

    if not layers:
        return _result("LLCC-9", "coldest cloud temperature on the ascent path (C); "
                        "depth of cloud colder than -10 C (ft)",
                        None, "C", None, None, TRIBO_C,
                        f"in cloud/visible moisture at or colder than {TRIBO_C:.0f} C",
                        False if lv else None, "medium" if lv else "low",
                        "no forecast cloud on the ascent path (RH stays below 90% at every level)",
                        CALIBRATION_NOTE)

    coldest = min(layers, key=lambda l: l["top_c"])
    value = coldest["top_c"]
    cold_layers = [l for l in layers if l["top_c"] <= TRIBO_C]
    depth_cold = sum(l["thickness_ft"] for l in cold_layers)
    gap = _bracket_gap_ft(lv, coldest["top_ft"]) or 0.0
    # Temperature band from the level-to-level lapse rate spanning that gap (a real, derivable
    # quantity: how much temp could change across one un-resolved interpolation interval).
    lapse = 0.0
    for a, b in zip(lv, lv[1:]):
        if a["ft"] <= coldest["top_ft"] <= b["ft"] and b["ft"] != a["ft"]:
            lapse = abs(b["temp_c"] - a["temp_c"])
            break
    band = lapse / 2.0
    low, high = value - band, value + band
    violated = value <= TRIBO_C
    conf = _confidence(value, low, high, TRIBO_C, colder_is_violation=True)
    basis = (f"coldest top temperature among RH>=90% layers (interpolated between pressure "
             f"levels) reached on the ascent; {depth_cold:.0f} ft of that cloud is colder than "
             f"{TRIBO_C:.0f} C; band is the temperature spread across the un-resolved "
             f"pressure-level interval bracketing the coldest point")
    return _result("LLCC-9", "coldest cloud temperature on the ascent path (C); "
                    "depth of cloud colder than -10 C (ft)",
                    round(value, 1), "C", round(low, 1), round(high, 1), TRIBO_C,
                    f"in cloud/visible moisture at or colder than {TRIBO_C:.0f} C",
                    violated, conf, basis, CALIBRATION_NOTE)


def estimate_llcc10(ctx) -> dict:
    """Cumulus: cumulus with tops colder than +5 C. Primary estimate is a pseudo-adiabatic
    parcel lift (equilibrium level), bias-corrected against real radar echo tops; falls back
    to the interpolated RH layer when there isn't enough sounding data to lift a parcel."""
    levels = ctx.get("levels") or []
    sfc = ctx.get("surface") or {}
    frz = _freezing_level_ft(ctx)
    lv = _valid_levels(levels)
    cape = sfc.get("cape")
    layers = interpolated_layers(levels) or (ctx.get("layers") or [])
    low_layers = [l for l in layers if frz is None or l["base_ft"] < frz]

    parcel = parcel_equilibrium_level(levels, cape)

    if parcel is not None:
        value = parcel["el_ft"] + CUMULUS_TOP_BIAS_FT
        value = max(value, parcel.get("lcl_ft", 0.0))  # can't top out below its own base
        top_c = parcel["el_c"]
        low, high = value - CUMULUS_TOP_RMSE_FT, value + CUMULUS_TOP_RMSE_FT
        conf_cap = "medium" if CUMULUS_TOP_CORR >= 0.3 else "low"
        basis = (f"pseudo-adiabatic parcel lift from the lowest sounding level (Bolton 1980 LCL, "
                 f"Rogers-Yau moist lapse rate) to its equilibrium level, CAPE={cape:.0f} J/kg; "
                 f"bias-corrected {CUMULUS_TOP_BIAS_FT:+.0f} ft against {CUMULUS_TOP_N} observed "
                 f"radar echo tops (r={CUMULUS_TOP_CORR:.2f})")
    elif low_layers:
        deepest = max(low_layers, key=lambda l: l["top_ft"])
        value = deepest["top_ft"]
        top_c = deepest["top_c"]
        gap = _bracket_gap_ft(lv, value) or 0.0
        low, high = value - gap, value + gap
        conf_cap = "low"  # RH-diagnosed stratiform proxy standing in for a cumulus rule
        basis = (f"CAPE ({cape!r} J/kg) is at or below the {CAPE_FLOOR:.0f} J/kg floor for a "
                 f"meaningful parcel lift; fell back to the top of the deepest RH>=90% layer "
                 f"based below the freezing level -- a stratiform proxy, not a cumulus top")
    else:
        return _result("LLCC-10", "convective cloud top temperature (C)",
                        None, "C", None, None, CUMULUS_TOP_C,
                        f"cumulus top colder than +{CUMULUS_TOP_C:.0f} C",
                        False, "medium",
                        f"CAPE ({cape!r} J/kg) below the convective floor and no low cloud layer "
                        f"in the sounding: no cumulus", CALIBRATION_NOTE)

    violated = top_c < CUMULUS_TOP_C
    conf = _confidence(top_c, top_c, top_c, CUMULUS_TOP_C, colder_is_violation=True)
    # cap confidence at the method's own ceiling (parcel fit or RH fallback), whichever is lower
    order = {"low": 0, "medium": 1, "high": 2}
    conf = conf if order[conf] < order[conf_cap] else conf_cap
    # The rule tests top TEMPERATURE, so that is the value; the height and its range go in the quantity text.
    return _result("LLCC-10", f"convective cloud top temperature (C); top height ~{round(value, -2):,.0f} ft "
                              f"(range {round(low, -2):,.0f}-{round(high, -2):,.0f} ft)",
                    round(top_c, 1), "C", None, None, CUMULUS_TOP_C,
                    f"cumulus top colder than +{CUMULUS_TOP_C:.0f} C",
                    violated, conf, basis, CALIBRATION_NOTE)


if __name__ == "__main__":
    def level(hpa, ft, t, rh):
        return {"hPa": hpa, "ft": ft, "temp_c": t, "rh": rh, "wind_kt": 10, "wind_dir": 270}

    def mkctx(levels, cape=0.0, cin=0.0, precip=0.0, frz=None):
        return {"levels": levels,
                "surface": {"cape": cape, "cin": cin, "lifted_index": 0, "precipitation": precip,
                             "cloud_cover_low": 0, "cloud_cover_mid": 0, "cloud_cover_high": 0,
                             "visibility": 10000, "freezing_level_ft": frz, "wind_kt": 10, "gust_kt": 15},
                "weather_code": 0, "weather_codes_last3h": [0, 0, 0], "layers": []}

    # Case 1: clear and dry -- no cloud anywhere, no CAPE.
    clear = mkctx([
        level(1000, 0, 28, 40), level(925, 2500, 22, 35), level(850, 5000, 15, 30),
        level(700, 10000, 4, 25), level(600, 14000, -4, 20), level(500, 18500, -14, 15),
        level(400, 24500, -28, 10), level(300, 31500, -42, 10), level(250, 34500, -48, 10),
        level(200, 38500, -52, 10),
    ], cape=0.0, precip=0.0, frz=16000)
    r5, r6, r9, r10 = estimate_llcc5(clear), estimate_llcc6(clear), estimate_llcc9(clear), estimate_llcc10(clear)
    assert r5["violated"] is False, r5
    assert r6["violated"] is False, r6
    assert r9["violated"] is False, r9
    assert r10["violated"] is False, r10
    print("clear/dry:", r5["value"], r6["value"], r9["value"], r10)

    # Case 2: deep moist convection -- big CAPE, saturated column well past the freezing level.
    convective = mkctx([
        level(1000, 0, 29, 85), level(925, 2500, 22, 92), level(850, 5000, 16, 95),
        level(700, 10000, 5, 96), level(600, 14000, -3, 95), level(500, 18500, -12, 93),
        level(400, 24500, -25, 88), level(300, 31500, -40, 80), level(250, 34500, -48, 60),
        level(200, 38500, -54, 30),
    ], cape=2500.0, cin=0.0, precip=8.0, frz=15800)
    r5, r6, r9, r10 = (estimate_llcc5(convective), estimate_llcc6(convective),
                       estimate_llcc9(convective), estimate_llcc10(convective))
    assert r5["violated"] is True, r5
    assert r6["violated"] is True, r6
    assert r9["violated"] is True, r9
    assert r10["violated"] is True, r10
    print("deep convective:", r5["value"], r6["value"], r9["value"], r10["value"], r10["basis"][:60])

    # Case 3: high cirrus only -- cold thin layer near the tropopause, nothing near the
    # freezing level, no CAPE. Must NOT trip LLCC-10 (that's the whole point of the rule:
    # high cirrus isn't cumulus).
    cirrus = mkctx([
        level(1000, 0, 27, 30), level(925, 2500, 21, 28), level(850, 5000, 14, 25),
        level(700, 10000, 3, 20), level(600, 14000, -5, 18), level(500, 18500, -15, 20),
        level(400, 24500, -30, 40), level(300, 31500, -45, 93), level(250, 34500, -52, 91),
        level(200, 38500, -56, 40),
    ], cape=0.0, precip=0.0, frz=16200)
    r5, r6, r9, r10 = estimate_llcc5(cirrus), estimate_llcc6(cirrus), estimate_llcc9(cirrus), estimate_llcc10(cirrus)
    assert r10["violated"] is False, r10          # cirrus-only must not violate the cumulus rule
    assert r9["violated"] is True, r9             # cirrus is still visible moisture colder than -10 C
    assert r6["violated"] is False, r6            # thin, and not through the freezing level
    print("cirrus-only:", r5["value"], r6["value"], r9["value"], r10)

    print("all self-checks passed")
