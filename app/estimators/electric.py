"""Quantitative estimators for NASA-STD-4010 Lightning Launch Commit Criteria
LLCC-8 (surface electric field / field mill rule) and LLCC-7 (smoke plume /
pyrocumulus rule), from a model forecast context (no field-mill or fire data).

LLCC-8 regime model and sources
--------------------------------
Real field-mill measurements are not a continuous function of CAPE or cloud
depth, they are a discontinuous, highly local quantity dominated by cloud
charge structure and ground corona. We approximate the physics with four
literature-anchored regimes and a smooth (piecewise-linear) index between
them, driven by the forecast quantities the rule guidance calls out
(cloud cover / precipitation, mixed-phase cloud depth 0 to -20 C, CAPE, and
observed/recent thunderstorm weather codes):

  0. Fair weather            ~80-150 V/m,  mid ~115 V/m
     Chalmers, J.A. (1967) "Atmospheric Electricity", Pergamon; Israel, H.
     (1973) "Atmospheric Electricity"; MacGorman & Rust (1998) "The
     Electrical Nature of Storms", Oxford Univ. Press -- fair-weather
     vertical field at the surface is consistently reported ~100-130 V/m.

  1. Non-electrified cloud / light precipitation   ~200-800 V/m, mid ~400 V/m
     Overcast skies and light rain raise the field a few-fold over fair
     weather via space charge and precipitation/triboelectric charging even
     without organized thunderstorm electrification (Chalmers 1967, ch. on
     "disturbed weather"; MacGorman & Rust 1998).

  2. Electrified cloud (deep cloud spanning the mixed-phase zone, 0 to -20 C,
     with CAPE present -- nimbostratus / towering cumulus / young Cb)
     ~1000-5000 V/m, mid ~2500 V/m
     Marshall, T.C. & Marsh, S.J. (1993) "Negative Charge Location in
     Stratiform Clouds", J. Geophys. Res.; Stolzenburg, M. & Marshall, T.C.
     (various) measured nimbostratus/stratiform charge layers producing
     surface fields of order 1-4 kV/m; non-inductive graupel-ice charging in
     the -10 to -20 C mixed-phase zone (Takahashi 1978; Saunders 1993) is the
     accepted mechanism, consistent with the ctx "layers" 0/-20 C guidance.

  3. Active thunderstorm (WMO code 95/96/99 now or in the last 3 h)
     ~3000-10000 V/m, mid ~6000 V/m
     Standler, R.B. & Winn, W.P. (1979) "Effects of coronae on electric
     fields beneath small thunderstorms", Q. J. R. Meteorol. Soc. 105,
     285-302 -- corona space charge from the ground/vegetation limits
     (saturates) the surface field under active storms to roughly a few kV/m
     up to ~10 kV/m even though the cloud-base field is far larger; this
     saturation is why field mills, not a simple E-field extrapolation, are
     used operationally.

  KSC operational context: the 1500 V/m / 5 nmi criterion and the KSC/CCAFS
  field mill network are described in the NASA/KSC Lightning Launch Commit
  Criteria technical reports (Krider, E.P.; Willett, J.C.; Merceret, F.J.,
  et al., "Natural and Triggered Lightning Launch Commit Criteria", NASA/KSC
  TM series, and NASA-STD-4010 itself).

Calibration honesty
--------------------
There is no public field-mill dataset to statistically fit these regimes
against for 2015+: NASA's KSC field mill archive on NASA Earthdata ends in
2012 and requires an Earthdata login to even inspect. The regime anchors
above are literature order-of-magnitude values, not a regression fit to KSC
field mills. Treat every LLCC-8 output as a physically-motivated estimate,
not a calibrated instrument reading -- confidence is capped at "medium".

LLCC-7 (smoke plume)
---------------------
A model forecast carries no fire or plume location data, so whether the
flight path passes through a pyrocumulus is fundamentally unknowable from
ctx alone. We optionally report a documented fire-weather proxy (low RH +
high wind + no precip -> elevated fire spread/plume potential, cf. the
US NWS/NIFC Fire Weather Watch/Red Flag criteria) purely as context in
"basis". "violated" stays None unless there is a real fire/plume
observation -- this estimator never fabricates one.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# LLCC-8 helpers
# ---------------------------------------------------------------------------

_IDX = [0.0, 1.0, 2.0, 3.0]
_MID = [115.0, 400.0, 2500.0, 6000.0]
_LOW = [80.0, 200.0, 1000.0, 3000.0]
_HIGH = [150.0, 800.0, 5000.0, 10000.0]

_THUNDER_CODES = (95, 96, 99)


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            frac = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return ys[-1]


def _find_freezing_level_ft(levels: list[dict]) -> float | None:
    """Interpolate the 0 C crossing from the level profile (surface -> aloft)."""
    prev = None
    for lvl in levels:
        t = lvl.get("temp_c")
        ft = lvl.get("ft")
        if t is None or ft is None:
            continue
        if prev is not None:
            pt, pft = prev
            if (pt >= 0 and t < 0) or (pt <= 0 and t > 0):
                if t != pt:
                    frac = (0 - pt) / (t - pt)
                    return pft + frac * (ft - pft)
                return ft
        prev = (t, ft)
    return None


def _mixed_phase_depth_ft(layers: list[dict], freezing_ft: float | None) -> float:
    """Thickest cloud layer that spans from at/below the freezing level up
    through at least -10 C -- the non-inductive graupel-ice charging zone."""
    if freezing_ft is None or not layers:
        return 0.0
    best = 0.0
    for layer in layers:
        top_c = layer.get("top_c")
        base_ft = layer.get("base_ft")
        if top_c is None or base_ft is None:
            continue
        if top_c <= -10.0 and base_ft <= freezing_ft:
            thickness = layer.get("thickness_ft")
            if thickness is None and layer.get("top_ft") is not None:
                thickness = layer["top_ft"] - base_ft
            if thickness:
                best = max(best, thickness)
    return best


def _regime_index(ctx: dict) -> float:
    surface = ctx.get("surface") or {}
    cape = surface.get("cape") or 0.0
    precip = surface.get("precipitation") or 0.0
    cc_low = surface.get("cloud_cover_low") or 0.0
    cc_mid = surface.get("cloud_cover_mid") or 0.0
    cc_high = surface.get("cloud_cover_high") or 0.0

    freezing_ft = surface.get("freezing_level_ft")
    if freezing_ft is None:
        freezing_ft = _find_freezing_level_ft(ctx.get("levels") or [])
    mixed_phase_ft = _mixed_phase_depth_ft(ctx.get("layers") or [], freezing_ft)

    cloud_term = min(1.0, (cc_mid + cc_high + 0.5 * cc_low) / 150.0) * 0.4
    precip_term = min(1.0, precip / 5.0) * 0.3
    mixed_term = min(1.0, mixed_phase_ft / 8000.0) * 1.0
    cape_term = min(1.0, cape / 1500.0) * 1.0

    idx = min(3.0, cloud_term + precip_term + mixed_term + cape_term)

    codes = [ctx.get("weather_code")] + list(ctx.get("weather_codes_last3h") or [])
    if any(c in _THUNDER_CODES for c in codes if c is not None):
        idx = 3.0
    return idx


def estimate_llcc8(ctx: dict) -> dict:
    idx = _regime_index(ctx)
    value = round(_interp(idx, _IDX, _MID), 1)
    low = round(_interp(idx, _IDX, _LOW), 1)
    high = round(_interp(idx, _IDX, _HIGH), 1)
    threshold = 1500.0

    if idx < 0.3:
        confidence = "medium"  # decades-old, well-replicated fair-weather baseline
    elif idx >= 3.0:
        confidence = "medium"  # saturation mechanism is well documented; magnitude isn't site-calibrated
    else:
        confidence = "low"  # interpolated electrified-cloud regime, no site calibration

    return {
        "id": "LLCC-8",
        "quantity": "estimated surface electric field magnitude (1-min average, field-mill equivalent)",
        "value": value,
        "unit": "V/m",
        "low": low,
        "high": high,
        "threshold": threshold,
        "threshold_desc": "1-min avg |E| > 1500 V/m within 5 nmi of pad (NASA-STD-4010 LLCC-8 field mill rule)",
        "violated": value > threshold,
        "confidence": confidence,
        "basis": (
            "4-regime literature model (fair weather -> non-electrified cloud/light "
            "precip -> electrified cloud through the 0/-20 C mixed-phase zone with "
            "CAPE -> active thunderstorm), blended by a continuous index built from "
            "cloud cover, precipitation, mixed-phase cloud depth, CAPE, and WMO "
            "weather codes 95/96/99. Regime anchors: fair weather ~100-150 V/m "
            "(Chalmers 1967; Israel 1973; MacGorman & Rust 1998); electrified "
            "stratiform/cumulus ~1-5 kV/m (Marshall & Marsh 1993 JGR; Stolzenburg & "
            "Marshall stratiform-charge studies); active-storm surface field "
            "saturated by ground corona to ~3-10 kV/m (Standler & Winn 1979, QJRMS "
            "105:285-302). KSC field-mill network and the 1500 V/m/5 nmi criterion "
            "itself: NASA/KSC LLCC technical reports (Krider, Willett, Merceret et "
            "al.) and NASA-STD-4010."
        ),
        "calibration": (
            "Not statistically fit to field mills: NASA's KSC field mill archive on "
            "Earthdata ends in 2012 and requires an Earthdata login even to inspect, "
            "so there is no public post-2012 ground truth. Regime values above are "
            "literature order-of-magnitude anchors interpolated smoothly by forecast "
            "proxies, not a calibrated instrument reading -- treat as order-of-"
            "magnitude guidance, confidence capped at medium."
        ),
    }


# ---------------------------------------------------------------------------
# LLCC-7
# ---------------------------------------------------------------------------

def _fire_weather_proxy_note(ctx: dict) -> str:
    surface = ctx.get("surface") or {}
    levels = ctx.get("levels") or []
    rh0 = None
    for lvl in levels:
        if lvl.get("rh") is not None:
            rh0 = lvl["rh"]
            break
    wind = surface.get("wind_kt")
    gust = surface.get("gust_kt")
    precip = surface.get("precipitation")

    if rh0 is None or wind is None:
        return "insufficient data for even a fire-weather proxy (no near-surface RH/wind)"
    dry_windy = rh0 < 25 and max(wind, gust or 0) >= 15 and (precip or 0) == 0
    if dry_windy:
        return (
            f"low RH ({rh0:.0f}%), wind {wind:.0f} kt, no precip -- pattern "
            "resembles US NWS/NIFC Red Flag Warning criteria (elevated fire "
            "spread potential), for context only"
        )
    return f"RH {rh0:.0f}%, wind {wind:.0f} kt -- no elevated fire-weather proxy signal"


def estimate_llcc7(ctx: dict) -> dict:
    return {
        "id": "LLCC-7",
        "quantity": "plausibility of a pyrocumulus / smoke-plume-generated cumulus cloud in the flight path",
        "value": None,
        "unit": None,
        "low": None,
        "high": None,
        "threshold": None,
        "threshold_desc": "flight path passes through a cumulus cloud formed by a smoke plume, e.g. wildfire pyroconvection (NASA-STD-4010 LLCC-7)",
        "violated": None,
        "confidence": "low",
        "basis": (
            "A model forecast carries no fire location, smoke, or plume-top data, "
            "so whether a pyrocumulus exists cannot be determined from ctx. "
            f"Fire-weather proxy (context only, not evidence of a plume): "
            f"{_fire_weather_proxy_note(ctx)}."
        ),
        "calibration": (
            "No calibration is possible without an actual fire/smoke observation "
            "(satellite hotspot/plume detection, NOTAM, or range safety report). "
            "'violated' is left None on principle here -- this estimator will never "
            "assert a smoke plume from atmospheric fields alone."
        ),
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _levels(freezing_ft):
        # Simple linear lapse from a surface temp down to -60C at 45000 ft,
        # with the 0C crossing placed at freezing_ft.
        pts = [(1000, 0), (850, 5000), (700, 10000), (500, 20000), (300, 32000), (200, 39000)]
        out = []
        for hpa, ft in pts:
            # 2 C / 1000 ft roughly, anchored so temp=0 at freezing_ft
            temp_c = 0 - 0.002 * (ft - freezing_ft) * 1000 / 1000
            out.append({"hPa": hpa, "ft": ft, "temp_c": temp_c, "rh": 40, "wind_kt": 10, "wind_dir": 270})
        return out

    fair = {
        "levels": _levels(12000),
        "surface": {"cape": 0, "cin": 0, "lifted_index": 4, "precipitation": 0,
                     "cloud_cover_low": 0, "cloud_cover_mid": 0, "cloud_cover_high": 5,
                     "visibility": 10, "freezing_level_ft": 12000, "wind_kt": 8, "gust_kt": 12},
        "weather_code": 0, "weather_codes_last3h": [0, 0, 0],
        "layers": [],
    }

    stratiform = {
        "levels": _levels(8000),
        "surface": {"cape": 300, "cin": -20, "lifted_index": 0, "precipitation": 2.0,
                     "cloud_cover_low": 90, "cloud_cover_mid": 85, "cloud_cover_high": 30,
                     "visibility": 4, "freezing_level_ft": 8000, "wind_kt": 12, "gust_kt": 18},
        "weather_code": 61, "weather_codes_last3h": [61, 51, 61],
        "layers": [{"base_ft": 3000, "top_ft": 18000, "top_c": -18, "thickness_ft": 15000}],
    }

    thunderstorm = {
        "levels": _levels(12000),
        "surface": {"cape": 2500, "cin": -30, "lifted_index": -6, "precipitation": 15.0,
                     "cloud_cover_low": 95, "cloud_cover_mid": 95, "cloud_cover_high": 90,
                     "visibility": 2, "freezing_level_ft": 12000, "wind_kt": 18, "gust_kt": 35},
        "weather_code": 95, "weather_codes_last3h": [95, 96, 95],
        "layers": [{"base_ft": 2000, "top_ft": 45000, "top_c": -60, "thickness_ft": 43000}],
    }

    r_fair = estimate_llcc8(fair)
    r_strat = estimate_llcc8(stratiform)
    r_ts = estimate_llcc8(thunderstorm)

    for label, r in [("fair", r_fair), ("stratiform", r_strat), ("thunderstorm", r_ts)]:
        print(f"{label:12s} value={r['value']:>8.1f} V/m  range=[{r['low']:.0f}, {r['high']:.0f}]  "
              f"violated={r['violated']}  confidence={r['confidence']}")

    assert r_fair["value"] < 1500 and r_fair["violated"] is False
    assert r_ts["violated"] is True
    assert r_fair["value"] < r_strat["value"] < r_ts["value"]
    assert r_fair["low"] < r_strat["low"] < r_ts["low"]
    assert r_fair["high"] < r_strat["high"] < r_ts["high"]

    r7 = estimate_llcc7(thunderstorm)
    print("\nLLCC-7:", r7["quantity"])
    print(" violated:", r7["violated"], "| basis:", r7["basis"])
    assert r7["violated"] is None and r7["value"] is None

    print("\nAll self-checks passed.")
