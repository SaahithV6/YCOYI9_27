"""Estimated values for all ten Lightning Launch Commit Criteria from a model forecast.

These are ESTIMATES, graded as such everywhere they are shown. A forecast has no flash positions, no
radar, no field mill and no measured cloud tops, so scrubline's engine grades those rules UNAVAILABLE.
Here each rule gets a best estimate from the forecast profile instead, with its method, a confidence,
and the instrument that would replace the estimate.

Cloud layers are diagnosed from the forecast's relative humidity on pressure levels: contiguous levels
at or above CLOUD_RH form a layer; its base/top heights come from the geopotential heights and its top
temperature from the level temperature. CLOUD_RH is a method parameter, disclosed on every result.
Rule thresholds are the ones in scrubline/llcc.py (NASA-STD-4010): 10 nmi / 30 min lightning,
4500 ft thick layer through the freezing level, cumulus tops colder than +5 C, 1500 V/m field.

Statuses: ESTIMATED_VIOLATION, ESTIMATED_CLEAR, NOT_ESTIMABLE.
"""
CLOUD_RH = 90            # % RH on a pressure level treated as cloud (method parameter)
THICK_LAYER_FT = 4500    # LLCC-6
CUMULUS_TOP_C = 5.0      # LLCC-10: tops colder than +5 C
TRIBO_C = -10.0          # LLCC-9: visible moisture colder than -10 C (ice-phase charging)
THUNDER = {95, 96, 99}
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
M_FT = 3.28084


def hourly_vars():
    return [f"{v}_{p}hPa" for p in LEVELS for v in ("relative_humidity", "temperature", "geopotential_height")] + \
           ["weather_code", "cape", "convective_inhibition", "freezing_level_height", "precipitation", "cloud_cover_low"]


def layers(h, i):
    """Cloud layers at hour index i: [{base_ft, top_ft, top_c, thickness_ft}] from RH >= CLOUD_RH."""
    out, cur = [], None
    for p in LEVELS:
        rh, t, z = h.get(f"relative_humidity_{p}hPa", [None])[i], h.get(f"temperature_{p}hPa", [None])[i], h.get(f"geopotential_height_{p}hPa", [None])[i]
        if None in (rh, t, z):
            continue
        if rh >= CLOUD_RH:
            if cur is None:
                cur = {"base_ft": round(z * M_FT), "top_ft": round(z * M_FT), "top_c": t}
            else:
                cur.update(top_ft=round(z * M_FT), top_c=t)
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    for l in out:
        l["thickness_ft"] = l["top_ft"] - l["base_ft"]
    return out


def _r(rid, name, status, confidence, method, evidence, replaces):
    return {"id": rid, "rule": name, "status": status, "confidence": confidence, "method": method,
            "evidence": evidence, "measured_by": replaces}


def estimate(h, i):
    ls = layers(h, i)
    frz_ft = round((h["freezing_level_height"][i] or 0) * M_FT)
    cape, cin = h["cape"][i] or 0, h.get("convective_inhibition", [None])[i]
    wcode = h["weather_code"][i]
    rain = h["precipitation"][i] or 0
    recent = [h["weather_code"][j] for j in range(max(0, i - 3), i)]
    thunder_now, thunder_recent = wcode in THUNDER, any(c in THUNDER for c in recent)
    through_frz = [l for l in ls if l["base_ft"] < frz_ft < l["top_ft"]]
    # Cumulus grows from low levels: only layers based below the freezing level count (not high cirrus).
    cold_tops = [l for l in ls if l["top_c"] < CUMULUS_TOP_C and l["base_ft"] < frz_ft]
    very_cold = [l for l in ls if l["top_c"] < TRIBO_C]
    convective = cape > 0 and (cin is None or cin > -50)   # cin: J/kg, negative = inhibition
    lay = [f"{l['base_ft']}-{l['top_ft']} ft, top {l['top_c']:.0f} C" for l in ls] or ["no layer at RH>=%d%%" % CLOUD_RH]
    ev = {"layers": lay, "freezing_level_ft": frz_ft, "cape": cape, "cin": cin, "weather_code": wcode, "rain_mm": rain}

    rules = [
        _r("LLCC-1", "Lightning within 10 nmi / 30 min", "ESTIMATED_VIOLATION" if thunder_now or thunder_recent else "ESTIMATED_CLEAR",
           "medium" if thunder_now else "low", "forecast thunderstorm weather code this hour or the last 3 h",
           {"thunder_now": thunder_now, "thunder_last_3h": thunder_recent, "cape": cape}, "GOES GLM flash positions"),
        _r("LLCC-2/3", "Attached / detached anvil", "ESTIMATED_VIOLATION" if (thunder_now or thunder_recent) and any(l["top_c"] < -20 for l in ls) else "ESTIMATED_CLEAR",
           "low", "thunderstorm code with a cloud layer topping out colder than -20 C (anvil level)",
           {"layers": lay}, "weather radar + satellite anvil edge"),
        _r("LLCC-4", "Debris cloud", "ESTIMATED_VIOLATION" if thunder_recent and not thunder_now else "ESTIMATED_CLEAR",
           "low", "thunderstorm in the last 3 h that has ended (decaying storm leaves charged debris)",
           {"thunder_last_3h": thunder_recent, "thunder_now": thunder_now}, "radar + visual observation"),
        _r("LLCC-5", "Disturbed weather", "ESTIMATED_VIOLATION" if rain > 0 and through_frz else "ESTIMATED_CLEAR",
           "medium", "forecast precipitation with a cloud layer extending through the freezing level",
           {"rain_mm": rain, "layers_through_freezing_level": len(through_frz)}, "radar + surface observation"),
        _r("LLCC-6", "Thick cloud layer (>4500 ft through freezing level)",
           "ESTIMATED_VIOLATION" if any(l["thickness_ft"] > THICK_LAYER_FT for l in through_frz) else "ESTIMATED_CLEAR",
           "medium", f"RH-diagnosed layer thicker than {THICK_LAYER_FT} ft spanning the freezing level",
           {"layers": lay, "freezing_level_ft": frz_ft}, "ceilometer + cloud-top measurement"),
        _r("LLCC-7", "Smoke plume", "NOT_ESTIMABLE", "n/a", "a forecast carries no fire or plume information",
           {}, "fire/plume observation"),
        _r("LLCC-8", "Surface electric field (>1500 V/m)",
           "ESTIMATED_VIOLATION" if thunder_now or (through_frz and convective) else "ESTIMATED_CLEAR",
           "low", "electrified-cloud conditions: thunderstorm, or convective cloud through the freezing level",
           {"thunder_now": thunder_now, "convective": convective, "layers_through_freezing_level": len(through_frz)},
           "45th Weather Squadron field mill network"),
        _r("LLCC-9", "Triboelectrification", "ESTIMATED_VIOLATION" if very_cold else "ESTIMATED_CLEAR",
           "low", f"flight path through a cloud layer colder than {TRIBO_C:.0f} C, including cirrus (ice-phase charging); "
                  "many vehicles carry surface treatments that waive this rule",
           {"layers": lay}, "cloud temperature along the flight path"),
        _r("LLCC-10", "Cumulus cloud (tops colder than +5 C)",
           "ESTIMATED_VIOLATION" if convective and cold_tops else "ESTIMATED_CLEAR",
           "medium", "convective environment (CAPE > 0, weak inhibition) with an RH-diagnosed layer based below the freezing level and topping colder than +5 C (high cirrus excluded)",
           {"convective": convective, "layers": lay}, "radar echo tops + ceilometer"),
    ]
    return {"rules": rules, "evidence": ev, "cloud_rh_threshold": CLOUD_RH}
