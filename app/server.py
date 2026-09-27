"""Scrubline launch weather officer: localhost app.

    cd ~/YCOYI9_27 && .venv/bin/python app/server.py      ->  http://127.0.0.1:8787

GET  /                       the page
GET  /api/stats              problem numbers for the header
GET  /api/assess?date=YYYY-MM-DD&time=HH:MM&site=CCSFS|KSC&vehicle=falcon_9
GET  /api/week?site=CCSFS&vehicle=falcon_9     hourly risk for the next 7 days
POST /api/correct            {"assessment": {...}, "correction": "..."} -> data/corrections.jsonl

Verdicts follow the scrubline rule: a forecast can flag risk but cannot clear the lightning rules
(no flash positions, no cloud layers), so a forecast-only assessment is never GO. It is NO-GO when
the forecast itself shows a violation the rules name (precipitation over the pad, thunderstorm
weather codes); otherwise UNDECIDED with the risk and the missing instruments named.
"""
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import build_weather_table as bwt  # noqa: E402
sys.path.insert(0, str(ROOT / "app"))
import rules_estimate as rules  # noqa: E402
import memory  # noqa: E402
import officer  # noqa: E402

SITES = {"CCSFS": ("Cape Canaveral SFS", 28.49, -80.57), "KSC": ("Kennedy Space Center", 28.59, -80.65)}
VEHICLES = {"falcon_9": "Falcon 9", "falcon_heavy": "Falcon Heavy", "atlas_v": "Atlas V", "vulcan": "Vulcan",
            "new_glenn": "New Glenn", "sls": "SLS"}
THUNDER_CODES = {95, 96, 99}
# Shown on the board but not allowed to decide the call alone: vehicles routinely fly through cirrus with
# triboelectric surface treatments that waive LLCC-9 (see the rule's own waiver provision).
ADVISORY = {"LLCC-9"}
DECIDING = ("medium", "high")
PRETTY = {"cape": "Storm energy (CAPE)", "cape_max5h": "Storm energy, peak ±2 h", "precipitation": "Rain at T-0",
          "precipitation_max5h": "Rain, peak ±2 h", "wind_gusts_10m": "Surface gusts", "wind_gusts_10m_max5h": "Gusts, peak ±2 h",
          "cloud_cover_low": "Low cloud cover", "cloud_cover_low_max5h": "Low cloud, peak ±2 h", "cloud_cover_mid": "Mid cloud cover",
          "cloud_cover_high": "High cloud cover", "freezing_level_height": "Freezing level", "lifted_index": "Lifted index (instability)",
          "max_shear_kt_per_km": "Upper-level wind shear", "wind_250hPa_kt": "Jet-level wind (250 hPa)",
          "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity", "visibility": "Visibility",
          "wind_speed_10m": "Surface wind", "hour_sin": "Time of day", "hour_cos": "Time of day", "month_sin": "Season",
          "month_cos": "Season"}

MODEL = lgb.Booster(model_file=str(ROOT / "models" / "weather_model.txt"))
MODEL_META = json.loads((ROOT / "models" / "weather_model.json").read_text())
FEATURES = MODEL_META["features"]
TABLE = pd.read_csv(ROOT / "data" / "weather" / "train_table.csv")
UNEXPLAINED = pd.read_csv(ROOT / "data" / "weather" / "unexplained_table.csv")
_cache = {}


def _frame(df):
    X = df.drop(columns=[c for c in ("event_index", "mission", "time_utc", "site", "vehicle", "kind", "cause", "label",
                                     "weather_risk_score") if c in df.columns])
    X = pd.concat([X, pd.get_dummies(df[["site", "vehicle"]].astype(str), prefix=["site", "vehicle"], dtype=float)], axis=1)
    return X.reindex(columns=FEATURES, fill_value=0.0).astype(float)


_REF = json.loads((ROOT / "models" / "reference_scores.json").read_text())  # out-of-fold, not in-sample
FLEW_SCORES, WX_SCORES = np.array(_REF["flew"]), np.array(_REF["weather_scrub"])
HIST_X = _frame(pd.concat([TABLE, UNEXPLAINED], ignore_index=True))
HIST_ROWS = pd.concat([TABLE, UNEXPLAINED], ignore_index=True)
_mu, _sd = HIST_X.mean(), HIST_X.std().replace(0, 1)


def fetch_hourly(site, day):
    """Live forecast for dates within 16 days, archived forecast for past dates. Cached per site/day."""
    k = (site, day.isoformat())
    if k in _cache:
        return _cache[k]
    _, lat, lon = SITES[site]
    names = bwt.SURFACE + ["weather_code"] + [f"{v}_{p}hPa" for p in bwt.LEVELS
                                               for v in ("wind_speed", "wind_direction", "geopotential_height")] + rules.hourly_vars() + \
        [f"{v}_{p}hPa" for p in rules.LEVELS for v in ("wind_speed", "wind_direction")] + ["lifted_index", "visibility"]
    hourly = ",".join(dict.fromkeys(names))
    start, end = day - timedelta(days=1), day + timedelta(days=1)
    today = date.today()
    base = "https://api.open-meteo.com/v1/forecast" if end >= today else "https://historical-forecast-api.open-meteo.com/v1/forecast"
    url = f"{base}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&hourly={hourly}&timezone=UTC&wind_speed_unit=kn"
    req = urllib.request.Request(url, headers={"User-Agent": "scrubline/0.2 (research)"})
    h = json.loads(urllib.request.urlopen(req, timeout=60).read())["hourly"]
    _cache[k] = (h, "live forecast" if base.startswith("https://api.") else "archived forecast")
    return _cache[k]


def percentile(sorted_scores, s):
    return round(100 * np.searchsorted(sorted_scores, s) / max(1, len(sorted_scores)))


def assess(day, hhmm, site, vehicle):
    t = datetime.combine(day, datetime.strptime(hhmm, "%H:%M").time()).replace(tzinfo=timezone.utc, minute=0)
    if (day - date.today()).days > 15:
        return {"error": "Forecasts reach 16 days ahead. Pick an earlier date."}
    hourly, source = fetch_hourly(site, day)
    hours = bwt.Hours(hourly=hourly)
    f = hours.features(t)
    if f is None:
        return {"error": "No forecast hour available for that time."}
    row = pd.DataFrame([{**f, "site": site, "vehicle": vehicle}])
    X = _frame(row)
    score = float(MODEL.predict(X)[0])
    contrib = MODEL.predict(X, pred_contrib=True)[0][:-1]
    factors = {}
    for name, c in zip(FEATURES, contrib):
        label = PRETTY.get(name) or ("Vehicle" if name.startswith("vehicle_") else "Site" if name.startswith("site_") else name)
        factors[label] = factors.get(label, 0.0) + float(c)
    top = sorted(factors.items(), key=lambda x: -abs(x[1]))[:6]

    i = hours.at(t)
    est = rules.estimate(hourly, i)
    rain, wcode = f["precipitation"] or 0, hourly["weather_code"][i]
    measured = [
        {"id": "PRECIP", "rule": "Flight through precipitation", "status": "NO_GO" if rain > 0 else "CLEAR_IN_FORECAST",
         "confidence": "forecast", "method": "forecast precipitation at T-0", "evidence": {"rain_mm": rain}, "measured_by": "radar / surface obs",
         "value": rain, "unit": "mm", "threshold": 0, "threshold_desc": "any precipitation on the flight path"},
        {"id": "THUNDER", "rule": "Thunderstorm over the site", "status": "NO_GO" if wcode in THUNDER_CODES else "CLEAR_IN_FORECAST",
         "confidence": "forecast", "method": "forecast WMO weather code (95/96/99 = thunderstorm)", "evidence": {"weather_code": wcode},
         "measured_by": "GOES GLM / radar", "value": wcode, "unit": "WMO code", "threshold": 95, "threshold_desc": "code 95, 96 or 99"},
        {"id": "WIND", "rule": "Surface winds", "status": "MEASURED", "confidence": "forecast",
         "method": f"{round(f['wind_speed_10m'] or 0)} kt sustained, gusts {round(f['wind_gusts_10m'] or 0)} kt at 10 m", "evidence": {},
         "measured_by": "tower anemometers", "value": round(f["wind_gusts_10m"] or 0), "unit": "kt gust", "threshold": None,
         "threshold_desc": "vehicle-specific (unpublished)"},
        {"id": "SHEAR", "rule": "Upper-level wind shear", "status": "MEASURED", "confidence": "forecast",
         "method": "max vector wind change between pressure levels (not in the Space Force POV)", "evidence": {},
         "measured_by": "balloon / jimsphere", "value": round(f["max_shear_kt_per_km"] or 0, 1), "unit": "kt/km", "threshold": None,
         "threshold_desc": "vehicle-specific (unpublished)"},
    ]
    checks = est["rules"] + measured
    profile = []
    for lev in rules.LEVELS:
        val = lambda k: (hourly.get(f"{k}_{lev}hPa") or [None] * (i + 1))[i]  # noqa: E731
        z, tc, rh, ws = val("geopotential_height"), val("temperature"), val("relative_humidity"), val("wind_speed")
        if None not in (z, tc, rh):
            profile.append({"hPa": lev, "ft": round(z * rules.M_FT), "temp_c": tc, "rh": rh, "wind_kt": ws})
    no_go = [c["rule"] for c in measured if c["status"] == "NO_GO"]
    likely = [c["rule"] for c in est["rules"] if c["status"] == "ESTIMATED_VIOLATION" and c["confidence"] in DECIDING
              and c["id"] not in ADVISORY]
    low = [c["rule"] for c in est["rules"] if c["status"] == "ESTIMATED_VIOLATION" and
           (c["confidence"] not in DECIDING or c["id"] in ADVISORY)]
    pct_flew = percentile(FLEW_SCORES, score)
    if no_go:
        verdict, reason = "NO-GO", f"The forecast itself shows {', '.join(no_go).lower()} at T-0."
    elif likely:
        verdict, reason = "LIKELY NO-GO", f"Estimated violation of {', '.join(likely)} from the forecast cloud profile."
    elif pct_flew <= 50:
        verdict, reason = "LIKELY GO", "No medium-confidence rule violation is estimated and the weather risk is no higher than the typical launch that flew."
    else:
        verdict, reason = "UNDECIDED", "No rule violation is estimated with confidence, but the weather risk is above the typical launch that flew."
    reason += (f" Advisory / low-confidence flags: {', '.join(low)}." if low else "") + \
        " Estimates replace instruments a forecast lacks; a real GO needs the Space Force forecast (L-5 to L-1) or observations on the day."

    z = ((X - _mu) / _sd).fillna(0).values[0]
    dist = np.sqrt((((HIST_X - _mu) / _sd).fillna(0).values - z) ** 2).sum(axis=1)
    precedents = []
    for j in np.argsort(dist)[:6]:
        r = HIST_ROWS.iloc[j]
        outcome = "flew" if r.get("label") == 0 else ("weather scrub" if r.get("label") == 1 else "no published cause")
        precedents.append({"mission": r["mission"], "time_utc": str(r["time_utc"])[:16].replace("T", " "), "outcome": outcome,
                           "cause": "" if pd.isna(r.get("cause")) else r.get("cause"), "site": r["site"]})
    wx_share = sum(p["outcome"] == "weather scrub" for p in precedents)

    return {
        "input": {"date": day.isoformat(), "time": hhmm, "site": site, "site_name": SITES[site][0], "vehicle": VEHICLES.get(vehicle, vehicle)},
        "source": source,
        "verdict": verdict,
        "verdict_reason": reason,
        "cloud_layers": est["evidence"]["layers"],
        "profile": profile,
        "freezing_level_ft": est["evidence"]["freezing_level_ft"],
        "cloud_rh_threshold": est["cloud_rh_threshold"],
        "risk": {"score": round(score, 3), "pct_vs_flew": pct_flew,
                 "pct_vs_weather_scrubs": percentile(WX_SCORES, score)},
        "factors": [{"name": n, "push": round(v, 3)} for n, v in top],
        "checks": checks,
        "weather": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in f.items() if not k.endswith(("_sin", "_cos"))},
        "precedents": precedents,
        "precedent_summary": f"{wx_share} of the 6 most similar past attempts were stopped by weather.",
        "model": {"trained_rows": MODEL_META["trained_rows"], "test": MODEL_META["test_metrics"]["lightgbm"]},
    }


def week(site, vehicle):
    days = [date.today() + timedelta(days=d) for d in range(7)]
    out = []
    for d in days:
        hourly, _ = fetch_hourly(site, d)
        hours = bwt.Hours(hourly=hourly)
        for h in range(24):
            t = datetime.combine(d, datetime.min.time()).replace(hour=h, tzinfo=timezone.utc)
            f = hours.features(t)
            if f is None:
                continue
            s = float(MODEL.predict(_frame(pd.DataFrame([{**f, "site": site, "vehicle": vehicle}])))[0])
            i = hours.at(t)
            est = rules.estimate(hourly, i)
            pct = percentile(FLEW_SCORES, s)
            forecast_nogo = (f["precipitation"] or 0) > 0 or hourly["weather_code"][i] in THUNDER_CODES
            likely = [r["id"] for r in est["rules"] if r["status"] == "ESTIMATED_VIOLATION" and r["confidence"] in DECIDING
                      and r["id"] not in ADVISORY]
            call = "NO-GO" if forecast_nogo else "LIKELY NO-GO" if likely else "LIKELY GO" if pct <= 50 else "UNDECIDED"
            out.append({"t": t.strftime("%Y-%m-%dT%H:00"), "score": round(s, 3), "pct_vs_flew": pct, "call": call,
                        "rain": f["precipitation"], "violations": likely})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
    best = sorted((h for h in out if h["call"] == "LIKELY GO" and h["t"] >= now), key=lambda h: h["pct_vs_flew"])[:6]
    return {"site": site, "hours": out, "best_windows": best}


QWEN_CKPT = ROOT / "models" / "river_weather_checkpoint.json"


def qwen(day, hhmm, site, vehicle):
    """Second opinion from the River-fine-tuned Qwen3.5-9B, prompted exactly as in training."""
    if not QWEN_CKPT.exists() or not os.environ.get("RIVER_API_KEY"):
        return {"available": False, "reason": "no River checkpoint or RIVER_API_KEY on this machine"}
    import build_river_weather as brw
    import river_client as river
    ck = json.loads(QWEN_CKPT.read_text())
    t = datetime.combine(day, datetime.strptime(hhmm, "%H:%M").time()).replace(tzinfo=timezone.utc, minute=0)
    hourly, _ = fetch_hourly(site, day)
    f = bwt.Hours(hourly=hourly).features(t)
    if f is None:
        return {"available": False, "reason": "no forecast hour"}
    row = pd.Series({**f, "site": site, "vehicle": vehicle, "time_utc": t.isoformat()})
    msgs = [{"role": "system", "content": brw.SYSTEM}, {"role": "user", "content": brw.describe(row)}]
    client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
    res = client.chat_complete_from_checkpoint(msgs, checkpoint_path=ck["inference"], base_model=ck["base_model"],
                                               max_tokens=64, temperature=0.0, timeout=60,
                                               chat_template_kwargs={"enable_thinking": False})
    body = res.response_json if isinstance(res.response_json, dict) else json.loads(res.response_json)
    text = body["choices"][0]["message"].get("content") or ""
    call = "yes" if '"yes"' in text else "no" if '"no"' in text else "unparseable"
    return {"available": True, "weather_stop": call, "raw": text[-120:], "checkpoint": ck["inference"].split("/")[-1]}


def stats():
    ev = json.loads((Path.home() / "Documents/scrubline/artifacts/all_events.json").read_text())
    unk = sum(e["category"] == "UNKNOWN" for e in ev)
    known = [e for e in ev if e["kind"] == "SCRUB" and e["category"] != "UNKNOWN"]
    wx = [e for e in known if e["category"] in ("WX_LAUNCH_SITE", "WX_UPPER_WINDS", "WX_RECOVERY")]
    thr = np.percentile(FLEW_SCORES, 90)  # move the riskiest 10% of launch hours (out-of-fold ranking)
    return {"weather_scrubs": len(wx), "known_cause_scrubs": len(known), "weather_share_pct": round(100 * len(wx) / len(known)),
            "avoidable_pct_moving_top10": round(100 * float(np.mean(WX_SCORES >= thr))),
            "missed": len(ev), "scrubs": sum(e["kind"] == "SCRUB" for e in ev), "slips": sum(e["kind"] == "SLIP" for e in ev),
            "unexplained": unk, "unexplained_pct": round(100 * unk / len(ev)), "pads_demonstrated": 143, "pads_flown": 73,
            "model": MODEL_META["test_metrics"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (ROOT / "app" / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/stats":
                return self._send(200, stats())
            if u.path == "/api/assess":
                return self._send(200, assess(date.fromisoformat(q["date"]), q.get("time", "12:00"), q.get("site", "CCSFS"),
                                              q.get("vehicle", "falcon_9")))
            if u.path == "/api/ask":
                params = None if q.get("q") else {"date": q["date"], "time": q.get("time", "12:00"),
                                                  "site": q.get("site", "CCSFS"), "vehicle": q.get("vehicle", "falcon_9")}
                return self._send(200, officer.run(assess, qwen, question=q.get("q"), params=params))
            if u.path == "/api/memory":
                return self._send(200, memory.status())
            if u.path == "/api/qwen":
                return self._send(200, qwen(date.fromisoformat(q["date"]), q.get("time", "12:00"), q.get("site", "CCSFS"),
                                            q.get("vehicle", "falcon_9")))
            if u.path == "/api/week":
                return self._send(200, week(q.get("site", "CCSFS"), q.get("vehicle", "falcon_9")))
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/correct":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body}
        with (ROOT / "data" / "corrections.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self._send(200, {"saved": True})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print("Scrubline on http://127.0.0.1:8787")
    ThreadingHTTPServer(("127.0.0.1", 8787), Handler).serve_forever()
