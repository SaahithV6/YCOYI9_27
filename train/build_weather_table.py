"""Build the weather-risk training table: archived forecast features at each launch attempt's hour.

label 1  = a countdown scrub caused by weather at the site or aloft (known WX_LAUNCH_SITE / WX_UPPER_WINDS,
           from the operator/webcast/press record or verified web research), at the moment the count stopped.
label 0  = a launch that flew, at liftoff (every weather criterion was satisfied at that moment).
excluded = non-weather causes and recovery-zone weather (offshore; the pad forecast cannot see it).
unscored = unexplained scrubs: written separately, the trained model scores them afterwards.

Timestamps: a scrub is taken at its announcement, which is when the count stopped (as in scrubline's
backtest). Slips are left out entirely: their announcement can precede the planned T-0 by days, and
their parsed T-0 is a date with no clock, which is never used as a weather timestamp.

Features come only from the archived forecast for that site and hour (what a forecast would have
said), plus hour/month/site/vehicle. No observation enters a feature, so train matches serving.

    .venv/bin/python train/build_weather_table.py  ->  data/weather/train_table.csv, unexplained_table.csv
"""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRUB = Path.home() / "Documents" / "scrubline"
HIST = SCRUB / "cache" / "hist_forecast"
OUT = Path(__file__).resolve().parent.parent / "data" / "weather"
POS = {"WX_LAUNCH_SITE", "WX_UPPER_WINDS"}
LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 50]
SURFACE = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover_low", "cloud_cover_mid",
           "cloud_cover_high", "visibility", "wind_speed_10m", "wind_gusts_10m", "cape", "lifted_index",
           "freezing_level_height"]
WINDOW = ["precipitation", "cape", "wind_gusts_10m", "cloud_cover_low"]  # max over T-2h..T+2h


def site_of(pad: str) -> str:
    return "KSC" if "39A" in pad or "39B" in pad else "CCSFS"


def vehicle_family(v: str) -> str:
    v = v.lower()
    for fam in ("falcon heavy", "falcon 9", "atlas v", "vulcan", "new glenn", "sls", "delta iv", "terran", "starship"):
        if fam in v:
            return fam.replace(" ", "_")
    return "other"


class Hours:
    """Hourly archived forecast for one site, indexed by UTC hour."""

    def __init__(self, site=None, hourly=None):
        """Archived files for `site`, or one in-memory Open-Meteo `hourly` dict (live forecast at serving)."""
        self.idx, self.cols = {}, defaultdict(list)
        sources = [hourly] if hourly is not None else [json.loads(f.read_text()) for f in
                                                          sorted(HIST.glob(f"{site}_*.json")) + sorted((HIST / "days").glob(f"{site}_*.json"))]
        for h in sources:
            base = len(self.cols["time"])
            for i, t in enumerate(h["time"]):
                self.idx[t] = base + i
            for k, v in h.items():
                self.cols[k].extend(v)

    def at(self, t: datetime):
        return self.idx.get(t.strftime("%Y-%m-%dT%H:00"))

    def features(self, t: datetime):
        i = self.at(t)
        if i is None:
            return None
        c = self.cols
        f = {k: c[k][i] for k in SURFACE}
        for k in WINDOW:
            vals = [c[k][j] for j in range(max(0, i - 2), min(len(c[k]), i + 3)) if c[k][j] is not None]
            f[f"{k}_max5h"] = max(vals) if vals else None
        shears, prev = [], None
        for p in LEVELS:
            s, d, z = c[f"wind_speed_{p}hPa"][i], c[f"wind_direction_{p}hPa"][i], c[f"geopotential_height_{p}hPa"][i]
            if None in (s, d, z):
                prev = None
                continue
            u, v = -s * math.sin(math.radians(d)), -s * math.cos(math.radians(d))
            if prev and z > prev[2]:
                shears.append(math.hypot(u - prev[0], v - prev[1]) / ((z - prev[2]) / 1000))
            prev = (u, v, z)
        f["max_shear_kt_per_km"] = max(shears) if shears else None
        f["wind_250hPa_kt"] = c["wind_speed_250hPa"][i]
        f["hour_sin"], f["hour_cos"] = math.sin(2 * math.pi * t.hour / 24), math.cos(2 * math.pi * t.hour / 24)
        f["month_sin"], f["month_cos"] = math.sin(2 * math.pi * t.month / 12), math.cos(2 * math.pi * t.month / 12)
        return f


def main():
    events = json.loads((SCRUB / "artifacts" / "all_events.json").read_text())
    launches = json.loads((SCRUB / "cache" / "launches.json").read_text())
    web = {}
    vpath = SCRUB / "artifacts" / "research" / "verified.json"
    if vpath.exists():
        for r in json.loads(vpath.read_text()):
            if r["accepted"]:
                web[r["event_index"]] = r.get("cause_override") or r["cause"]
    hours = {s: Hours(s) for s in ("CCSFS", "KSC")}

    rows, unexplained, skipped = [], [], 0
    for i, e in enumerate(events):
        if e["kind"] != "SCRUB":
            continue
        cause = web.get(i, e["category"])
        t = datetime.fromisoformat(f"{e['date']}T{e.get('time') or '00:00'}").replace(tzinfo=timezone.utc)
        site = site_of(e["pad"])
        f = hours[site].features(t)
        if f is None:
            skipped += 1
            continue
        base = {"event_index": i, "mission": e["mission"], "time_utc": t.isoformat(), "site": site,
                "vehicle": vehicle_family(e["vehicle"]), "kind": e["kind"], "cause": cause, **f}
        if cause == "UNKNOWN":
            unexplained.append(base)
        elif cause in POS:
            rows.append({**base, "label": 1})
    for l in launches:
        if (l.get("status") or {}).get("abbrev") not in ("Success", "Failure", "Partial Failure") or l["net"] < "2021":
            continue
        t = datetime.fromisoformat(l["net"].replace("Z", "+00:00"))
        pad = (l.get("pad") or {}).get("name", "")
        site = site_of(pad)
        f = hours[site].features(t)
        if f is None:
            skipped += 1
            continue
        rows.append({"event_index": "", "mission": l["name"].split("|")[-1].strip(), "time_utc": t.isoformat(),
                     "site": site, "vehicle": vehicle_family(l["name"]), "kind": "FLEW", "cause": "", **f, "label": 0})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("train_table.csv", rows), ("unexplained_table.csv", unexplained)):
        cols = list(data[0].keys())
        with (OUT / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    pos = sum(r["label"] for r in rows)
    print(f"train table: {len(rows)} rows ({pos} weather-stopped, {len(rows) - pos} flew); "
          f"unexplained: {len(unexplained)}; skipped (no forecast hour): {skipped}")


if __name__ == "__main__":
    main()
