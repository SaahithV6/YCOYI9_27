"""Fetch archived forecasts only for the days that have a scrub or a launch (2021+), per site.

Much smaller than whole years: dates within 2 days of each other are merged into one request.
Days already covered by year files in cache/hist_forecast/ are skipped.

    .venv/bin/python train/fetch_event_days.py  ->  cache/hist_forecast/days/<SITE>_<start>_<end>.json
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from build_weather_table import HIST, LEVELS, SCRUB, SURFACE, site_of
from fetch_hist_forecast import SITES, URL

DAYS = HIST / "days"
HOURLY = ",".join(SURFACE + ["weather_code"] + [f"{v}_{p}hPa" for p in LEVELS for v in ("wind_speed", "wind_direction", "geopotential_height")])


def needed():
    events = json.loads((SCRUB / "artifacts" / "all_events.json").read_text())
    launches = json.loads((SCRUB / "cache" / "launches.json").read_text())
    want = {s: set() for s in SITES}
    for e in events:
        if e["kind"] == "SCRUB" and e["date"] >= "2021":
            want[site_of(e["pad"])].add(date.fromisoformat(e["date"]))
    for l in launches:
        if (l.get("status") or {}).get("abbrev") in ("Success", "Failure", "Partial Failure") and l["net"] >= "2021":
            want[site_of((l.get("pad") or {}).get("name", ""))].add(datetime.fromisoformat(l["net"].replace("Z", "+00:00")).date())
    have_years = {(f.stem.split("_")[0], int(f.stem.split("_")[1])) for f in HIST.glob("*_*.json")}
    jobs = []
    for site, ds in want.items():
        ds = sorted(d for d in ds if (site, d.year) not in have_years and d < date.today())
        group = []
        for d in ds:
            if group and (d - group[-1]).days > 2:
                jobs.append((site, group[0] - timedelta(days=1), group[-1] + timedelta(days=1)))
                group = []
            group.append(d)
        if group:
            jobs.append((site, group[0] - timedelta(days=1), min(group[-1] + timedelta(days=1), date.today() - timedelta(days=1))))
    return jobs


def fetch(job):
    site, start, end = job
    path = DAYS / f"{site}_{start}_{end}.json"
    if path.exists():
        return "cached"
    lat, lon = SITES[site]
    url = f"{URL}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&hourly={HOURLY}&timezone=UTC&wind_speed_unit=kn"
    for attempt in range(4):
        try:
            data = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "scrubline/0.2"}), timeout=90).read())
            path.write_text(json.dumps(data["hourly"]))
            return "ok"
        except Exception as e:
            time.sleep(5 * (attempt + 1))
            err = e
    return f"fail {err}"


def main():
    DAYS.mkdir(parents=True, exist_ok=True)
    jobs = needed()
    print(f"{len(jobs)} requests", flush=True)
    t = time.time()
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(fetch, jobs))
    fails = [j for j, r in zip(jobs, results) if r.startswith("fail")]
    print(f"done in {time.time() - t:.0f}s: ok {results.count('ok')}, cached {results.count('cached')}, failed {len(fails)}")
    for j in fails[:10]:
        print("  FAIL", j)


if __name__ == "__main__":
    main()
