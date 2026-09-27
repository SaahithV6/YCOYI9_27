"""Fetch archived model forecasts (Open-Meteo historical forecast API) for CCSFS and KSC, 2021-now.

Training on archived *forecasts* (not observations) matches what the demo sees at serving time:
a forecast for a future date. Same source and variables as scrubline/upper_air.py, minus
precipitation_probability, which the archive does not keep (precipitation amount is used instead).

    .venv/bin/python train/fetch_hist_forecast.py   ->  ~/Documents/scrubline/cache/hist_forecast/<SITE>_<YEAR>.json
"""
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

SITES = {"CCSFS": (28.49, -80.57), "KSC": (28.59, -80.65)}
OUT = Path.home() / "Documents" / "scrubline" / "cache" / "hist_forecast"
LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 50]
SURFACE = ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "cloud_cover_low",
           "cloud_cover_mid", "cloud_cover_high", "visibility", "wind_speed_10m", "wind_gusts_10m",
           "wind_direction_10m", "cape", "lifted_index", "freezing_level_height"]
PER_LEVEL = ["wind_speed", "wind_direction", "geopotential_height"]
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hourly = ",".join(SURFACE + [f"{v}_{p}hPa" for p in LEVELS for v in PER_LEVEL])
    today = dt.date.today()
    for site, (lat, lon) in SITES.items():
        for year in range(2021, today.year + 1):
            path = OUT / f"{site}_{year}.json"
            if path.exists() and year < today.year:
                continue
            end = min(dt.date(year, 12, 31), today - dt.timedelta(days=1))
            url = (f"{URL}?latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={end}"
                   f"&hourly={hourly}&timezone=UTC&wind_speed_unit=kn")
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "scrubline/0.2 (research)"})
                    data = json.loads(urllib.request.urlopen(req, timeout=180).read())
                    break
                except Exception as e:  # rate limit / transient
                    print(f"  retry {site} {year}: {e}")
                    time.sleep(20 * (attempt + 1))
            else:
                raise SystemExit(f"failed {site} {year}")
            path.write_text(json.dumps(data["hourly"]))
            print(f"{site} {year}: {len(data['hourly']['time'])} hours -> {path.name}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
