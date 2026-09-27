"""Seven-day upper-air and space-weather forecast for a launch site.

Fills the gaps `forecast.py` documents in the NWS gridpoint product, and covers the
three risks the 45th Weather Squadron's Probability of Violation leaves out
(LaunchFAQ.pdf: upper-level wind shear, solar activity, recovery conditions):

  * Freezing level, CAPE, and low/mid/high cloud cover, hourly.
  * Wind speed, direction, temperature and height on pressure levels 1000-50 hPa,
    and the vector wind change between adjacent levels (the shear a vehicle flies
    through). Reported as a number; no shear limit is invented here - limits are
    vehicle-specific user constraints the provider does not publish.
  * NOAA SWPC planetary Kp forecast (3 days).

WHAT IT IS NOT. Every value here is a numerical-model forecast (Open-Meteo "best match",
GFS/HRRR over Florida), never an observation, and is tagged `source: FORECAST`. There
are still no cloud layer bases or tops, so the cloud rules stay UNAVAILABLE exactly as
`forecast.py` grades them. Recovery-zone sea state is not included: the recovery point
depends on the mission's trajectory, which is not known from the site alone.

    python3 -m scrubline.upper_air CCSFS  ->  artifacts/forecast_week_CCSFS.json
"""
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone

from .sites import SITES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts")
UA = "scrubline/0.2 (research; saahith.veeramaneni@gmail.com)"
LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 50]
SURFACE = ["temperature_2m", "precipitation_probability", "precipitation", "weather_code", "cloud_cover_low",
           "cloud_cover_mid", "cloud_cover_high", "visibility", "wind_speed_10m", "wind_gusts_10m",
           "wind_direction_10m", "cape", "lifted_index", "freezing_level_height"]
PER_LEVEL = ["wind_speed", "wind_direction", "temperature", "geopotential_height"]
KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return json.loads(r.read())


def _uv(speed, direction_deg):
    rad = math.radians(direction_deg)
    return -speed * math.sin(rad), -speed * math.cos(rad)


def fetch(site):
    lat, lon = SITES[site]["lat"], SITES[site]["lon"]
    hourly = SURFACE + [f"{v}_{p}hPa" for p in LEVELS for v in PER_LEVEL]
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={','.join(hourly)}"
           f"&forecast_days=7&timezone=UTC&wind_speed_unit=kn")
    om = _get(url)
    h = om["hourly"]
    hours = []
    for i, t in enumerate(h["time"]):
        row = {"time_utc": t, "source": "FORECAST", **{k: h[k][i] for k in SURFACE}}
        levels = []
        for p in LEVELS:
            levels.append({"hPa": p, **{v: h[f"{v}_{p}hPa"][i] for v in PER_LEVEL}})
        for lo, hi in zip(levels, levels[1:]):
            if None in (lo["wind_speed"], lo["wind_direction"], hi["wind_speed"], hi["wind_direction"],
                        lo["geopotential_height"], hi["geopotential_height"]):
                hi["shear_kt_per_km"] = None
                continue
            u1, v1 = _uv(lo["wind_speed"], lo["wind_direction"])
            u2, v2 = _uv(hi["wind_speed"], hi["wind_direction"])
            dz_km = (hi["geopotential_height"] - lo["geopotential_height"]) / 1000.0
            hi["shear_kt_per_km"] = round(math.hypot(u2 - u1, v2 - v1) / dz_km, 2) if dz_km > 0 else None
        row["levels"] = levels
        valid = [l["shear_kt_per_km"] for l in levels[1:] if l.get("shear_kt_per_km") is not None]
        row["max_shear_kt_per_km"] = max(valid) if valid else None
        hours.append(row)

    kp_rows = _get(KP_URL)
    if kp_rows and isinstance(kp_rows[0], list):  # older header-row format
        kp_rows = [dict(zip(kp_rows[0], r)) for r in kp_rows[1:]]
    kp = [k for k in kp_rows if k.get("observed") in ("predicted", "estimated")]

    return {
        "site": site, "lat": lat, "lon": lon,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"weather": "Open-Meteo forecast API (best-match NWP model)", "kp": KP_URL},
        "units": {"wind": "kt", "height": "m", "temperature": "C", "shear": "kt per km (vector difference between adjacent levels)"},
        "hours": hours,
        "kp_forecast": kp,
    }


def main():
    site = sys.argv[1] if len(sys.argv) > 1 else "CCSFS"
    data = fetch(site)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"forecast_week_{site}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    hrs = data["hours"]
    print(f"{len(hrs)} forecast hours {hrs[0]['time_utc']} .. {hrs[-1]['time_utc']} -> {path}")
    print(f"{len(data['kp_forecast'])} Kp forecast periods")


if __name__ == "__main__":
    main()
