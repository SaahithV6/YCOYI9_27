# Cape Canaveral 7-day forecast

`forecast_week_CCSFS.json` and `forecast_week_KSC.json`: hourly, 7 days, pulled 2026-09-27 22:17 UTC.

Per hour: temperature, precipitation probability and amount, weather code, low/mid/high cloud cover,
visibility, 10 m wind / gusts / direction, CAPE, lifted index, freezing level, and 11 pressure levels
(1000-50 hPa: wind speed/direction, temperature, height) with the vector wind shear between adjacent
levels (`shear_kt_per_km`, `max_shear_kt_per_km`). Plus a 3-day NOAA SWPC Kp forecast.

This covers the three risks the 45th Weather Squadron's Probability of Violation leaves out (upper-level
wind shear, solar activity; recovery sea state is NOT included because the recovery point depends on the
mission). **Every value is a model forecast (Open-Meteo best-match NWP), tagged `source: FORECAST`, never an
observation.** No cloud layer bases/tops, so the cloud rules stay UNAVAILABLE. No shear limit is applied:
limits are vehicle-specific and unpublished.

Refresh (inside the scrubline repo): `python3 -m scrubline.upper_air CCSFS` and `... KSC`.
`upper_air.py` is the module, copied here for reference.
