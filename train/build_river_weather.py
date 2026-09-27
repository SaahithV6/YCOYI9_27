"""Turn the weather training table into chat JSONL for a River LoRA fine-tune.

Same rows, same time split as the LightGBM model (train < 2025-01-01 <= test), so the two are compared
on identical launches. Input: the archived forecast at T-0 in plain words. Output: {"weather_stop": "yes"|"no"}.

    .venv/bin/python train/build_river_weather.py  ->  data/weather/river_train.jsonl, river_test.jsonl
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "weather"
SYSTEM = ("You are a launch weather officer at Cape Canaveral / Kennedy. Given the model forecast for the launch hour, "
          "answer whether weather will stop this launch attempt. Reply with JSON {\"weather_stop\": \"yes\"} or "
          "{\"weather_stop\": \"no\"}.")


def describe(r):
    t = pd.Timestamp(r.time_utc)
    f = lambda v, nd=0: "n/a" if pd.isna(v) else (f"{v:.{nd}f}")  # noqa: E731
    return (f"Site: {r.site} | Vehicle: {r.vehicle.replace('_', ' ')} | Month: {t.month} | Hour UTC: {t.hour}\n"
            f"Forecast at T-0: temperature {f(r.temperature_2m, 1)} C, humidity {f(r.relative_humidity_2m)}%, "
            f"rain {f(r.precipitation, 1)} mm (peak +/-2h {f(r.precipitation_max5h, 1)} mm), "
            f"cloud low/mid/high {f(r.cloud_cover_low)}/{f(r.cloud_cover_mid)}/{f(r.cloud_cover_high)}% (low peak +/-2h {f(r.cloud_cover_low_max5h)}%), "
            f"visibility {f(r.visibility)} m, wind {f(r.wind_speed_10m)} kt gusting {f(r.wind_gusts_10m)} kt (peak +/-2h {f(r.wind_gusts_10m_max5h)} kt), "
            f"CAPE {f(r.cape)} J/kg (peak +/-2h {f(r.cape_max5h)}), lifted index {f(r.lifted_index, 1)}, "
            f"freezing level {f(r.freezing_level_height)} m, max wind shear {f(r.max_shear_kt_per_km, 1)} kt/km, "
            f"250 hPa wind {f(r.wind_250hPa_kt)} kt.")


def row(r):
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": describe(r)},
                         {"role": "assistant", "content": json.dumps({"weather_stop": "yes" if r.label == 1 else "no"})}],
            "metadata": {"mission": r.mission, "time_utc": r.time_utc, "label": int(r.label)}}


def main():
    df = pd.read_csv(DATA / "train_table.csv")
    for name, part in (("river_train", df[df.time_utc < "2025-01-01"]), ("river_test", df[df.time_utc >= "2025-01-01"])):
        with (DATA / f"{name}.jsonl").open("w") as fh:
            for r in part.itertuples():
                fh.write(json.dumps(row(r)) + "\n")
        print(f"{name}: {len(part)} rows, {int(part.label.sum())} weather-stopped")


if __name__ == "__main__":
    main()
