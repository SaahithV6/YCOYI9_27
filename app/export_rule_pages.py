"""Write GBrain pages with hourly launch-rule estimates for the next 7 days at CCSFS and KSC.

Uses the same assessment as the app (model risk + all 10 lightning rules estimated from the forecast
profile + forecast user constraints), one page per site per day, plus a method page.

    .venv/bin/python app/export_rule_pages.py  ->  ~/Documents/scrubline/artifacts/gbrain_rule_pages/*.md
"""
from datetime import date, timedelta
from pathlib import Path

import server  # the app module: model, fetch, assess

OUT = Path.home() / "Documents" / "scrubline" / "artifacts" / "gbrain_rule_pages"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.md"):
        old.unlink()
    for site, (name, _, _) in server.SITES.items():
        for d in range(7):
            day = date.today() + timedelta(days=d)
            rows, pulled = [], None
            for hr in range(24):
                r = server.assess(day, f"{hr:02d}:00", site, "falcon_9")
                if "error" in r:
                    continue
                pulled = r["source"]
                viol = [c["id"] + ("?" if c.get("confidence") == "low" else "") for c in r["checks"]
                        if c["status"] in ("ESTIMATED_VIOLATION", "NO_GO")]
                rows.append(f"| {hr:02d}:00 | {r['verdict']} | {r['risk']['pct_vs_flew']}% | {', '.join(viol) or '-'} | "
                            f"{'; '.join(r['cloud_layers'])} |")
            page = "\n".join([
                "---", f"title: \"{name} launch rules {day}\"", "type: forecast", "---", "",
                f"# {name} ({site}) launch-rule estimates, {day} UTC", "",
                f"From the {pulled or 'forecast'} (Open-Meteo best-match model), Falcon 9. **Estimates, not observations**: "
                "the 10 Lightning Launch Commit Criteria are estimated from the forecast cloud profile; see page "
                "\"Launch rule estimation method\". `?` = low-confidence estimate. Verdicts are never plain GO from a forecast.", "",
                "| UTC | call | risk vs flown launches | violations (estimated / forecast) | cloud layers (RH >= 90%) |",
                "|---|---|---|---|---|", *rows, "",
                "Rule ids: LLCC-1 lightning, 2/3 anvil, 4 debris, 5 disturbed weather, 6 thick cloud, 7 smoke, "
                "8 field mill, 9 triboelectrification, 10 cumulus; PRECIP, THUNDER = forecast user constraints.",
            ])
            (OUT / f"forecasts-rules-{site.lower()}-{day}.md").write_text(page + "\n")
    (OUT / "concepts-launch-rule-estimation-method.md").write_text("""---
title: "Launch rule estimation method"
type: concept
---

# Launch rule estimation method

How Scrubline estimates all ten Lightning Launch Commit Criteria (NASA-STD-4010) from a model forecast.
Every value is an ESTIMATE, graded as such, with the instrument that would replace it.

Cloud layers: contiguous pressure levels (1000-200 hPa) with relative humidity >= 90% form a layer; base and
top heights from geopotential height, top temperature from level temperature. 90% is a method parameter.

| rule | estimate from the forecast | confidence | replaced by |
|---|---|---|---|
| LLCC-1 lightning 10 nmi / 30 min | thunderstorm weather code this hour or last 3 h | medium (now) / low | GOES GLM flash positions |
| LLCC-2/3 anvil | thunderstorm code + layer topping colder than -20 C | low | radar + satellite |
| LLCC-4 debris cloud | thunderstorm in last 3 h that has ended | low | radar + visual |
| LLCC-5 disturbed weather | precipitation + layer through the freezing level | medium | radar + surface obs |
| LLCC-6 thick cloud | layer > 4500 ft thick spanning the freezing level | medium | ceilometer + cloud tops |
| LLCC-7 smoke plume | not estimable from a forecast | n/a | plume observation |
| LLCC-8 field mill > 1500 V/m | thunderstorm, or convective cloud through the freezing level | low | 45 WS field mill network |
| LLCC-9 triboelectrification | cloud layer colder than -10 C on the path, incl. cirrus | low | cloud temperature on path |
| LLCC-10 cumulus | CAPE > 0, weak inhibition, layer based below freezing level topping colder than +5 C | medium | radar echo tops + ceilometer |

Call: NO-GO if the forecast itself shows precipitation or a thunderstorm at T-0; LIKELY NO-GO on a
medium-confidence estimated violation; LIKELY GO if none and weather risk is no higher than the typical
launch that flew; otherwise UNDECIDED. Never plain GO: that needs the Space Force forecast or observations.
Thresholds (10 nmi, 4500 ft, +5 C, 1500 V/m) are from scrubline/llcc.py.
""")
    print(f"{len(list(OUT.glob('*.md')))} pages -> {OUT}")


if __name__ == "__main__":
    main()
