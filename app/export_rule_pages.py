"""Write GBrain pages with hourly launch-rule estimates for the next 7 days at CCSFS and KSC.

Uses the same assessment as the app (model risk + all 10 lightning rules estimated from the forecast
profile + forecast user constraints), one page per site per day, plus a method page.

    .venv/bin/python app/export_rule_pages.py  ->  ~/Documents/scrubline/artifacts/gbrain_rule_pages/*.md
"""
from concurrent.futures import ThreadPoolExecutor
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
            server.fetch_hourly(site, day)  # warm the cache once, then score the 24 hours in parallel
            with ThreadPoolExecutor(8) as ex:
                results = list(ex.map(lambda hr: (hr, server.assess(day, f"{hr:02d}:00", site, "falcon_9")), range(24)))
            for hr, r in results:
                if "error" in r:
                    continue
                pulled = r["source"]
                val = lambda c: ("n/a" if c.get("value") is None else f"{c['value']:g} {c.get('unit') or ''}".strip())  # noqa: E731
                cells = [f"{c['id']}{'!' if c['status'] in ('ESTIMATED_VIOLATION', 'NO_GO') else ''} {val(c)}"
                         for c in r["checks"] if c["id"].startswith("LLCC") or c["id"] in ("PRECIP", "THUNDER")]
                rows.append(f"| {hr:02d}:00 | {r['verdict']} | {r['risk']['pct_vs_flew']}% | {'; '.join(cells)} |")
            page = "\n".join([
                "---", f"title: \"{name} launch rules {day}\"", "type: forecast", "---", "",
                f"# {name} ({site}) launch-rule estimates, {day} UTC", "",
                f"From the {pulled or 'forecast'} (Open-Meteo best-match model), Falcon 9. **Estimates, not observations**: "
                "the 10 Lightning Launch Commit Criteria are estimated from the forecast cloud profile; see page "
                "\"Launch rule estimation method\". `?` = low-confidence estimate. Verdicts are never plain GO from a forecast.", "",
                "Thresholds: LLCC-1 no flash within 10 nmi (value = est. nearest flash, nmi); LLCC-5/6 cloud depth / layer "
                "thickness through the freezing level (6: 4500 ft); LLCC-8 field 1500 V/m; LLCC-9 cloud at or colder than -10 C; "
                "LLCC-10 cumulus tops colder than +5 C; PRECIP any; THUNDER WMO 95/96/99. `!` = estimated or forecast violation.", "",
                "| UTC | call | risk vs flown launches | rule values (estimated) |",
                "|---|---|---|---|", *rows, "",
                "Rule ids: LLCC-1 lightning, 2/3 anvil, 4 debris, 5 disturbed weather, 6 thick cloud, 7 smoke, "
                "8 field mill, 9 triboelectrification, 10 cumulus; PRECIP, THUNDER = forecast user constraints.",
            ])
            (OUT / f"forecasts-rules-{site.lower()}-{day}.md").write_text(page + "\n")
    (OUT / "concepts-launch-rule-estimation-method.md").write_text("""---
title: "Launch rule estimation method"
type: concept
---

# Launch rule estimation method

Scrubline estimates a physical value for all ten Lightning Launch Commit Criteria (NASA-STD-4010) from a
model forecast, with a range, the rule threshold, a confidence, and the instrument that would replace it.
Every value is an ESTIMATE. Built as three estimator modules in parallel (app/estimators/).

| rule | estimated value | threshold | calibration |
|---|---|---|---|
| LLCC-1 lightning | nearest flash distance (nmi) from a CAPE / lifted index / precipitation score + thunder code | no flash within 10 nmi / 30 min | vs GOES GLM nearest flash on 192 past attempts: flash-within-10-nmi rate 0% / 4.9% / 8.6% / 25% by score 0-3; logistic CV AUC 0.75 |
| LLCC-2/3 anvil | anvil extent downwind (nmi) = storm tops colder than -20 C x 250-300 hPa wind x 0.5-3 h | anvil within the rule distance | none (no labeled anvil events); physical estimate, low confidence |
| LLCC-4 debris cloud | minutes since the parent storm ended (exp decay, 90 min) | within 3 h of a dissipated storm | none; physical estimate, low confidence |
| LLCC-5 disturbed weather | cloud depth above the freezing level (ft) with precipitation | any depth with precipitation | not calibrated; range from pressure-level spacing |
| LLCC-6 thick cloud | thickness of the thickest layer through the freezing level (ft) | 4500 ft | not calibrated; interpolated layer bases/tops |
| LLCC-7 smoke plume | not estimable from a forecast (fire-weather context only) | cumulus from a smoke plume | n/a |
| LLCC-8 field mill | surface electric field (V/m) by regime: fair ~115, light precip ~400, electrified cloud ~2500, thunderstorm 3000-10000 | 1500 V/m | literature regimes (Chalmers 1967; MacGorman & Rust 1998; Marshall & Marsh 1993; Standler & Winn 1979); no public field-mill data after 2012 |
| LLCC-9 triboelectrification | coldest cloud temperature on the ascent path (C) | cloud at or colder than -10 C | not calibrated; advisory for the call (vehicles fly cirrus with tribo treatments) |
| LLCC-10 cumulus | convective cloud top temperature (C) from a pseudo-adiabatic parcel lift | tops colder than +5 C | vs 113 observed radar echo tops: raw parcel tops 19.4 kft too high, r = 0.18, so confidence capped at low |

Cloud layers: RH >= 90% on pressure levels, bases/tops interpolated between levels (method parameter).
Call: NO-GO if the forecast shows precipitation or a thunderstorm at T-0; LIKELY NO-GO on a medium/high-confidence
estimated violation (LLCC-9 advisory); LIKELY GO if none and weather risk is no higher than the typical launch
that flew; otherwise UNDECIDED. Never plain GO from a forecast.
""")
    print(f"{len(list(OUT.glob('*.md')))} pages -> {OUT}")


if __name__ == "__main__":
    main()
