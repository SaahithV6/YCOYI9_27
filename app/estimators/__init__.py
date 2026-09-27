"""Quantitative launch-rule estimators, one module per rule group (built in parallel).

estimate_all(ctx) -> {rule_id: result} for every estimator whose module is present. Each result carries
value / unit / low / high / threshold / threshold_desc / violated / confidence / basis / calibration.
A missing or failing module is skipped so the app keeps working with the rest.
"""
import importlib

GROUPS = {"convective": ["estimate_llcc1", "estimate_llcc23", "estimate_llcc4"],
          "clouds": ["estimate_llcc5", "estimate_llcc6", "estimate_llcc9", "estimate_llcc10"],
          "electric": ["estimate_llcc7", "estimate_llcc8"]}


def estimate_all(ctx):
    out, errors = {}, {}
    for mod_name, fns in GROUPS.items():
        try:
            mod = importlib.import_module(f"estimators.{mod_name}")
        except Exception as e:
            errors[mod_name] = f"{type(e).__name__}: {e}"
            continue
        for fn in fns:
            f = getattr(mod, fn, None)
            if f is None:
                continue
            try:
                r = f(ctx)
                out[r["id"]] = r
            except Exception as e:
                errors[fn] = f"{type(e).__name__}: {e}"
    return out, errors
