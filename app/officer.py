"""The launch weather officer: Qwen-driven control flow over the tools.

    question -> [Qwen: parse] -> in parallel: [assess: forecast + rule estimates + LightGBM]
                                              [fine-tuned Qwen (River checkpoint): weather stop?]
                                              [memory: precedents from GBrain or local]
             -> [gate: which calls the evidence allows] -> [Qwen: write the call + reasons] -> answer

The gate is deterministic and cannot be overridden: forecast rain/thunder at T-0 forces NO-GO, and a
forecast-only call is never plain GO. Qwen chooses within the allowed calls and must cite the tools.
Every step is timed and returned as a trace for the UI.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import memory

BASE_MODEL = "Qwen/Qwen3.5-9B"
CALLS = ["NO-GO", "LIKELY NO-GO", "UNDECIDED", "LIKELY GO"]


def _client():
    import river_client as river
    return river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")


def _chat(messages, max_tokens=400, tools=None):
    kw = {"tools": tools, "tool_choice": "auto"} if tools else {}
    r = _client().chat_complete(messages, base_model=BASE_MODEL, max_tokens=max_tokens, temperature=0.0, timeout=60,
                                chat_template_kwargs={"enable_thinking": False}, **kw)
    body = r.response_json if isinstance(r.response_json, dict) else json.loads(r.response_json)
    msg = body["choices"][0]["message"]
    return msg if tools else (msg.get("content") or "")


SEARCH_TOOL = [{"type": "function", "function": {
    "name": "search_memory",
    "description": "Search Scrubline memory (GBrain): one page per past Cape/KSC mission with every scrub and slip, "
                   "its cause, evidence grade and the weather at the pad; plus launch-rule and forecast pages. "
                   "Returns matching pages with a text snippet. Use it to find precedents before making the call.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "keywords, e.g. 'KSC cumulus scrub October'"}},
                   "required": ["query"]}}}]


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def parse(question):
    today = date.today().isoformat()
    text = _chat([{"role": "system", "content":
                   "Extract launch parameters from the user's question. Reply only JSON "
                   '{"date":"YYYY-MM-DD","time":"HH:MM","site":"CCSFS|KSC","vehicle":"falcon_9|falcon_heavy|atlas_v|vulcan|new_glenn|sls"}. '
                   f"Today is {today} (UTC). Times are UTC unless stated; EDT = UTC-4. SLC-40, SLC-41, SLC-37 are CCSFS; "
                   "LC-39A and LC-39B are KSC. Default vehicle falcon_9, default site CCSFS, default time 12:00."},
                  {"role": "user", "content": question}], max_tokens=120)
    p = _json(text) or {}
    return {"date": p.get("date", today), "time": p.get("time", "12:00"), "site": p.get("site", "CCSFS"),
            "vehicle": p.get("vehicle", "falcon_9")}


def gate(a):
    """Calls the evidence allows. Qwen picks within these; it cannot move the call across the deterministic line."""
    # The call is the deterministic one (rules + forecast + risk). Qwen explains it, searches memory and suggests
    # better windows; it cannot move the call in either direction.
    return [a["verdict"]]


def narrate(a, qw, mem, allowed, local_search=None, trace=None, alts=None):
    viol = [f"{c['id']} {c['rule']}: {c.get('value')} {c.get('unit') or ''} vs limit {c.get('threshold_desc') or c.get('threshold')} "
            f"({c.get('confidence')} confidence{', advisory' if c['id'] == 'LLCC-9' else ''})"
            for c in a["checks"] if c["status"] in ("ESTIMATED_VIOLATION", "NO_GO")]
    evidence = {
        "mission": a["input"], "forecast_source": a["source"],
        "lightgbm_risk": (f"This hour's weather risk is higher than {a['risk']['pct_vs_flew']}% of past launches that flew, "
                          f"and higher than {a['risk']['pct_vs_weather_scrubs']}% of past weather scrubs."),
        "finetuned_qwen_weather_stop": qw.get("weather_stop") if qw.get("available") else "unavailable",
        "rule_violations": viol or ["none estimated"],
        "top_risk_factors": [f"{f['name']} {'+' if f['push'] > 0 else ''}{f['push']}" for f in a["factors"][:4]],
        "cloud_layers": a.get("cloud_layers"), "freezing_level_ft": a.get("freezing_level_ft"),
        "precedents": [p.get("mission", "") + (f" ({p['outcome']})" if p.get("outcome") else "") for p in mem["items"][:5]],
        "precedent_source": mem["source"],
        "better_windows": [f"{x['date']} {x['time']} UTC ({x['shift_hours']:+d} h): {x['call']}, risk higher than "
                           f"{x['pct_vs_flew']}% of flown, {x['rules_clear']}/{x['rules_total']} rules clear"
                           for x in (alts or {}).get("alternatives", [])[:3]],
    }
    messages = [{"role": "system", "content":
                   "You are the launch weather officer for Cape Canaveral / Kennedy. First call search_memory (at most 3 times) to "
                   "find past attempts like this one, then, using ONLY the evidence JSON and the memory results, make the "
                   f"weather call. The call MUST be one of {allowed}. Final reply: only JSON "
                   '{"call": "...", "headline": "one sentence", "reasons": ["3 short reasons, each citing a tool: rules, LightGBM, fine-tuned Qwen, precedents"], '
                   '"watch": ["what would change the call"], "suggest": "if the call is not LIKELY GO, recommend the best of better_windows in one sentence"}. Quote numbers exactly as the evidence states them; '
                   'never state a threshold, limit or number that is not in the evidence. "watch" items must name conditions, not new numbers.'},
                  {"role": "user", "content": json.dumps(evidence)}]
    text, searches = "", []
    for _ in range(4):  # tool loop: Qwen decides what to look up; our code executes the search
        msg = _chat(messages, max_tokens=500, tools=SEARCH_TOOL if len(searches) < 3 else None)
        if isinstance(msg, str):
            text = msg
            break
        calls = msg.get("tool_calls") or []
        if not calls:
            text = msg.get("content") or ""
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            s = time.time()
            q = json.loads(c["function"].get("arguments") or "{}").get("query", "")
            res = memory.search(q, local_search)
            searches.append({"query": q, "source": res["source"], "hits": len(res["hits"])})
            if trace is not None:
                trace.append({"step": f"Qwen → search_memory(\"{q[:60]}\") · {res['source']} · {len(res['hits'])} hits",
                              "ok": True, "ms": round(1000 * (time.time() - s)), "note": res.get("gbrain_error", "")})
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(res)[:3500]})
    evidence["memory_searches"] = searches
    out = _json(text) or {}
    if out.get("call") not in allowed:  # the gate wins over the model
        out["call_overridden"] = out.get("call")
        out["call"] = a["verdict"] if a["verdict"] in allowed else allowed[0]
    return out, evidence


def run(assess_fn, qwen_fn, question=None, params=None, local_search=None, alternatives_fn=None):
    trace, t0 = [], time.time()

    def step(name, fn, *args):
        s = time.time()
        try:
            out, ok = fn(*args), True
        except Exception as e:
            out, ok = {"error": f"{type(e).__name__}: {e}"[:300]}, False
        trace.append({"step": name, "ok": ok, "ms": round(1000 * (time.time() - s))})
        return out

    if question:
        params = step("Qwen · parse question", parse, question)
    d = date.fromisoformat(params["date"])
    args = (d, params["time"], params["site"], params["vehicle"])

    s = time.time()
    with ThreadPoolExecutor(4) as ex:
        fa = ex.submit(assess_fn, *args)
        fq = ex.submit(qwen_fn, *args)
        falt = ex.submit(alternatives_fn, *args) if alternatives_fn else None
        a = fa.result()
        if "error" in a:
            return {"params": params, "error": a["error"], "trace": trace}
        q = f"{params['vehicle'].replace('_', ' ')} {params['site']} weather scrub " + " ".join(
            c["rule"] for c in a["checks"] if c["status"] in ("ESTIMATED_VIOLATION", "NO_GO"))[:200]
        fm = ex.submit(memory.precedents, a["precedents"], q)
        qw, mem = fq.result(), fm.result()
        alts = falt.result() if falt else {"alternatives": []}
    par_ms = round(1000 * (time.time() - s))
    trace += [{"step": "Forecast + rule estimates + LightGBM", "ok": True, "ms": par_ms, "parallel": True},
              {"step": "Fine-tuned Qwen (River checkpoint)", "ok": qw.get("available", False), "ms": par_ms, "parallel": True,
               "note": qw.get("reason", "")},
              {"step": f"Memory · precedents ({mem['source']})", "ok": True, "ms": par_ms, "parallel": True,
               "note": mem.get("gbrain_error", "")},
              {"step": f"Better windows · {len(alts['alternatives'])} LIKELY GO hours with full rule checks", "ok": True,
               "ms": par_ms, "parallel": True}]

    allowed = step("Gate · allowed calls", gate, a)
    try:
        s = time.time()
        brief, evidence = narrate(a, qw, mem, allowed, local_search, trace, alts)
        trace.append({"step": "Qwen · write the call", "ok": True, "ms": round(1000 * (time.time() - s))})
    except Exception as e:
        brief, evidence = {"call": a["verdict"], "headline": a["verdict_reason"], "reasons": [], "watch": []}, {}
        trace.append({"step": "Qwen · write the call", "ok": False, "ms": 0, "note": f"{type(e).__name__}: {e}"[:200]})
    return {"params": params, "assessment": a, "qwen_classifier": qw, "memory": mem, "allowed_calls": allowed,
            "alternatives": alts.get("alternatives", []),
            "brief": brief, "evidence": evidence, "trace": trace, "total_ms": round(1000 * (time.time() - t0)),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
