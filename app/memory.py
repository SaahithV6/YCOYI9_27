"""Memory adapter: where the officer looks up past attempts and launch-rule knowledge.

Local by default (the training table + unexplained scrubs, nearest neighbours on forecast features).
Switches to GBrain when GBRAIN_MCP_URL and GBRAIN_TOKEN are set (hosted gbrain.io or a local
`gbrain serve --http`): full-text/hybrid search over the launches/, forecasts/ and concepts/ pages that
were uploaded (one page per mission with every scrub/slip, cause, evidence grade and weather).
Any GBrain error falls back to local, and the result says which source answered.
"""
import json
import os
import urllib.request


class GBrain:
    def __init__(self, url, token):
        self.url, self.token, self.n = url, token, 0

    def call(self, tool, args, timeout=15):
        self.n += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                           "params": {"name": tool, "arguments": args}}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "content-type": "application/json", "accept": "application/json, text/event-stream",
            "authorization": f"Bearer {self.token}"})
        raw = urllib.request.urlopen(req, timeout=timeout).read().decode()
        if raw.lstrip().startswith("event:") or "\ndata:" in raw:  # streamable-HTTP SSE framing
            raw = next(l[5:] for l in raw.splitlines() if l.startswith("data:"))
        msg = json.loads(raw)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        res = msg["result"]
        text = next((c.get("text") for c in res.get("content", []) if c.get("text")), "")
        return res.get("structuredContent") or (json.loads(text) if text.strip().startswith(("[", "{")) else text)


def _gbrain():
    url, token = os.environ.get("GBRAIN_MCP_URL"), os.environ.get("GBRAIN_TOKEN")
    return GBrain(url, token) if url and token else None


def precedents(local_rows, query):
    """local_rows: precedents already computed locally (nearest neighbours). query: text for GBrain search."""
    g = _gbrain()
    if g is None:
        return {"source": "local", "items": local_rows}
    try:
        hits = g.call("search", {"query": query, "limit": 6, "types": ["launch"], "snippet_chars": 280})
        items = [{"mission": h.get("title"), "slug": h.get("slug"), "snippet": (h.get("chunk_text") or "")[:280]}
                 for h in (hits if isinstance(hits, list) else hits.get("results", []))]
        return {"source": "gbrain", "items": items, "local": local_rows}
    except Exception as e:
        return {"source": "local", "items": local_rows, "gbrain_error": f"{type(e).__name__}: {e}"[:200]}


def status():
    g = _gbrain()
    return {"backend": "gbrain" if g else "local", "url": os.environ.get("GBRAIN_MCP_URL", "")}
