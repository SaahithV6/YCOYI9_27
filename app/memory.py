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
    """Minimal MCP streamable-HTTP client: initialize once (keeps the Mcp-Session-Id), then tools/call."""
    _session = None

    def __init__(self, url, token):
        self.url, self.token, self.n = url, token, 0

    def _post(self, payload, timeout):
        headers = {"content-type": "application/json", "accept": "application/json, text/event-stream",
                   "authorization": f"Bearer {self.token}"}
        if GBrain._session:
            headers["mcp-session-id"] = GBrain._session
        resp = urllib.request.urlopen(urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers),
                                      timeout=timeout)
        GBrain._session = resp.headers.get("mcp-session-id") or GBrain._session
        return resp.read().decode()

    def _init(self, timeout):
        self._post({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "scrubline", "version": "0.1"}}}, timeout)
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout)

    def call(self, tool, args, timeout=15):
        if GBrain._session is None:
            self._init(timeout)
        self.n += 1
        raw = self._post({"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                          "params": {"name": tool, "arguments": args}}, timeout)
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


def search(query, local_search, k=6):
    """Tool body for Qwen's search_memory: GBrain page search when configured, else local keyword search."""
    g = _gbrain()
    if g is not None:
        try:
            hits = g.call("search", {"query": query, "limit": k, "snippet_chars": 300})
            rows = hits if isinstance(hits, list) else hits.get("results", [])
            return {"source": "gbrain", "hits": [{"page": h.get("slug"), "title": h.get("title"),
                                                  "text": (h.get("chunk_text") or "")[:300]} for h in rows]}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:160]
            return {"source": "local", "gbrain_error": err, "hits": local_search(query, k)}
    return {"source": "local", "hits": local_search(query, k)}


def remember(fact, provenance, entity=None):
    """Write a correction back to GBrain as a sourced fact (the learning loop). Local-only when GBrain is off."""
    g = _gbrain()
    if g is None:
        return {"saved_to": "local"}
    try:
        args = {"fact": fact, "provenance": provenance, "kind": "fact"}
        if entity:
            args["entity"] = entity
        r = g.call("remember", args)
        return {"saved_to": "gbrain", "id": (r or {}).get("id") if isinstance(r, dict) else None}
    except Exception as e:
        return {"saved_to": "local", "gbrain_error": f"{type(e).__name__}: {e}"[:200]}


def status():
    g = _gbrain()
    return {"backend": "gbrain" if g else "local", "url": os.environ.get("GBRAIN_MCP_URL", "")}
