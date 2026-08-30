"""Local web UI for Projects 2A + 3A.

One page, two tools:
  * 2A — paste a messy support note, get the cleaned structured record.
  * 3A — paste an English message + Spanish translation, get the send verdict
         (model judgement + deterministic safety gate, exactly as evaluated).

Standard library ONLY (http.server) — no Flask, no new pip installs, same
Python-only / light-dependency rules as the rest of the repo. Binds to
127.0.0.1 by default so nothing is exposed beyond this machine, and customer
text is never logged (same privacy stance as the CLIs).

Run from the project root:
    python app.py            # then open http://127.0.0.1:8765
    python app.py --port 9000
    python app.py --host 0.0.0.0   # ONLY when testers reach the box over the
                                   # network (e.g. WSL/remote Linux) — exposes
                                   # the UI to whoever can reach that interface
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src import nli_check
from src.provider_adapter import ProviderError, resolve_provider
from src.service_request_cleanup import (
    CleanupError,
    clean_service_request,
)
from src.translation_quality_checker import (
    TranslationError,
    check_translation,
)

DEFAULT_PORT = 8765


# --------------------------------------------------------------------------- #
# Request handlers (pure functions -> easy to test offline)
# --------------------------------------------------------------------------- #

def handle_cleanup(payload: dict) -> tuple[int, dict]:
    """2A: {'text': ..., 'provider'?: ..., 'model'?: ...} -> (status, body)."""
    text = (payload.get("text") or "").strip()
    model = payload.get("model") or os.getenv("CLEANUP_MODEL") or None
    started = time.perf_counter()
    try:
        result = clean_service_request(
            text, provider=payload.get("provider"), model=model
        )
    except CleanupError as exc:
        return 400, {"error": str(exc)}
    except ProviderError as exc:
        return 502, {"error": str(exc)}
    return 200, {
        "result": result.model_dump(mode="json"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def handle_check(payload: dict) -> tuple[int, dict]:
    """3A: {'english': ..., 'spanish': ...} -> (status, body)."""
    model = payload.get("model") or os.getenv("TRANSLATION_MODEL") or None
    started = time.perf_counter()
    try:
        result = check_translation(
            payload.get("english") or "",
            payload.get("spanish") or "",
            provider=payload.get("provider"),
            model=model,
        )
    except TranslationError as exc:  # blank input / not Spanish / bad JSON
        return 400, {"error": str(exc)}
    except ProviderError as exc:
        return 502, {"error": str(exc)}
    return 200, {
        "result": result.model_dump(mode="json"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def handle_info() -> tuple[int, dict]:
    """Footer info: which provider/models the UI is wired to right now."""
    fallback = os.getenv("LOCAL_MODEL") or "phi4-mini"
    return 200, {
        "provider": resolve_provider(None),
        "cleanup_model": os.getenv("CLEANUP_MODEL") or fallback,
        "translation_model": os.getenv("TRANSLATION_MODEL") or fallback,
        "semantic_check": "on" if nli_check.is_enabled() else "off",
    }


_ROUTES = {
    "/api/cleanup": handle_cleanup,
    "/api/check": handle_check,
}


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet: no per-request console noise
        pass

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Compare the path only — "/?tab=3a" (shareable demo links) is still "/".
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            data = _PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/info":
            self._send_json(*handle_info())
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self):
        handler = _ROUTES.get(self.path)
        if handler is None:
            self._send_json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Browsers always send UTF-8; this tolerates non-UTF-8 clients
                # (older curl/PowerShell) instead of rejecting accented text.
                text = raw.decode("latin-1")
            payload = json.loads(text or "{}")
        except ValueError:
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return
        try:
            self._send_json(*handler(payload))
        except Exception as exc:  # last-resort: never let the server die silently
            self._send_json(500, {"error": f"Unexpected server error: {exc}"})


# --------------------------------------------------------------------------- #
# Page (embedded so deployment stays "copy the repo, run one file")
# --------------------------------------------------------------------------- #

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Service Desk AI - 2A &amp; 3A</title>
<style>
  :root { --bg:#f6f7f9; --card:#ffffff; --ink:#1d2433; --muted:#5b6472;
          --line:#e3e7ee; --accent:#2458d6;
          --ok:#0f7b40; --ok-bg:#e7f6ec; --warn:#9a6700; --warn-bg:#fff4d6;
          --bad:#b3261e; --bad-bg:#fdecea; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui, "Segoe UI", sans-serif; }
  header { background:var(--card); border-bottom:1px solid var(--line); padding:14px 22px; }
  header h1 { margin:0; font-size:18px; }
  header p { margin:2px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:880px; margin:20px auto 60px; padding:0 16px; }
  .tabs { display:flex; gap:8px; margin-bottom:16px; }
  .tabs button { flex:1; padding:10px; font-size:15px; font-weight:600; cursor:pointer;
                 border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--muted); }
  .tabs button.active { border-color:var(--accent); color:var(--accent); background:#eef3fe; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }
  textarea { width:100%; min-height:92px; padding:10px; font:14px/1.4 inherit;
             border:1px solid var(--line); border-radius:8px; resize:vertical; }
  textarea:focus { outline:2px solid var(--accent); border-color:transparent; }
  label { display:block; font-weight:600; font-size:13px; margin:12px 0 4px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:14px; flex-wrap:wrap; }
  .go { background:var(--accent); color:#fff; border:0; border-radius:8px;
        padding:10px 22px; font-size:15px; font-weight:600; cursor:pointer; }
  .go:disabled { opacity:.5; cursor:wait; }
  .examples { display:flex; gap:6px; flex-wrap:wrap; }
  .examples button { border:1px solid var(--line); background:var(--bg); border-radius:20px;
                     padding:4px 12px; font-size:12px; cursor:pointer; color:var(--muted); }
  .examples button:hover { border-color:var(--accent); color:var(--accent); }
  .elapsed { color:var(--muted); font-size:13px; margin-left:auto; }
  .verdict { border-radius:8px; padding:14px 16px; font-size:17px; font-weight:700; margin-top:18px; }
  .v-send { background:var(--ok-bg); color:var(--ok); }
  .v-review { background:var(--warn-bg); color:var(--warn); }
  .v-block { background:var(--bad-bg); color:var(--bad); }
  .v-error { background:var(--bad-bg); color:var(--bad); font-weight:600; font-size:14px; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .chip { border:1px solid var(--line); border-radius:20px; padding:3px 12px; font-size:12.5px; color:var(--muted); }
  dl { margin:14px 0 0; }
  dt { font-weight:600; font-size:13px; margin-top:12px; color:var(--muted); }
  dd { margin:2px 0 0; }
  dd ul { margin:4px 0 0; padding-left:20px; }
  .gate { background:var(--bad-bg); border-radius:6px; padding:1px 6px; }
  details { margin-top:16px; }
  summary { cursor:pointer; color:var(--muted); font-size:13px; }
  pre { background:var(--bg); border:1px solid var(--line); border-radius:8px;
        padding:12px; overflow-x:auto; font-size:12.5px; }
  footer { text-align:center; color:var(--muted); font-size:12.5px; margin-top:26px; }
  .hidden { display:none; }
</style>
</head>
<body>
<header>
  <h1>Service Desk AI</h1>
  <p>2A: clean a messy note &nbsp;&middot;&nbsp; 3A: is this Spanish translation safe to send?</p>
</header>
<main>
  <div class="tabs">
    <button id="tab2a" class="active" onclick="showTab('2a')">2A &mdash; Clean a note</button>
    <button id="tab3a" onclick="showTab('3a')">3A &mdash; Check a translation</button>
  </div>

  <section id="panel2a" class="card">
    <label for="note">Messy support note</label>
    <textarea id="note" placeholder="cust says cant login been trying since morning..."></textarea>
    <div class="row">
      <button class="go" id="go2a" onclick="runCleanup()">Clean it</button>
      <div class="examples">
        <button onclick="ex2a(0)">example: login</button>
        <button onclick="ex2a(1)">example: angry refund</button>
        <button onclick="ex2a(2)">example: API bug</button>
      </div>
      <span class="elapsed" id="t2a"></span>
    </div>
    <div id="out2a"></div>
  </section>

  <section id="panel3a" class="card hidden">
    <label for="en">English original</label>
    <textarea id="en" placeholder="We are sorry your order was delayed..."></textarea>
    <label for="es">Spanish translation</label>
    <textarea id="es" placeholder="Lamentamos que su pedido se haya retrasado..."></textarea>
    <div class="row">
      <button class="go" id="go3a" onclick="runCheck()">Check it</button>
      <div class="examples">
        <button onclick="ex3a(0)">example: good translation</button>
        <button onclick="ex3a(1)">example: added promise</button>
        <button onclick="ex3a(2)">example: wrong amount</button>
      </div>
      <span class="elapsed" id="t3a"></span>
    </div>
    <div id="out3a"></div>
  </section>

  <footer id="info"></footer>
</main>
<script>
const EX2A = [
  "cust says cant login been trying since morning says password reset not working maybe email wrong needs help asap very upset",
  "this is the THIRD time im writing in and nobody has fixed my refund its been 2 weeks i am absolutely furious you people are useless just give me my money back",
  "checkout api returning 500 on POST /orders when promo code applied, looks like null pointer in DiscountService.calc, only happens for orders over $200, added temp logging",
];
const EX3A = [
  ["We are sorry your order was delayed. Please send us your order number so we can review the issue.",
   "Lamentamos que su pedido se haya retrasado. Envíenos su número de pedido para que podamos revisar el problema."],
  ["Thank you for contacting us. We will look into your billing question and follow up.",
   "Gracias por contactarnos. Le garantizamos un reembolso completo de inmediato."],
  ["Your refund of $45 has been processed and will appear within 5 business days.",
   "Su reembolso de $54 ha sido procesado y aparecerá en un plazo de 5 días hábiles."],
];
function $(id) { return document.getElementById(id); }
function showTab(t) {
  $("panel2a").classList.toggle("hidden", t !== "2a");
  $("panel3a").classList.toggle("hidden", t !== "3a");
  $("tab2a").classList.toggle("active", t === "2a");
  $("tab3a").classList.toggle("active", t === "3a");
}
function ex2a(i) { $("note").value = EX2A[i]; }
function ex3a(i) { $("en").value = EX3A[i][0]; $("es").value = EX3A[i][1]; }
function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }
function list(items) {
  if (!items || !items.length) return "<dd>(none)</dd>";
  return "<dd><ul>" + items.map(x => {
    const h = esc(x);
    return "<li>" + (x.startsWith("[safety gate]") ? '<span class="gate">' + h + "</span>" : h) + "</li>";
  }).join("") + "</ul></dd>";
}
async function post(url, body) {
  const resp = await fetch(url, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || ("HTTP " + resp.status));
  return data;
}
async function runCleanup() {
  const btn = $("go2a"), out = $("out2a"), t = $("t2a");
  btn.disabled = true; t.textContent = "working..."; out.innerHTML = "";
  try {
    const data = await post("/api/cleanup", { text: $("note").value });
    const r = data.result;
    t.textContent = data.elapsed_seconds + "s";
    out.innerHTML =
      '<div class="verdict v-send">' + esc(r.short_summary) + "</div>" +
      '<div class="chips"><span class="chip">Type: ' + esc(r.issue_type) + "</span>" +
      '<span class="chip">Confidence: ' + r.confidence_score + "/100</span></div>" +
      "<dl>" +
      "<dt>Cleaned description</dt><dd>" + esc(r.cleaned_description) + "</dd>" +
      "<dt>Customer impact</dt><dd>" + esc(r.customer_impact) + "</dd>" +
      "<dt>Technical details</dt>" + list(r.technical_details) +
      "<dt>Missing information</dt>" + list(r.missing_information) +
      "<dt>Suggested next step (for the agent)</dt><dd>" + esc(r.suggested_next_step) + "</dd>" +
      "</dl>" +
      "<details><summary>Full JSON</summary><pre>" + esc(JSON.stringify(r, null, 2)) + "</pre></details>";
  } catch (e) {
    t.textContent = "";
    out.innerHTML = '<div class="verdict v-error">' + esc(e.message) + "</div>";
  } finally { btn.disabled = false; }
}
async function runCheck() {
  const btn = $("go3a"), out = $("out3a"), t = $("t3a");
  btn.disabled = true; t.textContent = "working..."; out.innerHTML = "";
  try {
    const data = await post("/api/check", { english: $("en").value, spanish: $("es").value });
    const r = data.result;
    t.textContent = data.elapsed_seconds + "s";
    const cls = { "Send": "v-send", "Review First": "v-review", "Do Not Send": "v-block" }[r.send_recommendation] || "v-review";
    out.innerHTML =
      '<div class="verdict ' + cls + '">' + esc(r.send_recommendation).toUpperCase() +
      " &mdash; " + esc(r.short_summary) + "</div>" +
      '<div class="chips"><span class="chip">Accuracy: ' + esc(r.accuracy_rating) + "</span>" +
      '<span class="chip">Tone: ' + esc(r.tone_check) + "</span>" +
      '<span class="chip">Confidence: ' + r.confidence_score + "/100</span></div>" +
      "<dl>" +
      "<dt>Risky phrases</dt>" + list(r.risky_phrases) +
      "<dt>Missing meaning</dt>" + list(r.missing_meaning) +
      "<dt>Added meaning</dt>" + list(r.added_meaning) +
      "<dt>Back-translation (what the Spanish actually says)</dt><dd>" + esc(r.back_translation) + "</dd>" +
      (r.suggested_correction
        ? "<dt>Suggested correction</dt><dd>" + esc(r.suggested_correction) + "</dd>" : "") +
      "<dt>Explanation</dt><dd>" + esc(r.explanation) + "</dd>" +
      "</dl>" +
      "<details><summary>Full JSON</summary><pre>" + esc(JSON.stringify(r, null, 2)) + "</pre></details>";
  } catch (e) {
    t.textContent = "";
    out.innerHTML = '<div class="verdict v-error">' + esc(e.message) + "</div>";
  } finally { btn.disabled = false; }
}
fetch("/api/info").then(r => r.json()).then(i => {
  $("info").textContent = "provider: " + i.provider +
    "  |  2A model: " + i.cleanup_model + "  |  3A model: " + i.translation_model +
    " + safety gate + semantic check: " + i.semantic_check;
});
document.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") {
    if (!$("panel2a").classList.contains("hidden")) runCleanup(); else runCheck();
  }
});
// Shareable demo links: ?tab=3a&en=...&es=...&run=1 pre-fills the form and
// (with run=1) submits it on load. Also used to capture doc screenshots.
const q = new URLSearchParams(location.search);
if (q.get("tab") === "3a") showTab("3a");
if (q.get("text")) $("note").value = q.get("text");
if (q.get("en")) $("en").value = q.get("en");
if (q.get("es")) $("es").value = q.get("es");
if (q.get("run")) {
  if (!$("panel2a").classList.contains("hidden")) runCleanup(); else runCheck();
}
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="app", description="Local web UI for Projects 2A + 3A."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default 127.0.0.1 = this machine only; "
        "0.0.0.0 to allow access from other machines — use only on a trusted network)",
    )
    args = parser.parse_args(argv)

    try:  # load .env if python-dotenv is available; harmless if not
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    server = ThreadingHTTPServer((args.host, args.port), _Handler)

    # Preload the optional NLI model in the background so the first 3A click
    # doesn't pay its ~10s load. No-op when the layer is off/unavailable.
    if nli_check.is_enabled():
        threading.Thread(target=nli_check.warm_up, daemon=True).start()

    info = handle_info()[1]
    shown_host = "127.0.0.1" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print("Service Desk AI - local UI")
    print(f"  2A model: {info['cleanup_model']}   3A model: {info['translation_model']}")
    print(f"  Semantic check (NLI): {info['semantic_check']}")
    print(f"  Open:  http://{shown_host}:{args.port}")
    if args.host == "0.0.0.0":
        print("  NOTE: bound to all interfaces - reachable from other machines.")
    print("  Stop:  Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
