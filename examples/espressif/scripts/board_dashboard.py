# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Live Chrome dashboard for on-board runs of examples/espressif/espnn_runner.

Used by benchmark.py --dashboard; can also show a saved results file:
    python examples/espressif/scripts/board_dashboard.py --results benchmarks/esp32s3.json
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import threading
import time
import webbrowser
from pathlib import Path


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.version = 0
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.logs: list[str] = []
        self.running = False

    def touch(self):
        self.version += 1
        self.changed.notify_all()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "jobs": self.jobs,
                "logs": self.logs[-200:],
                "running": self.running,
                "version": self.version,
                "order": list(self.order),
            }


class Dashboard:
    """Thread-safe facade that benchmark.py calls while a run progresses."""

    def __init__(self, state: State, url: str):
        self.state = state
        self.url = url
        with state.lock:
            state.running = True
            state.touch()

    def begin_job(self, job: str, title: str, cases: dict) -> None:
        with self.state.lock:
            self.state.jobs[job] = {
                "title": title,
                "status": "queued",
                "cases": cases,
                "result_line": "",
            }
            if job not in self.state.order:
                self.state.order.append(job)
            self.state.touch()

    def status(self, job: str, status: str) -> None:
        with self.state.lock:
            self.state.jobs.setdefault(job, {"cases": {}})["status"] = status
            self.state.touch()

    def case(self, job: str, fields: dict) -> None:
        with self.state.lock:
            cases = self.state.jobs.setdefault(job, {"cases": {}}).setdefault(
                "cases", {}
            )
            cases.setdefault(fields.get("name", "?"), {})["device"] = fields
            self.state.touch()

    def log(self, line: str) -> None:
        if not line.startswith(
            ("SUITE", "ET ", "CHIP", "ESPNN", "ERROR", "Guru", "abort")
        ):
            return
        with self.state.lock:
            self.state.logs.append(line[:240])
            del self.state.logs[:-400]
            self.state.touch()

    def finish(self) -> None:
        with self.state.lock:
            self.state.running = False
            self.state.touch()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP-NN Board Dashboard</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --text:#1d2330; --muted:#5f6b7d; --line:#dde2ea;
        --accent:#2563eb; --good:#15803d; --good-bg:#dcfce7; --bad:#b91c1c; --bad-bg:#fee2e2;
        --warn:#b45309; --warn-bg:#fef3c7; --chip:#eef2f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0f1218; --panel:#171b23; --text:#e7ebf2; --muted:#98a2b3; --line:#2a303c; --accent:#6ea0ff;
  --good:#4ade80; --good-bg:#12301f; --bad:#f87171; --bad-bg:#3a1414; --warn:#fbbf24; --warn-bg:#3a2a0b; --chip:#222835; } }
:root[data-theme="dark"] { --bg:#0f1218; --panel:#171b23; --text:#e7ebf2; --muted:#98a2b3; --line:#2a303c;
  --accent:#6ea0ff; --good:#4ade80; --good-bg:#12301f; --bad:#f87171; --bad-bg:#3a1414; --warn:#fbbf24; --warn-bg:#3a2a0b; --chip:#222835; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
header { display:flex; align-items:center; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; font-weight:600; }
#summary { margin-left:auto; color:var(--muted); font-size:13px; }
main { max-width:1280px; margin:0 auto; padding:16px; display:flex; flex-direction:column; gap:16px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; overflow:hidden; }
.panel h2 { font-size:15px; margin:0 0 4px; }
.sub { color:var(--muted); font-size:12px; margin-bottom:10px; }
.chip { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600; background:var(--chip); color:var(--muted); }
.PASS { background:var(--good-bg); color:var(--good); } .FAIL { background:var(--bad-bg); color:var(--bad); }
.busy { background:var(--warn-bg); color:var(--warn); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th { text-align:left; color:var(--muted); font-weight:600; font-size:12px; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
details summary { cursor:pointer; color:var(--accent); }
.ops { color:var(--muted); font-size:12px; max-width:420px; }
.bar { height:6px; border-radius:3px; background:var(--chip); overflow:hidden; min-width:90px; display:flex; }
.bar span { display:block; height:100%; }
.bar .d { background:var(--accent); } .bar .c { background:var(--warn); }
pre { font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); white-space:pre-wrap; max-height:260px; overflow:auto; margin:0; }
.legend { font-size:12px; color:var(--muted); display:flex; gap:14px; margin-top:8px; flex-wrap:wrap; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }
</style></head>
<body>
<header><h1>ESP-NN board dashboard</h1><span class="chip" id="state">idle</span><span id="summary"></span></header>
<main id="jobs"></main>
<main><section class="panel"><h2>Live log</h2><pre id="log"></pre></section></main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function chip(status) {
  const cls = status === "PASS" ? "PASS" : (status === "FAIL" || /failed/.test(status||"") ? "FAIL" : "busy");
  return `<span class="chip ${cls}">${esc(status || "pending")}</span>`;
}
function ms(us) { return us === undefined ? "" : (Number(us) / 1000).toFixed(Number(us) < 10000 ? 2 : 0); }
function accuracy(d) {
  if (!d) return "";
  if (d.exact !== undefined) return `${d.exact}/${d.samples} bit-exact${d.ansi === "1" ? ", ANSI too" : ""}`;
  if (d.exact_required === "1") return d.mismatched === "0" ? "bit-exact" : `${d.mismatched} mismatches`;
  const s = Number(d.sqnr_db);
  return s >= 999 ? "bit-exact vs host" : `${s.toFixed(1)} dB vs host (min ${d.min_sqnr})`;
}
function row(name, c) {
  const d = c.device || {}, h = c.host || {};
  const total = Number(d.us || 0), del = Number(d.delegate_us || 0), cpu = Number(d.cpu_us || 0);
  const delPct = total ? Math.min(100, 100 * del / total) : 0, cpuPct = total ? Math.min(100 - delPct, 100 * cpu / total) : 0;
  const cpuOps = h.cpu_ops || [];
  const groups = {};
  cpuOps.forEach(o => { groups[o.target] = (groups[o.target] || 0) + 1; });
  const opsText = Object.entries(groups).sort((a, b) => b[1] - a[1]).map(([t, n]) => `${esc(t)} ×${n}`).join(", ");
  const reasons = [...new Set(cpuOps.map(o => o.reason))].slice(0, 4).map(esc).join("<br>");
  const opsCell = cpuOps.length ? `<details><summary>${cpuOps.length} on CPU</summary><div class="ops">${opsText}<br><br>${reasons}</div></details>` : (h.cpu_ops ? "none" : "");
  const io = h.inputs ? `${h.inputs.map(s => s.join("×")).join(", ")} → ${h.outputs.map(s => s.join("×")).join(", ")}` : `${d.inputs||""} in / ${d.outputs||""} out`;
  const opsInfo = h.operations ? Object.entries(h.operations).map(([k, v]) => `${k} ${v}`).join(", ") : (h.delegated_ops !== undefined ? `${h.delegated_ops} delegated in ${h.partitions} partition(s)` : "");
  return `<tr><td><b>${esc(name)}</b><div class="ops">${esc(io)}</div></td>
    <td class="ops">${esc(opsInfo)}</td><td>${opsCell}</td>
    <td class="num">${ms(d.us)}</td>
    <td>${total && d.delegate_us !== undefined ? `<div class="bar"><span class="d" style="width:${delPct}%"></span><span class="c" style="width:${cpuPct}%"></span></div><div class="ops">${ms(d.delegate_us)} / ${ms(d.cpu_us)} ms</div>` : ""}</td>
    <td>${esc(accuracy(d))}${h.host_sqnr_vs_pytorch_db !== undefined ? `<div class="ops">host vs PyTorch ${h.host_sqnr_vs_pytorch_db} dB</div>` : ""}${h.accuracy ? `<div class="ops">${esc(h.accuracy)}</div>` : ""}</td>
    <td>${chip(d.status || (Object.keys(d).length ? "?" : "pending"))}</td></tr>`;
}
function render(s) {
  const order = (s.order || Object.keys(s.jobs)).filter(n => s.jobs[n]);
  let pass = 0, total = 0;
  $("jobs").innerHTML = order.map(n => {
    const j = s.jobs[n], cases = j.cases || {};
    const names = Object.keys(cases);
    names.forEach(c => { total++; if ((cases[c].device || {}).status === "PASS") pass++; });
    return `<section class="panel"><h2>${esc(j.title || n)} ${chip(j.status)}</h2>
      <div class="sub">${esc(j.result_line || "")} ${esc(j.binary || "")}</div>
      <div class="scroll"><table><thead><tr><th>Case</th><th>Operators</th><th>CPU fallback</th><th>Latency ms</th><th>Delegate / CPU</th><th>Accuracy</th><th>Status</th></tr></thead>
      <tbody>${names.map(c => row(c, cases[c])).join("")}</tbody></table></div>
      <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>ESP-NN delegate time</span><span><span class="sw" style="background:var(--warn)"></span>ExecuTorch portable CPU kernels</span></div></section>`;
  }).join("");
  $("summary").textContent = total ? `${pass}/${total} cases passing` : "";
  $("state").textContent = s.running ? "running" : "idle";
  $("state").className = "chip " + (s.running ? "busy" : "");
  const log = $("log"), stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = s.logs.join("\n");
  if (stick) log.scrollTop = log.scrollHeight;
}
function connect() {
  const es = new EventSource("/events");
  es.onmessage = e => render(JSON.parse(e.data));
  es.onerror = () => { $("state").textContent = "reconnecting"; };
}
fetch("/state.json").then(r => r.json()).then(render);
connect();
</script></body></html>
"""


def make_handler(state: State):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/state.json":
                    body = json.dumps(state.snapshot()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    seen = -1
                    while True:
                        with state.changed:
                            state.changed.wait_for(
                                lambda seen=seen: state.version != seen, timeout=10
                            )
                        snapshot = state.snapshot()
                        seen = snapshot["version"]
                        self.wfile.write(
                            b"data: " + json.dumps(snapshot).encode() + b"\n\n"
                        )
                        self.wfile.flush()
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start(
    port: int = 8770, open_browser: bool = False, state: State | None = None
) -> Dashboard:
    state = state or State()
    server = Server(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"dashboard: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    return Dashboard(state, url)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="benchmark.py JSON results"
    )
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    dashboard = start(args.port, open_browser=True)
    title = f"{results.get('target', '')} benchmark {results.get('date', '')}"
    dashboard.begin_job("results", title, {})
    for case in results.get("raw_cases", {}).values():
        dashboard.case("results", case)
    dashboard.status("results", "loaded")
    dashboard.finish()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
