# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Live viewer for the ESP32-S3 camera inference stream (apps/camera_stream).

The board prints text lines on its USB-Serial-JTAG port:
    #CAM ...                       stream configuration
    #CROP x= y= side=              centre crop fed to the model, in camera pixels
    #RES seq= pre_us= infer_us= cycle_us= top=idx:prob,...
    #JPG kind=cam|input seq= res= w= h= enc_us= len= crc= b64=...

Terminal dashboard (stdlib only; --preview draws the frame with Pillow):
    python3 tools/camera_viewer.py --port /dev/ttyACM0 [--preview]

Chrome dashboard on http://127.0.0.1:8765 (stdlib only):
    python3 tools/camera_viewer.py --web

Automated check that prints a JSON summary and exits:
    python3 tools/camera_viewer.py --duration 30 --summary-json out.json
"""

from __future__ import annotations

import argparse
import base64
import collections
import fcntl
import http.server
import io
import json
import os
import re
import select
import shutil
import socketserver
import struct
import sys
import termios
import threading
import time
import tty
import zlib
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
TIOCM_DTR = 0x002
TIOCM_RTS = 0x004
INJECTED_LOG = re.compile(rb"cam_hal: [^\r\n]{0,96}\r\n")


def load_labels(path: Path) -> list[str]:
    if path.is_file():
        return [line.strip() for line in path.read_text().splitlines()]
    return []


def parse_fields(text: str) -> dict[str, str]:
    fields = {}
    for item in text.split():
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


class SerialPort:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        attributes = termios.tcgetattr(self.fd)
        attributes[4] = attributes[5] = termios.B115200
        attributes[6][termios.VMIN] = 1
        attributes[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attributes)

    def _set_lines(self, dtr: bool, rts: bool) -> None:
        bits = struct.unpack(
            "I", fcntl.ioctl(self.fd, termios.TIOCMGET, struct.pack("I", 0))
        )[0]
        bits = (bits | TIOCM_DTR) if dtr else (bits & ~TIOCM_DTR)
        bits = (bits | TIOCM_RTS) if rts else (bits & ~TIOCM_RTS)
        fcntl.ioctl(self.fd, termios.TIOCMSET, struct.pack("I", bits))

    def reset(self) -> None:
        self._set_lines(dtr=False, rts=True)
        time.sleep(0.1)
        self._set_lines(dtr=False, rts=False)

    def read(self) -> bytes:
        ready, _, _ = select.select([self.fd], [], [], 0.5)
        if not ready:
            return b"\0"
        return os.read(self.fd, 65536)

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


class StreamState:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.version = 0
        self.config: dict[str, str] = {}
        self.crop: dict[str, int] = {}
        self.result: dict | None = None
        self.frames: dict[str, dict] = {}
        self.logs = collections.deque(maxlen=12)
        self.errors = collections.deque(maxlen=12)
        self.result_times = collections.deque(maxlen=20)
        self.frame_times = collections.deque(maxlen=60)
        self.byte_times = collections.deque(maxlen=400)
        self.infer_history = collections.deque(maxlen=120)
        self.counts = collections.Counter()
        self.started = time.time()
        self.first_result_at: float | None = None
        self.camera_log = ""
        self.link = "starting"

    def label(self, index: int) -> str:
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return f"class {index}"

    def note_camera_log(self, messages: list[bytes]) -> None:
        with self.lock:
            for message in messages:
                text = message.decode("utf-8", errors="replace").strip()
                key = (
                    "camera_overruns"
                    if "FB-OVF" in text or "FB-SIZE" in text
                    else "camera_logs"
                )
                self.counts[key] += 1
                self.camera_log = text

    def add_bytes(self, count: int) -> None:
        with self.lock:
            self.byte_times.append((time.time(), count))

    def handle_line(self, raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            return
        now = time.time()
        if line.startswith("#JPG "):
            self._handle_jpeg(line, now)
            return
        with self.lock:
            if line.startswith("#RES "):
                fields = parse_fields(line[5:])
                top = []
                for item in fields.get("top", "").split(","):
                    if ":" in item:
                        index, probability = item.split(":", 1)
                        top.append(
                            {
                                "index": int(index),
                                "label": self.label(int(index)),
                                "probability": float(probability),
                            }
                        )
                self.result = {
                    "seq": int(fields.get("seq", 0)),
                    "pre_ms": int(fields.get("pre_us", 0)) / 1000.0,
                    "infer_ms": int(fields.get("infer_us", 0)) / 1000.0,
                    "cycle_ms": int(fields.get("cycle_us", 0)) / 1000.0,
                    "latency_ms": int(fields.get("latency_us", 0)) / 1000.0,
                    "decode_ms": int(fields.get("dec_us", 0)) / 1000.0,
                    "decode_errors": int(fields.get("dec_err", 0)),
                    "dropped": int(fields.get("dropped", 0)),
                    "usb_fail": int(fields.get("usb_fail", 0)),
                    "top": top,
                    "host_time": now,
                }
                self.result_times.append(now)
                self.infer_history.append(self.result["infer_ms"])
                self.counts["results"] += 1
                if self.first_result_at is None:
                    self.first_result_at = now
            elif line.startswith("#CAM "):
                self.config = parse_fields(line[5:])
            elif line.startswith("#CROP "):
                self.crop = {k: int(v) for k, v in parse_fields(line[6:]).items()}
            elif line.startswith("#ERR"):
                self.errors.append(line)
                self.counts["device_errors"] += 1
            else:
                self.logs.append(line)
            self.version += 1
            self.changed.notify_all()

    def _handle_jpeg(self, line: str, now: float) -> None:
        head, _, payload = line.partition(" b64=")
        fields = parse_fields(head[5:])
        try:
            data = base64.b64decode(payload.strip(), validate=True)
            valid = len(data) == int(fields["len"]) and zlib.crc32(data) == int(
                fields["crc"], 16
            )
        except (ValueError, KeyError):
            valid = False
        with self.lock:
            if not valid:
                self.counts["corrupt_frames"] += 1
                return
            kind = fields.get("kind", "cam")
            self.frames[kind] = {
                "jpeg": data,
                "seq": int(fields.get("seq", 0)),
                "res": int(fields.get("res", 0)),
                "width": int(fields.get("w", 0)),
                "height": int(fields.get("h", 0)),
                "encode_ms": int(fields.get("enc_us", 0)) / 1000.0,
                "host_time": now,
            }
            self.counts[f"{kind}_frames"] += 1
            if kind == "cam":
                self.frame_times.append(now)
            self.version += 1
            self.changed.notify_all()

    @staticmethod
    def _rate(times: collections.deque) -> float:
        if len(times) < 2 or times[-1] == times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def stats(self) -> dict:
        now = time.time()
        window = [(t, n) for t, n in self.byte_times if now - t <= 3.0]
        throughput = sum(n for _, n in window) / 3.0 if window else 0.0
        result = self.result
        history = list(self.infer_history)
        return {
            "inference_fps": round(self._rate(self.result_times), 3),
            "preview_fps": round(self._rate(self.frame_times), 2),
            "link_kbps": round(throughput / 1024.0, 1),
            "result_age_ms": (
                round((now - result["host_time"]) * 1000.0) if result else None
            ),
            "infer_ms_mean": round(sum(history) / len(history), 1) if history else None,
            "infer_ms_min": min(history) if history else None,
            "infer_ms_max": max(history) if history else None,
            "counts": dict(self.counts),
            "latency_ms": result["latency_ms"] if result else None,
            "last_camera_log": self.camera_log,
            "uptime_s": round(now - self.started, 1),
            "link": self.link,
            "config": self.config,
            "crop": self.crop,
        }

    def snapshot(self) -> dict:
        with self.lock:
            frame = self.frames.get("cam")
            return {
                "stats": self.stats(),
                "result": self.result,
                "frame": (
                    {k: v for k, v in frame.items() if k != "jpeg"} if frame else None
                ),
                "infer_history": list(self.infer_history),
                "errors": list(self.errors),
                "logs": list(self.logs),
            }


def set_link(state: StreamState, text: str) -> None:
    with state.lock:
        state.link = text
        state.logs.append(f"viewer: {text}")
        state.version += 1
        state.changed.notify_all()


def reader_thread(  # noqa: C901
    path: str, reset: bool, pause_file: Path, state: StreamState, raw_log
) -> None:
    pending = bytearray()
    port = None
    while True:
        if port is None:
            if pause_file.exists():
                if state.link != "paused for flashing":
                    set_link(state, "paused for flashing")
                time.sleep(0.3)
                continue
            try:
                port = SerialPort(path)
            except OSError:
                if state.link != "waiting for board":
                    set_link(state, "waiting for board")
                time.sleep(0.5)
                continue
            if reset:
                port.reset()
                reset = False
            pending.clear()
            set_link(state, "connected")
        if pause_file.exists():
            port.close()
            port = None
            continue
        try:
            chunk = port.read()
        except OSError:
            chunk = b""
        if not chunk:
            port.close()
            port = None
            time.sleep(0.3)
            continue
        if chunk == b"\0":
            continue
        if raw_log is not None:
            raw_log.write(chunk)
        state.add_bytes(len(chunk))
        pending += chunk
        if b"cam_hal: " in pending:
            injected = INJECTED_LOG.findall(pending)
            if injected:
                pending = bytearray(INJECTED_LOG.sub(b"", pending))
                state.note_camera_log(injected)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                break
            line = bytes(pending[: newline + 1])
            del pending[: newline + 1]
            state.handle_line(line)
        if len(pending) > 4 * 1024 * 1024:
            pending.clear()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-S3 Live Inference</title>
<style>
:root { --bg:#f6f7f9; --panel:#ffffff; --text:#1d2330; --muted:#5f6b7d; --line:#dde2ea;
        --accent:#2563eb; --accent-soft:#dbe7ff; --good:#15803d; --warn:#b45309; --bad:#b91c1c; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#0f1218; --panel:#171b23; --text:#e7ebf2; --muted:#98a2b3;
        --line:#2a303c; --accent:#6ea0ff; --accent-soft:#1d2a44; --good:#4ade80; --warn:#fbbf24; --bad:#f87171; }
}
:root[data-theme="dark"] { --bg:#0f1218; --panel:#171b23; --text:#e7ebf2; --muted:#98a2b3;
        --line:#2a303c; --accent:#6ea0ff; --accent-soft:#1d2a44; --good:#4ade80; --warn:#fbbf24; --bad:#f87171; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
header { display:flex; align-items:center; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0; font-weight:600; }
#status { margin-left:auto; font-size:13px; color:var(--muted); display:flex; align-items:center; gap:6px; }
#dot { width:9px; height:9px; border-radius:50%; background:var(--bad); }
main { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:16px; padding:16px; max-width:1200px; margin:0 auto; }
@media (max-width:820px) { main { grid-template-columns:1fr; } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 10px; font-weight:600; }
.viewport { position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden; }
.viewport img { width:100%; height:100%; object-fit:contain; display:block; }
#cropbox { position:absolute; border:2px solid var(--accent); border-radius:3px; display:none; pointer-events:none; }
#prediction { font-size:26px; font-weight:650; margin:12px 0 2px; min-height:1.3em; }
#prediction-sub { color:var(--muted); font-size:13px; }
.bars { display:flex; flex-direction:column; gap:8px; }
.bar-row { display:grid; grid-template-columns:minmax(0,1fr) 56px; gap:8px; align-items:center; }
.bar-label { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:13px; }
.bar-track { grid-column:1 / span 2; height:6px; background:var(--accent-soft); border-radius:3px; overflow:hidden; }
.bar-fill { height:100%; background:var(--accent); width:0; transition:width .25s ease; }
.bar-value { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); font-size:13px; }
.metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.metric { border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
.metric .v { font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; }
.metric .k { font-size:12px; color:var(--muted); }
.side { display:flex; flex-direction:column; gap:16px; }
#input { width:100%; max-width:224px; aspect-ratio:1; image-rendering:auto; border-radius:6px; background:#000; display:block; }
.input-wrap { display:flex; gap:12px; align-items:flex-start; }
.input-meta { font-size:12px; color:var(--muted); }
canvas { width:100%; height:60px; display:block; }
#log { font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); white-space:pre-wrap; max-height:140px; overflow:auto; margin:0; }
.err { color:var(--bad); }
</style></head>
<body>
<header><h1>ESP32-S3 live inference</h1><span id="model" style="color:var(--muted);font-size:13px"></span>
<div id="status"><span id="dot"></span><span id="status-text">connecting</span></div></header>
<main>
  <section class="panel">
    <h2>Camera preview</h2>
    <div class="viewport" id="viewport"><img id="cam" src="stream.mjpg" alt="camera stream"><div id="cropbox"></div></div>
    <div id="prediction">waiting for first result</div>
    <div id="prediction-sub"></div>
    <div class="metrics" style="margin-top:14px">
      <div class="metric"><div class="v" id="m-ifps">-</div><div class="k">inferences / s</div></div>
      <div class="metric"><div class="v" id="m-infer">-</div><div class="k">inference ms</div></div>
      <div class="metric"><div class="v" id="m-pre">-</div><div class="k">preprocess ms</div></div>
      <div class="metric"><div class="v" id="m-pfps">-</div><div class="k">preview fps</div></div>
      <div class="metric"><div class="v" id="m-link">-</div><div class="k">USB KB/s</div></div>
      <div class="metric"><div class="v" id="m-lat">-</div><div class="k">capture to result ms</div></div>
    </div>
  </section>
  <div class="side">
    <section class="panel">
      <h2>Top 5</h2>
      <div class="bars" id="bars"></div>
    </section>
    <section class="panel">
      <h2>What the model sees</h2>
      <div class="input-wrap"><img id="input" alt="model input"><div class="input-meta" id="input-meta"></div></div>
    </section>
    <section class="panel">
      <h2>Inference latency</h2>
      <canvas id="spark" width="600" height="120"></canvas>
    </section>
    <section class="panel">
      <h2>Device log</h2>
      <pre id="log"></pre>
    </section>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const bars = $("bars");
for (let i = 0; i < 5; i++) {
  bars.insertAdjacentHTML("beforeend",
    `<div class="bar-row"><span class="bar-label" id="bl${i}">-</span><span class="bar-value" id="bv${i}"></span>
     <div class="bar-track"><div class="bar-fill" id="bf${i}"></div></div></div>`);
}
let lastInputSeq = -1, lastSeq = -1, camStale = false;
document.getElementById("cam").addEventListener("error", () => { camStale = true; });
function fmt(v, d) { return v === null || v === undefined ? "-" : Number(v).toFixed(d); }
function placeCrop(stats, frame) {
  const box = $("cropbox"), c = stats.crop, vp = $("viewport");
  if (!c || c.side === undefined || !frame || !frame.width) { box.style.display = "none"; return; }
  const W = vp.clientWidth, H = vp.clientHeight, s = Math.min(W / frame.width, H / frame.height);
  const ox = (W - frame.width * s) / 2, oy = (H - frame.height * s) / 2;
  Object.assign(box.style, { display:"block", left:(ox + c.x * s) + "px", top:(oy + c.y * s) + "px",
                             width:(c.side * s) + "px", height:(c.side * s) + "px" });
}
function spark(values) {
  const cv = $("spark"), g = cv.getContext("2d"), css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, cv.width, cv.height);
  if (values.length < 2) return;
  const lo = Math.min(...values) * 0.98, hi = Math.max(...values) * 1.02, span = Math.max(hi - lo, 1);
  g.strokeStyle = css.getPropertyValue("--accent").trim(); g.lineWidth = 3; g.beginPath();
  values.forEach((v, i) => {
    const x = i / (values.length - 1) * cv.width, y = cv.height - (v - lo) / span * (cv.height - 10) - 5;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
  g.fillStyle = css.getPropertyValue("--muted").trim(); g.font = "22px system-ui";
  g.fillText(`${lo.toFixed(0)}-${hi.toFixed(0)} ms`, 8, 26);
}
function render(s) {
  const st = s.stats, r = s.result;
  $("m-ifps").textContent = fmt(st.inference_fps, 2);
  $("m-pfps").textContent = fmt(st.preview_fps, 1);
  $("m-link").textContent = fmt(st.link_kbps, 0);
  const cfg = st.config || {};
  $("model").textContent = cfg.model_input ? `input ${cfg.model_input}, ${cfg.classes} classes, dual core ${cfg.dual_core === "1" ? "on" : "off"}` : "";
  const live = st.result_age_ms !== null && st.result_age_ms < 5000;
  $("dot").style.background = live ? "var(--good)" : "var(--warn)";
  $("status-text").textContent = st.link && st.link !== "connected" ? st.link : (live ? "streaming" : (r ? "stalled" : "waiting for board"));
  if (live && camStale) { camStale = false; $("cam").src = "stream.mjpg?t=" + Date.now(); }
  if (r) {
    $("m-infer").textContent = fmt(r.infer_ms, 0);
    $("m-pre").textContent = fmt(r.pre_ms, 0);
    $("m-lat").textContent = fmt(r.latency_ms, 0);
    if (r.seq !== lastSeq) {
      lastSeq = r.seq;
      const best = r.top[0];
      $("prediction").textContent = best ? best.label : "-";
      $("prediction-sub").textContent = best ? `${(best.probability * 100).toFixed(1)}% · result #${r.seq} · cycle ${r.cycle_ms.toFixed(0)} ms` : "";
      r.top.forEach((t, i) => {
        $("bl" + i).textContent = t.label;
        $("bv" + i).textContent = (t.probability * 100).toFixed(1) + "%";
        $("bf" + i).style.width = Math.max(0, Math.min(100, t.probability * 100)) + "%";
      });
    }
  }
  placeCrop(st, s.frame);
  if (s.input_seq !== undefined && s.input_seq !== lastInputSeq) {
    lastInputSeq = s.input_seq;
    $("input").src = "/input.jpg?seq=" + s.input_seq;
    $("input-meta").textContent = `frame behind result #${s.input_seq}`;
  }
  spark(s.infer_history);
  const c = st.counts || {};
  const lines = (s.errors || []).map(e => `<span class="err">${e.replace(/</g, "&lt;")}</span>`)
    .concat((s.logs || []).slice(-8).map(l => l.replace(/</g, "&lt;")));
  lines.push(`frames ok ${c.cam_frames || 0}, corrupt ${c.corrupt_frames || 0}, camera overruns ${c.camera_overruns || 0}, results ${c.results || 0}`
    + (r ? `, results dropped on device ${r.dropped}` : ""));
  $("log").innerHTML = lines.join("\n");
}
function connect() {
  const es = new EventSource("/events");
  es.onmessage = ev => render(JSON.parse(ev.data));
  es.onerror = () => { camStale = true; $("dot").style.background = "var(--bad)"; $("status-text").textContent = "viewer disconnected, retrying"; };
}
window.addEventListener("resize", () => {});
connect();
</script>
</body></html>
"""


def make_handler(state: StreamState):  # noqa: C901
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif path == "/state.json":
                    self._send(
                        200, "application/json", json.dumps(self._snapshot()).encode()
                    )
                elif path in ("/frame.jpg", "/input.jpg"):
                    kind = "cam" if path == "/frame.jpg" else "input"
                    with state.lock:
                        frame = state.frames.get(kind)
                    if frame is None:
                        self._send(404, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", frame["jpeg"])
                elif path == "/stream.mjpg":
                    self._mjpeg()
                elif path == "/events":
                    self._events()
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _snapshot(self) -> dict:
            snapshot = state.snapshot()
            with state.lock:
                frame = state.frames.get("input")
                snapshot["input_seq"] = frame["seq"] if frame else None
            return snapshot

        def _mjpeg(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last = None
            while True:
                with state.changed:
                    state.changed.wait_for(
                        lambda last=last: state.frames.get("cam") is not None
                        and state.frames["cam"]["seq"] != last,
                        timeout=5.0,
                    )
                    frame = state.frames.get("cam")
                if frame is None or frame["seq"] == last:
                    continue
                last = frame["seq"]
                jpeg = frame["jpeg"]
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                with state.changed:
                    state.changed.wait(timeout=0.5)
                self.wfile.write(
                    b"data: " + json.dumps(self._snapshot()).encode() + b"\n\n"
                )
                self.wfile.flush()
                time.sleep(0.2)

    return Handler


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def terminal_frame(jpeg: bytes, columns: int) -> list[str]:
    try:
        from PIL import Image
    except ImportError:
        return ["(install Pillow for --preview)"]
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    rows = max(2, int(columns * image.height / image.width) // 2 * 2)
    image = image.resize((columns, rows))
    pixels = image.load()
    lines = []
    for y in range(0, rows, 2):
        parts = []
        for x in range(columns):
            top = pixels[x, y]
            bottom = pixels[x, y + 1]
            parts.append(
                f"\x1b[38;2;{top[0]};{top[1]};{top[2]}m\x1b[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀"
            )
        lines.append("".join(parts) + "\x1b[0m")
    return lines


def run_terminal(state: StreamState, args) -> None:
    interactive = sys.stdout.isatty() and not args.plain
    last_seq = None
    deadline = time.time() + args.duration if args.duration else None
    if interactive:
        sys.stdout.write("\x1b[?25l\x1b[2J")
    try:
        while deadline is None or time.time() < deadline:
            time.sleep(0.25 if interactive else 0.1)
            snap = state.snapshot()
            result = snap["result"]
            stats = snap["stats"]
            if not interactive:
                if result and result["seq"] != last_seq:
                    last_seq = result["seq"]
                    top = ", ".join(
                        f"{t['label']} {t['probability'] * 100:.1f}%"
                        for t in result["top"][:3]
                    )
                    print(
                        f"#{result['seq']:4d} infer {result['infer_ms']:6.1f} ms  pre {result['pre_ms']:5.1f} ms  "
                        f"latency {result['latency_ms']:6.1f} ms  "
                        f"{stats['inference_fps']:.2f} inf/s  preview {stats['preview_fps']:.1f} fps  "
                        f"{stats['link_kbps']:.0f} KB/s | {top}",
                        flush=True,
                    )
                continue
            width = shutil.get_terminal_size((100, 40)).columns
            out = [
                "\x1b[H",
                f"\x1b[1mESP32-S3 live inference\x1b[0m  {state.config.get('model_input', '')} "
                f"dual_core={state.config.get('dual_core', '?')}\x1b[K",
                "\x1b[K",
            ]
            if args.preview:
                with state.lock:
                    frame = state.frames.get("cam")
                if frame:
                    out += list(
                        terminal_frame(
                            frame["jpeg"], min(args.preview_columns, width - 2)
                        )
                    )
                    out.append("\x1b[K")
            if result:
                out.append(
                    f"result #{result['seq']}  inference {result['infer_ms']:.1f} ms  preprocess "
                    f"{result['pre_ms']:.1f} ms  cycle {result['cycle_ms']:.0f} ms  "
                    f"capture-to-result {result['latency_ms']:.0f} ms\x1b[K"
                )
                bar_width = max(10, min(40, width - 50))
                for t in result["top"]:
                    filled = int(round(t["probability"] * bar_width))
                    out.append(
                        f"  {t['label'][:30]:30s} {'#' * filled}{'.' * (bar_width - filled)} "
                        f"{t['probability'] * 100:5.1f}%\x1b[K"
                    )
            else:
                out.append(
                    "waiting for the first result (the board runs its self-test first)\x1b[K"
                )
            counts = stats["counts"]
            out.append("\x1b[K")
            out.append(
                f"inference {stats['inference_fps']:.2f}/s  preview {stats['preview_fps']:.1f} fps  "
                f"USB {stats['link_kbps']:.0f} KB/s  result age {stats['result_age_ms']} ms\x1b[K"
            )
            out.append(
                f"frames {counts.get('cam_frames', 0)}  corrupt {counts.get('corrupt_frames', 0)}  "
                f"camera overruns {counts.get('camera_overruns', 0)}  "
                f"device errors {counts.get('device_errors', 0)}\x1b[K"
            )
            for line in snap["errors"][-3:] + snap["logs"][-4:]:
                out.append(f"\x1b[2m{line[:width - 1]}\x1b[0m\x1b[K")
            out.append("\x1b[J")
            sys.stdout.write("\n".join(out))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            sys.stdout.write("\x1b[0m\x1b[?25h\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default=os.environ.get("ESP_PORT", "/dev/ttyACM0"))
    parser.add_argument(
        "--labels", type=Path, default=TOOLS_DIR / "imagenet_classes.txt"
    )
    parser.add_argument(
        "--reset", action="store_true", help="reset the board after opening the port"
    )
    parser.add_argument("--web", action="store_true", help="serve the Chrome dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument(
        "--preview", action="store_true", help="draw the camera frame in the terminal"
    )
    parser.add_argument("--preview-columns", type=int, default=64)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="one line per result instead of a dashboard",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0, help="exit after this many seconds"
    )
    parser.add_argument(
        "--summary-json", type=Path, help="write final statistics here on exit"
    )
    parser.add_argument("--raw-log", type=Path, help="save the raw serial stream")
    parser.add_argument(
        "--pause-file",
        type=Path,
        default=TOOLS_DIR / "build" / ".viewer_pause",
        help="the serial port is released while this file exists (flash.sh creates it)",
    )
    args = parser.parse_args()

    state = StreamState(load_labels(args.labels))
    raw_log = open(args.raw_log, "wb") if args.raw_log else None
    threading.Thread(
        target=reader_thread,
        args=(args.port, args.reset, args.pause_file, state, raw_log),
        daemon=True,
    ).start()

    if args.web:
        server = ThreadingServer((args.host, args.http_port), make_handler(state))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Chrome dashboard: http://{args.host}:{args.http_port}/", flush=True)

    run_terminal(state, args)

    with state.lock:
        summary = state.stats()
        summary["last_result"] = state.result
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2))
    if args.duration:
        print(json.dumps(summary, indent=2))
    if raw_log:
        raw_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
