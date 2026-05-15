# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``AdbLens`` -- captures and visualizes ``SimpleADB`` activity.

Dashboard sections (rendered when the lens is enabled):
1. Device Info -- serial, host, soc, htp arch, workspace, build path.
2. Transfers -- compact summary of each push/pull group (file count,
   bytes, duration, status).

Per-record (one ``adb.execute`` record per inference call):
1. Status badge, full ``qnn_executor_runner`` command (copyable),
   exit code, duration.
2. Scrollable monospace log of stdout with error lines highlighted.
3. Optional ``logcat -d`` and ``dmesg`` panels collected after the
   inference exits.

The lens uses ``adb_patches.install_adb_patches`` to wrap
``SimpleADB`` for the duration of the Observatory session. Patches are
removed on ``on_session_end`` and on lens ``clear``.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from executorch.devtools.observatory.interfaces import (
    AnalysisResult,
    CustomBlock,
    CustomRecordSpec,
    Frontend,
    Lens,
    ObservationContext,
    RecordDigest,
    TableBlock,
    TableRecordSpec,
    ViewList,
)
from executorch.devtools.observatory.observatory import Observatory


_DEFAULT_MAX_STDOUT_BYTES = 4 * 1024 * 1024
_TRUNCATION_MARKER = "\n[... truncated by AdbLens to {bytes} bytes ...]\n"
_ERROR_RE = re.compile(
    r"\b(ERROR|FATAL|Aborted|Abort|Failed assertion|Segmentation fault|"
    r"FATAL EXCEPTION|panic|kernel BUG)\b",
    re.IGNORECASE,
)


@dataclass
class AdbExecuteEvent:
    """Artifact published per ``SimpleADB.execute`` call.

    Used as the artifact handed to ``Observatory.collect`` so the
    inference shows up as its own record in the left panel.
    """

    sequence: int
    started_at: float
    duration_s: float
    status: str
    exit_code: Optional[int]
    command: str
    adb_argv: List[str]
    stdout: str
    error_lines: List[int] = field(default_factory=list)
    logcat: Optional[str] = None
    logcat_status: Optional[str] = None
    dmesg: Optional[str] = None
    dmesg_status: Optional[str] = None
    related_pulls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_digest(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "status": self.status,
            "exit_code": self.exit_code,
            "command": self.command,
            "adb_argv": self.adb_argv,
            "stdout": self.stdout,
            "error_lines": self.error_lines,
            "logcat": self.logcat,
            "logcat_status": self.logcat_status,
            "dmesg": self.dmesg,
            "dmesg_status": self.dmesg_status,
            "related_pulls": self.related_pulls,
            "error": self.error,
        }


def _truncate(text: Optional[str], max_bytes: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_bytes:
        return text
    head = text[:max_bytes]
    return head + _TRUNCATION_MARKER.format(bytes=max_bytes)


def _scan_error_lines(text: str) -> List[int]:
    return [
        i + 1
        for i, line in enumerate(text.splitlines())
        if _ERROR_RE.search(line)
    ]


class AdbLens(Lens):
    """Lens that records ``SimpleADB`` activity for the run."""

    _lock = threading.Lock()

    _enabled: bool = True
    _config: Dict[str, Any] = {}
    _patches_installed: bool = False
    _install_patches: Optional[Callable[[], None]] = None

    _raw_events: List[Dict[str, Any]] = []
    _device_info: List[Dict[str, Any]] = []
    _device_info_seen: set = set()

    _push_groups: List[Dict[str, Any]] = []
    _pull_groups: List[Dict[str, Any]] = []
    _open_groups: List[Dict[str, Any]] = []

    _execute_seq: int = 0
    _active_inference: Optional[Dict[str, Any]] = None

    @classmethod
    def get_name(cls) -> str:
        return "adb"

    @classmethod
    def register_adb_patches(cls, install_fn: Callable[[], None]) -> None:
        cls._install_patches = install_fn

    @classmethod
    def setup(cls) -> None:
        cls.clear()

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._raw_events = []
            cls._device_info = []
            cls._device_info_seen = set()
            cls._push_groups = []
            cls._pull_groups = []
            cls._open_groups = []
            cls._execute_seq = 0
            cls._active_inference = None

        from .adb_patches import is_installed, uninstall_adb_patches

        if is_installed():
            uninstall_adb_patches()
        cls._patches_installed = False

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> Optional[Dict[str, Any]]:
        cfg = context.config.get("adb", {}) if context else {}
        cls._config = dict(cfg)
        cls._enabled = bool(cfg.get("enabled", True))
        if not cls._enabled:
            return {"enabled": False}

        if cls._install_patches is not None and not cls._patches_installed:
            cls._install_patches()
            cls._patches_installed = True

        return {"enabled": True, "started_at": _utc_now()}

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> Optional[Dict[str, Any]]:
        try:
            with cls._lock:
                payload = {
                    "enabled": cls._enabled,
                    "device_info": list(cls._device_info),
                    "transfers": {
                        "push_groups": list(cls._push_groups),
                        "pull_groups": list(cls._pull_groups),
                    },
                    "raw_events": list(cls._raw_events),
                }
            return payload
        finally:
            from .adb_patches import is_installed, uninstall_adb_patches

            if is_installed():
                uninstall_adb_patches()
            cls._patches_installed = False

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        if not isinstance(artifact, AdbExecuteEvent):
            return None
        return artifact.to_digest()

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    @staticmethod
    def analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        return AnalysisResult()

    # ------------------------------------------------------------------ Recording API
    # Called from ``adb_patches`` -- not part of the Lens protocol.

    @classmethod
    def record_event(
        cls,
        *,
        phase: str,
        argv: List[str],
        argv_info: Dict[str, Any],
        exit_code: Optional[int],
        duration_s: float,
        started_at: float,
        stdout: Optional[str],
        error: Optional[str],
        host_id: Optional[str],
        device_id: Optional[str],
        captured: bool,
    ) -> None:
        if not cls._enabled:
            return

        max_bytes = int(cls._config.get("max_stdout_bytes", _DEFAULT_MAX_STDOUT_BYTES))
        compact_event = {
            "phase": phase,
            "argv": list(argv),
            "argv_info": dict(argv_info),
            "exit_code": exit_code,
            "duration_s": duration_s,
            "started_at": started_at,
            "captured": captured,
            "error": error,
        }
        if captured and stdout is not None:
            compact_event["stdout"] = _truncate(stdout, max_bytes)

        with cls._lock:
            cls._raw_events.append(compact_event)
            for group in cls._open_groups:
                if group["phase"] == phase:
                    group["events"].append(compact_event)
                    if argv_info.get("kind") == "push" and "bytes" in argv_info:
                        group["total_bytes"] = group.get("total_bytes", 0) + int(
                            argv_info["bytes"]
                        )
                        group["file_count"] = group.get("file_count", 0) + 1
                    elif argv_info.get("kind") == "pull":
                        group["pull_count"] = group.get("pull_count", 0) + 1
                    if exit_code not in (0, None) or error is not None:
                        group["status"] = "fail"
                    break

        if cls._active_inference is not None and phase == "execute" and captured:
            cls._active_inference["last_capture"] = compact_event

    @classmethod
    def begin_group(cls, phase: str, *, subphase: Optional[str] = None) -> None:
        if not cls._enabled:
            return
        group = {
            "phase": phase,
            "subphase": subphase,
            "started_at": _utc_now(),
            "duration_s": 0.0,
            "file_count": 0,
            "pull_count": 0,
            "total_bytes": 0,
            "status": "pass",
            "events": [],
        }
        with cls._lock:
            cls._open_groups.append(group)
            t0_attr = f"_t0_{id(group)}"
            setattr(cls, t0_attr, _perf_now())

    @classmethod
    def end_group(cls, phase: str, *, subphase: Optional[str] = None) -> None:
        if not cls._enabled:
            return
        with cls._lock:
            for i in range(len(cls._open_groups) - 1, -1, -1):
                grp = cls._open_groups[i]
                if grp["phase"] == phase and grp.get("subphase") == subphase:
                    del cls._open_groups[i]
                    t0_attr = f"_t0_{id(grp)}"
                    t0 = getattr(cls, t0_attr, None)
                    if t0 is not None:
                        grp["duration_s"] = _perf_now() - t0
                        delattr(cls, t0_attr)
                    grp["id"] = (
                        f"{phase}_" + (f"{subphase}_" if subphase else "")
                        + str(
                            len(cls._push_groups if phase == "push" else cls._pull_groups)
                        )
                    )
                    if phase == "push":
                        cls._push_groups.append(grp)
                    else:
                        cls._pull_groups.append(grp)
                    return

    @classmethod
    def begin_inference(cls) -> None:
        if not cls._enabled:
            return
        with cls._lock:
            cls._execute_seq += 1
            cls._active_inference = {
                "sequence": cls._execute_seq,
                "started_at": _utc_now(),
                "t0": _perf_now(),
                "events_before": len(cls._raw_events),
                "last_capture": None,
            }

    @classmethod
    def end_inference(cls, *, status: str, error: Optional[str]) -> None:
        if not cls._enabled or cls._active_inference is None:
            return
        active = cls._active_inference
        with cls._lock:
            new_events = cls._raw_events[active["events_before"]:]

        last_shell = None
        for ev in reversed(new_events):
            argv = ev.get("argv", [])
            if "shell" in argv:
                last_shell = ev
                break

        captured_events = [ev for ev in new_events if ev.get("captured")]
        primary = active.get("last_capture") or (captured_events[-1] if captured_events else last_shell)

        related_pulls: List[Dict[str, Any]] = []
        for ev in new_events:
            ai = ev.get("argv_info", {})
            if ai.get("kind") == "pull":
                related_pulls.append({
                    "argv": ev["argv"][-3:],
                    "exit_code": ev.get("exit_code"),
                    "duration_s": ev.get("duration_s"),
                })

        argv = primary.get("argv", []) if primary else []
        cmd = ""
        if argv:
            shell_idx = None
            for i, tok in enumerate(argv):
                if tok == "shell":
                    shell_idx = i
                    break
            if shell_idx is not None and shell_idx + 1 < len(argv):
                cmd = " ".join(argv[shell_idx + 1:])

        stdout_text = primary.get("stdout", "") if primary else ""
        max_bytes = int(cls._config.get("max_stdout_bytes", _DEFAULT_MAX_STDOUT_BYTES))
        stdout_text = _truncate(stdout_text or "", max_bytes)
        error_lines = _scan_error_lines(stdout_text)

        exit_code = primary.get("exit_code") if primary else None

        event = AdbExecuteEvent(
            sequence=active["sequence"],
            started_at=active["started_at"],
            duration_s=_perf_now() - active["t0"],
            status=status,
            exit_code=exit_code,
            command=cmd,
            adb_argv=argv,
            stdout=stdout_text,
            error_lines=error_lines,
            related_pulls=related_pulls,
            error=error,
        )
        with cls._lock:
            cls._active_inference = event.to_digest()
            cls._active_inference["__event__"] = event

    @classmethod
    def maybe_fetch_logcat_dmesg(cls, simple_adb: Any, adb_runner: Callable) -> None:
        """Fetch logcat / dmesg post-execute when config allows.

        ``adb_runner`` is the wrapped ``_adb`` that records events under
        the appropriate phase.
        """
        if not cls._enabled or cls._active_inference is None:
            cls._publish_active_inference()
            return

        active = cls._active_inference
        event: AdbExecuteEvent = active.get("__event__") if isinstance(active, dict) else None
        if event is None:
            cls._publish_active_inference()
            return

        from . import adb_patches

        do_logcat = _resolve_auto(cls._config.get("fetch_logcat", "auto"), True)
        do_dmesg = _resolve_auto(cls._config.get("fetch_dmesg", "auto"), True)

        if do_logcat:
            adb_patches._push_phase("logcat")
            try:
                buf: List[str] = []
                adb_runner(
                    simple_adb,
                    ["logcat", "-d"],
                    output_callback=lambda r: buf.append(getattr(r, "stdout", "") or ""),
                )
                event.logcat = _truncate(
                    "".join(buf), int(cls._config.get("max_stdout_bytes", _DEFAULT_MAX_STDOUT_BYTES))
                )
                event.logcat_status = "ok"
            except BaseException as exc:
                event.logcat_status = f"failed: {exc!r}"
                logging.warning("[AdbLens] logcat fetch failed: %s", exc)
            finally:
                adb_patches._pop_phase()

        if do_dmesg:
            adb_patches._push_phase("dmesg")
            try:
                buf2: List[str] = []
                adb_runner(
                    simple_adb,
                    ["shell", "dmesg"],
                    output_callback=lambda r: buf2.append(getattr(r, "stdout", "") or ""),
                )
                event.dmesg = _truncate(
                    "".join(buf2), int(cls._config.get("max_stdout_bytes", _DEFAULT_MAX_STDOUT_BYTES))
                )
                event.dmesg_status = "ok"
            except BaseException as exc:
                event.dmesg_status = f"failed: {exc!r}"
                logging.warning("[AdbLens] dmesg fetch failed: %s", exc)
            finally:
                adb_patches._pop_phase()

        cls._publish_active_inference()

    @classmethod
    def _publish_active_inference(cls) -> None:
        active = cls._active_inference
        if not isinstance(active, dict) or "__event__" not in active:
            cls._active_inference = None
            return
        event: AdbExecuteEvent = active["__event__"]
        cls._active_inference = None
        try:
            Observatory.collect(f"adb.execute #{event.sequence}", event)
        except Exception as exc:
            logging.error("[AdbLens] failed to publish inference event: %s", exc)

    # ------------------------------------------------------------------ Device info

    @classmethod
    def note_simple_adb(cls, instance: Any) -> None:
        if not cls._enabled:
            return
        info = _summarize_simple_adb(instance)
        key = (info.get("host"), info.get("device_serial"))
        with cls._lock:
            if key in cls._device_info_seen:
                return
            cls._device_info_seen.add(key)
            cls._device_info.append(info)

    # ------------------------------------------------------------------ Frontend

    class AdbFrontend(Frontend):
        def resources(self) -> Dict[str, str]:
            return {"css": _LENS_ADB_CSS, "js": _LENS_ADB_JS}

        def dashboard(
            self,
            start: Dict[str, Any],
            end: Dict[str, Any],
            analysis: Dict[str, Any],
            records: List[Any],
        ) -> Optional[ViewList]:
            if not end:
                return None
            if not end.get("enabled", True):
                return None

            blocks: List[Any] = []

            device_info = end.get("device_info") or []
            if device_info:
                if len(device_info) == 1:
                    blocks.append(
                        TableBlock(
                            id="adb_device_info",
                            title="ADB Device Info",
                            record=TableRecordSpec(data=dict(device_info[0])),
                            order=0,
                        )
                    )
                else:
                    blocks.append(
                        CustomBlock(
                            id="adb_device_info",
                            title="ADB Devices",
                            record=CustomRecordSpec(
                                js_func="renderAdbDeviceList",
                                args={"devices": device_info},
                            ),
                            order=0,
                        )
                    )

            transfers = end.get("transfers") or {}
            if transfers.get("push_groups") or transfers.get("pull_groups"):
                blocks.append(
                    CustomBlock(
                        id="adb_transfers",
                        title="ADB File Transfers",
                        record=CustomRecordSpec(
                            js_func="renderAdbTransfers",
                            args={
                                "push_groups": transfers.get("push_groups", []),
                                "pull_groups": transfers.get("pull_groups", []),
                            },
                        ),
                        order=100,
                    )
                )

            return ViewList(blocks=blocks) if blocks else None

        def record(
            self,
            digest: Any,
            analysis: Dict[str, Any],
            context: Dict[str, Any],
        ) -> Optional[ViewList]:
            if not isinstance(digest, dict):
                return None
            if "command" not in digest:
                return None

            return ViewList(
                blocks=[
                    CustomBlock(
                        id="adb_execute",
                        title="ADB Inference",
                        record=CustomRecordSpec(
                            js_func="renderAdbExecute",
                            args={"event": digest},
                        ),
                        order=0,
                    )
                ]
            )

        def check_badges(self, digest: Any, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
            if not isinstance(digest, dict):
                return []
            status = digest.get("status")
            if status == "pass":
                return [{"label": "ADB OK", "class": "badge", "title": "Inference passed"}]
            if status == "fail":
                return [{"label": "ADB FAIL", "class": "badge badge-error", "title": "Inference failed"}]
            return []

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return AdbLens.AdbFrontend()


def _summarize_simple_adb(instance: Any) -> Dict[str, Any]:
    qnn_config = getattr(instance, "qnn_config", None)
    soc_model = getattr(qnn_config, "soc_model", None) if qnn_config else None
    target = getattr(qnn_config, "target", None) if qnn_config else None
    return {
        "device_serial": getattr(instance, "device_id", None),
        "host": getattr(instance, "host_id", None),
        "soc_model": soc_model,
        "htp_arch": getattr(instance, "htp_arch", None),
        "workspace": getattr(instance, "workspace", None),
        "build_path": getattr(instance, "build_path", None),
        "runner": getattr(instance, "runner", None),
        "target": target,
    }


def _resolve_auto(value: Any, auto_default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() == "auto":
        return auto_default
    return bool(value)


def _utc_now() -> float:
    import time as _time
    return _time.time()


def _perf_now() -> float:
    import time as _time
    return _time.perf_counter()


_LENS_ADB_CSS = """
.adb-status {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.adb-status-pass { background: var(--success-color, #22863a); color: #fff; }
.adb-status-fail { background: var(--error-color, #cb2431); color: #fff; }
.adb-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 6px 0 10px 0;
  font-size: 12px;
  color: var(--text-secondary, #6a737d);
}
.adb-meta span { white-space: nowrap; }
.adb-cmd-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
}
.adb-cmd-box {
  flex: 1;
  background: var(--bg-code, #f6f8fa);
  border: 1px solid var(--border-color, #d0d7de);
  border-radius: 4px;
  padding: 6px 8px;
  font-family: var(--font-mono, "SFMono-Regular", Consolas, monospace);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 96px;
  overflow: auto;
}
.adb-btn {
  font-size: 11px;
  padding: 4px 8px;
  border: 1px solid var(--border-color, #d0d7de);
  background: var(--bg-tertiary, #f6f8fa);
  color: var(--text-primary, #24292f);
  border-radius: 4px;
  cursor: pointer;
}
.adb-btn:hover { background: var(--border-color, #d0d7de); }
.adb-log-container {
  border: 1px solid var(--border-color, #d0d7de);
  border-radius: 4px;
  background: var(--bg-tertiary, #f6f8fa);
  max-height: 480px;
  overflow: auto;
  font-family: var(--font-mono, "SFMono-Regular", Consolas, monospace);
  font-size: 12px;
}
.adb-log-row {
  display: flex;
  white-space: pre;
}
.adb-log-row.adb-log-error {
  background: rgba(203, 36, 49, 0.10);
  color: var(--error-color, #cb2431);
}
.adb-log-num {
  flex: 0 0 48px;
  text-align: right;
  padding: 0 8px 0 4px;
  color: var(--text-secondary, #6a737d);
  user-select: none;
  border-right: 1px solid var(--border-color, #d0d7de);
}
.adb-log-text {
  flex: 1;
  padding: 0 8px;
  white-space: pre-wrap;
  word-break: break-all;
}
.adb-section { margin-top: 12px; }
.adb-section-title {
  font-size: 12px;
  font-weight: 600;
  margin: 4px 0;
  color: var(--text-secondary, #6a737d);
}
.adb-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 4px;
}
.adb-tab {
  font-size: 11px;
  padding: 3px 10px;
  border: 1px solid var(--border-color, #d0d7de);
  background: var(--bg-tertiary, #f6f8fa);
  cursor: pointer;
  border-radius: 4px 4px 0 0;
}
.adb-tab.adb-tab-active {
  background: var(--bg-primary, #fff);
  border-bottom-color: transparent;
}
.adb-transfer-summary {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.adb-transfer-summary th,
.adb-transfer-summary td {
  border: 1px solid var(--border-color, #d0d7de);
  padding: 4px 8px;
  text-align: left;
}
.adb-transfer-summary th {
  background: var(--bg-tertiary, #f6f8fa);
  color: var(--text-primary, #24292f);
}
.adb-transfer-detail {
  margin: 6px 0 0 0;
  display: none;
}
.adb-transfer-detail.adb-show { display: block; }
"""


_LENS_ADB_JS = r"""
(function () {
  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function fmtBytes(n) {
    if (n == null || n === 0) return "-";
    var units = ["B", "KB", "MB", "GB"];
    var i = 0; var v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return v.toFixed(v >= 10 || i === 0 ? 0 : 1) + " " + units[i];
  }
  function fmtDuration(s) {
    if (s == null) return "-";
    if (s < 1) return (s * 1000).toFixed(0) + " ms";
    if (s < 60) return s.toFixed(2) + " s";
    var m = Math.floor(s / 60); var sec = (s - m * 60).toFixed(1);
    return m + "m " + sec + "s";
  }
  function copy(text, btn) {
    var done = function () {
      if (!btn) return;
      var orig = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(function () { btn.textContent = orig; }, 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, done);
    } else {
      var ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta); done();
    }
  }
  function makeButton(label, onclick) {
    var b = document.createElement("button");
    b.className = "adb-btn"; b.textContent = label;
    b.addEventListener("click", onclick);
    return b;
  }
  function renderLog(text, errorLines) {
    var box = document.createElement("div");
    box.className = "adb-log-container";
    var errSet = {};
    (errorLines || []).forEach(function (n) { errSet[n] = true; });
    var lines = (text || "").split("\n");
    var max = lines.length;
    if (max > 0 && lines[max - 1] === "") max--;
    var frag = document.createDocumentFragment();
    for (var i = 0; i < max; i++) {
      var row = document.createElement("div");
      row.className = "adb-log-row" + (errSet[i + 1] ? " adb-log-error" : "");
      var num = document.createElement("span");
      num.className = "adb-log-num"; num.textContent = String(i + 1);
      var t = document.createElement("span");
      t.className = "adb-log-text"; t.textContent = lines[i];
      row.appendChild(num); row.appendChild(t);
      frag.appendChild(row);
    }
    box.appendChild(frag);
    return box;
  }

  window.renderAdbExecute = function (container, args, ctx, analysis) {
    var ev = (args && args.event) || {};
    container.innerHTML = "";

    var status = document.createElement("span");
    status.className = "adb-status " + (ev.status === "pass" ? "adb-status-pass" : "adb-status-fail");
    status.textContent = ev.status === "pass" ? "PASS" : "FAIL";
    container.appendChild(status);

    var meta = document.createElement("div");
    meta.className = "adb-meta";
    meta.innerHTML = (
      "<span>exit code: <b>" + escapeHtml(ev.exit_code == null ? "n/a" : ev.exit_code) + "</b></span>" +
      "<span>duration: <b>" + escapeHtml(fmtDuration(ev.duration_s)) + "</b></span>" +
      "<span>seq: <b>#" + escapeHtml(ev.sequence) + "</b></span>"
    );
    container.appendChild(meta);

    var cmdRow = document.createElement("div");
    cmdRow.className = "adb-cmd-row";
    var cmdBox = document.createElement("div");
    cmdBox.className = "adb-cmd-box";
    cmdBox.textContent = ev.command || "(no command captured)";
    cmdRow.appendChild(cmdBox);
    cmdRow.appendChild(makeButton("Copy command", function (e) { copy(ev.command || "", e.currentTarget); }));
    if (ev.adb_argv && ev.adb_argv.length) {
      cmdRow.appendChild(makeButton("Copy adb argv", function (e) {
        copy(ev.adb_argv.map(function (s) { return /[\s"']/.test(s) ? "'" + s.replace(/'/g, "'\\''") + "'" : s; }).join(" "), e.currentTarget);
      }));
    }
    container.appendChild(cmdRow);

    var logHeader = document.createElement("div");
    logHeader.className = "adb-section-title";
    logHeader.style.display = "flex";
    logHeader.style.alignItems = "center";
    logHeader.style.gap = "8px";
    logHeader.innerHTML = "<span>Inference stdout</span>";
    var copyLog = makeButton("Copy raw log", function (e) { copy(ev.stdout || "", e.currentTarget); });
    logHeader.appendChild(copyLog);
    container.appendChild(logHeader);
    container.appendChild(renderLog(ev.stdout || "", ev.error_lines || []));

    function makeStreamSection(title, text, status, marker) {
      if (!text && !status) return null;
      var sec = document.createElement("div");
      sec.className = "adb-section";
      var title_el = document.createElement("div");
      title_el.className = "adb-section-title";
      var statusText = (status && status !== "ok") ? " — " + status : "";
      title_el.innerHTML = "<span>" + escapeHtml(title) + escapeHtml(statusText) + "</span>";
      var copyBtn = makeButton("Copy", function (e) { copy(text || "", e.currentTarget); });
      copyBtn.style.marginLeft = "8px";
      title_el.style.display = "flex";
      title_el.style.alignItems = "center";
      title_el.appendChild(copyBtn);
      sec.appendChild(title_el);
      if (text) {
        sec.appendChild(renderLog(text, []));
      }
      return sec;
    }
    var logcatSec = makeStreamSection("logcat -d", ev.logcat, ev.logcat_status, "logcat");
    if (logcatSec) container.appendChild(logcatSec);
    var dmesgSec = makeStreamSection("adb shell dmesg", ev.dmesg, ev.dmesg_status, "dmesg");
    if (dmesgSec) container.appendChild(dmesgSec);

    if (ev.related_pulls && ev.related_pulls.length) {
      var pulls = document.createElement("div");
      pulls.className = "adb-section";
      pulls.innerHTML = "<div class='adb-section-title'>Related pulls</div>";
      var ul = document.createElement("ul");
      ev.related_pulls.forEach(function (p) {
        var li = document.createElement("li");
        li.style.fontSize = "12px";
        li.textContent = (p.argv || []).join(" ") + "  exit=" + p.exit_code + "  " + fmtDuration(p.duration_s);
        ul.appendChild(li);
      });
      pulls.appendChild(ul);
      container.appendChild(pulls);
    }
  };

  window.renderAdbTransfers = function (container, args, ctx, analysis) {
    var pushes = (args && args.push_groups) || [];
    var pulls = (args && args.pull_groups) || [];
    container.innerHTML = "";
    var table = document.createElement("table");
    table.className = "adb-transfer-summary";
    table.innerHTML = "<thead><tr><th>#</th><th>Phase</th><th>Files</th><th>Bytes</th><th>Duration</th><th>Status</th><th></th></tr></thead>";
    var tbody = document.createElement("tbody");

    function addRow(g, kind, idx) {
      var tr = document.createElement("tr");
      var nFiles = g.file_count != null ? g.file_count : (g.pull_count || (g.events ? g.events.length : 0));
      var statusBadge = "<span class='adb-status " + (g.status === "fail" ? "adb-status-fail" : "adb-status-pass") + "'>" + (g.status === "fail" ? "FAIL" : "OK") + "</span>";
      var label = kind + (g.subphase ? " (" + g.subphase + ")" : "");
      tr.innerHTML = (
        "<td>" + idx + "</td>" +
        "<td>" + escapeHtml(label) + "</td>" +
        "<td>" + nFiles + "</td>" +
        "<td>" + fmtBytes(g.total_bytes) + "</td>" +
        "<td>" + fmtDuration(g.duration_s) + "</td>" +
        "<td>" + statusBadge + "</td>" +
        "<td><button class='adb-btn'>Detail</button></td>"
      );
      tbody.appendChild(tr);
      var detailRow = document.createElement("tr");
      detailRow.style.display = "none";
      var td = document.createElement("td"); td.colSpan = 7;
      var detail = document.createElement("div");
      detail.className = "adb-log-container";
      detail.style.maxHeight = "240px";
      var inner = "";
      (g.events || []).forEach(function (ev) {
        var line = (ev.argv || []).join(" ") + "  exit=" + ev.exit_code + "  " + fmtDuration(ev.duration_s);
        if (ev.argv_info && ev.argv_info.bytes != null) line += "  " + fmtBytes(ev.argv_info.bytes);
        inner += "<div class='adb-log-row" + (ev.exit_code !== 0 ? " adb-log-error" : "") + "'><span class='adb-log-num'></span><span class='adb-log-text'>" + escapeHtml(line) + "</span></div>";
      });
      detail.innerHTML = inner || "<div style='padding:6px;color:var(--text-secondary)'>(no events)</div>";
      td.appendChild(detail);
      detailRow.appendChild(td);
      tbody.appendChild(detailRow);
      tr.querySelector("button").addEventListener("click", function () {
        detailRow.style.display = detailRow.style.display === "none" ? "table-row" : "none";
      });
    }

    var i = 1;
    pushes.forEach(function (g) { addRow(g, "push", i++); });
    pulls.forEach(function (g) { addRow(g, "pull", i++); });
    table.appendChild(tbody);
    container.appendChild(table);
  };

  window.renderAdbDeviceList = function (container, args, ctx, analysis) {
    var devices = (args && args.devices) || [];
    container.innerHTML = "";
    devices.forEach(function (d, i) {
      var h = document.createElement("div");
      h.style.marginBottom = "8px";
      h.innerHTML = "<b>Device " + (i + 1) + "</b>: " + escapeHtml(d.device_serial || "") + (d.host ? " (host: " + escapeHtml(d.host) + ")" : "");
      var pre = document.createElement("pre");
      pre.style.background = "var(--bg-tertiary)";
      pre.style.border = "1px solid var(--border-color)";
      pre.style.padding = "6px";
      pre.style.fontSize = "12px";
      pre.textContent = JSON.stringify(d, null, 2);
      container.appendChild(h);
      container.appendChild(pre);
    });
  };
})();
"""
