# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``AdbLens`` and its ``SimpleADB`` patches.

Exercises the wrapper functions directly with a fake ``SimpleADB``-shaped
instance so we don't need a real device or a full QNN environment.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executorch.backends.qualcomm.debugger.observatory.lenses import adb_patches
from executorch.backends.qualcomm.debugger.observatory.lenses.adb import (
    AdbExecuteEvent,
    AdbLens,
    _scan_error_lines,
    _truncate,
)


def _fake_simple_adb(host=None, device="ABCD1234"):
    return SimpleNamespace(
        device_id=device,
        host_id=host,
        error_only=False,
        runner="examples/qualcomm/executor_runner/qnn_executor_runner",
        workspace="/data/local/tmp/exec/run",
        build_path="/build-android",
        htp_arch="79",
        qnn_config=SimpleNamespace(soc_model="SM8550", target="aarch64-android"),
    )


def _completed(rc=0, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout)


@pytest.fixture(autouse=True)
def _reset_lens():
    AdbLens.clear()
    AdbLens._enabled = True
    AdbLens._config = {}
    yield
    AdbLens.clear()


def test_truncate_helper():
    assert _truncate("abc", 16) == "abc"
    assert _truncate(None, 16) == ""
    out = _truncate("x" * 100, 10)
    assert out.startswith("xxxxxxxxxx")
    assert "truncated" in out


def test_scan_error_lines():
    text = "ok\nFATAL: bad\nfine\nERROR: more\n"
    assert _scan_error_lines(text) == [2, 4]


def test_wrapped_adb_records_push_event_with_size(tmp_path):
    fake = _fake_simple_adb()
    f = tmp_path / "blob.bin"
    f.write_bytes(b"x" * 4096)

    with patch("subprocess.run", return_value=_completed(0)):
        adb_patches._wrapped_adb(fake, ["push", str(f), "/dev/foo"])

    events = AdbLens._raw_events
    assert len(events) == 1
    ev = events[0]
    assert ev["phase"] == "other"  # no surrounding wrapper
    assert ev["argv_info"]["kind"] == "push"
    assert ev["argv_info"]["bytes"] == 4096
    assert ev["exit_code"] == 0
    assert ev["argv"][:3] == ["adb", "-s", "ABCD1234"]
    assert ev["argv"][-2:] == [str(f), "/dev/foo"]


def test_wrapped_adb_remote_host_prefix():
    fake = _fake_simple_adb(host="weilhuan-linux", device="5382e6d2")
    with patch("subprocess.run", return_value=_completed(0)):
        adb_patches._wrapped_adb(fake, ["shell", "true"])
    ev = AdbLens._raw_events[0]
    assert ev["argv"][:5] == ["adb", "-H", "weilhuan-linux", "-s", "5382e6d2"]


def test_wrapped_adb_non_zero_records_event_then_raises():
    fake = _fake_simple_adb()
    with patch("subprocess.run", return_value=_completed(7, stdout="boom\nFATAL: bad")):
        with pytest.raises(RuntimeError):
            adb_patches._wrapped_adb(
                fake,
                ["shell", "exit 7"],
                output_callback=lambda r: None,
            )
    ev = AdbLens._raw_events[0]
    assert ev["exit_code"] == 7
    assert "FATAL" in ev["stdout"]


def test_wrapped_adb_subprocess_exception_records_then_reraises():
    fake = _fake_simple_adb()
    with patch("subprocess.run", side_effect=FileNotFoundError("adb missing")):
        with pytest.raises(FileNotFoundError):
            adb_patches._wrapped_adb(fake, ["shell", "true"])
    ev = AdbLens._raw_events[0]
    assert ev["exit_code"] is None
    assert "FileNotFoundError" in ev["error"]


def test_inference_lifecycle_collects_left_panel_record():
    fake = _fake_simple_adb()
    inference_stdout = (
        "starting...\nstep 1 ok\nstep 2 ok\nERROR: oh no\nfinish\n"
    )

    from executorch.devtools.observatory.observatory import Observatory

    Observatory.clear()
    Observatory.register_lens(AdbLens)

    with Observatory.enable_context(config={"adb": {"fetch_logcat": False, "fetch_dmesg": False}}):
        AdbLens.on_session_start(Observatory._get_current_context())

        AdbLens.begin_inference()
        adb_patches._push_phase("execute")
        with patch("subprocess.run", return_value=_completed(0, stdout=inference_stdout)):
            adb_patches._wrapped_adb(
                fake,
                ["shell", "cd /data/.../ws && ./qnn_executor_runner --model_path foo.pte"],
                output_callback=lambda r: None,
            )
        adb_patches._pop_phase()
        AdbLens.end_inference(status="pass", error=None)
        AdbLens.maybe_fetch_logcat_dmesg(fake, adb_patches._wrapped_adb_for_fetch)

    assert "adb.execute #1" in Observatory.list_collected()
    rec = Observatory.get("adb.execute #1")
    digest = rec.data["adb"]
    assert digest["status"] == "pass"
    assert digest["exit_code"] == 0
    assert digest["command"].startswith("cd /data/.../ws &&")
    assert digest["error_lines"] == [4]
    assert "foo.pte" in digest["command"]
    Observatory.clear()


def test_failed_execute_records_failure():
    fake = _fake_simple_adb()
    from executorch.devtools.observatory.observatory import Observatory

    Observatory.clear()
    Observatory.register_lens(AdbLens)

    with Observatory.enable_context(config={"adb": {"fetch_logcat": False, "fetch_dmesg": False}}):
        AdbLens.on_session_start(Observatory._get_current_context())

        AdbLens.begin_inference()
        adb_patches._push_phase("execute")
        try:
            with patch("subprocess.run", return_value=_completed(1, stdout="ERROR: boom")):
                with pytest.raises(RuntimeError):
                    adb_patches._wrapped_adb(
                        fake,
                        ["shell", "./qnn_executor_runner --model_path bad.pte"],
                        output_callback=lambda r: None,
                    )
        finally:
            adb_patches._pop_phase()
            AdbLens.end_inference(status="fail", error="RuntimeError")
            AdbLens.maybe_fetch_logcat_dmesg(fake, adb_patches._wrapped_adb_for_fetch)

    digest = Observatory.get("adb.execute #1").data["adb"]
    assert digest["status"] == "fail"
    assert digest["exit_code"] == 1
    assert digest["error_lines"] == [1]
    Observatory.clear()


def test_logcat_failure_does_not_drop_inference():
    fake = _fake_simple_adb()
    from executorch.devtools.observatory.observatory import Observatory

    Observatory.clear()
    Observatory.register_lens(AdbLens)

    call_count = {"n": 0}

    def fake_run(*a, **kw):
        call_count["n"] += 1
        # First call: the inference shell -- success.
        if call_count["n"] == 1:
            return _completed(0, stdout="ok")
        # Second call: logcat -- success.
        if call_count["n"] == 2:
            return _completed(0, stdout="logcat output")
        # Third call: dmesg -- denied.
        return _completed(1, stdout="permission denied")

    with Observatory.enable_context(config={"adb": {"fetch_logcat": True, "fetch_dmesg": True}}):
        AdbLens.on_session_start(Observatory._get_current_context())

        AdbLens.begin_inference()
        adb_patches._push_phase("execute")
        with patch("subprocess.run", side_effect=fake_run):
            adb_patches._wrapped_adb(
                fake,
                ["shell", "./qnn_executor_runner"],
                output_callback=lambda r: None,
            )
            adb_patches._pop_phase()
            AdbLens.end_inference(status="pass", error=None)
            AdbLens.maybe_fetch_logcat_dmesg(fake, adb_patches._wrapped_adb_for_fetch)

    digest = Observatory.get("adb.execute #1").data["adb"]
    assert digest["status"] == "pass"
    assert digest["logcat_status"] == "ok"
    assert "logcat output" in (digest["logcat"] or "")
    assert digest["dmesg_status"].startswith("failed:")
    Observatory.clear()


def test_push_pull_groups_aggregate(tmp_path):
    fake = _fake_simple_adb()
    a = tmp_path / "a.bin"; a.write_bytes(b"a" * 1024)
    b = tmp_path / "b.bin"; b.write_bytes(b"b" * 2048)

    AdbLens.begin_group("push")
    adb_patches._push_phase("push")
    with patch("subprocess.run", return_value=_completed(0)):
        adb_patches._wrapped_adb(fake, ["push", str(a), "/dev/foo"])
        adb_patches._wrapped_adb(fake, ["push", str(b), "/dev/foo"])
    adb_patches._pop_phase()
    AdbLens.end_group("push")

    AdbLens.begin_group("pull")
    adb_patches._push_phase("pull")
    with patch("subprocess.run", return_value=_completed(0)):
        adb_patches._wrapped_adb(fake, ["pull", "-a", "/dev/foo", str(tmp_path)])
    adb_patches._pop_phase()
    AdbLens.end_group("pull")

    assert len(AdbLens._push_groups) == 1
    grp = AdbLens._push_groups[0]
    assert grp["file_count"] == 2
    assert grp["total_bytes"] == 1024 + 2048
    assert grp["status"] == "pass"

    assert len(AdbLens._pull_groups) == 1
    pgrp = AdbLens._pull_groups[0]
    assert pgrp["pull_count"] == 1


def test_install_uninstall_round_trip():
    """install/uninstall_adb_patches restore SimpleADB methods."""
    from executorch.backends.qualcomm.export_utils import SimpleADB

    original_adb = SimpleADB._adb
    original_push = SimpleADB.push
    try:
        adb_patches.install_adb_patches()
        assert SimpleADB._adb is not original_adb
        assert SimpleADB.push is not original_push
        assert adb_patches.is_installed() is True

        # idempotent
        adb_patches.install_adb_patches()
        assert adb_patches.is_installed() is True

        adb_patches.uninstall_adb_patches()
        assert SimpleADB._adb is original_adb
        assert SimpleADB.push is original_push
        assert adb_patches.is_installed() is False
    finally:
        if adb_patches.is_installed():
            adb_patches.uninstall_adb_patches()


def test_session_end_serializes_to_json():
    fake = _fake_simple_adb()
    AdbLens.note_simple_adb(fake)
    AdbLens.begin_group("push")
    adb_patches._push_phase("push")
    with patch("subprocess.run", return_value=_completed(0)):
        adb_patches._wrapped_adb(fake, ["shell", "mkdir -p /data/foo"])
    adb_patches._pop_phase()
    AdbLens.end_group("push")

    payload = AdbLens.on_session_end(None)
    json.dumps(payload)  # must not raise

    assert payload["device_info"][0]["device_serial"] == "ABCD1234"
    assert payload["transfers"]["push_groups"][0]["status"] == "pass"


def test_disabled_lens_skips_patching():
    from executorch.devtools.observatory.observatory import Observatory

    Observatory.clear()

    with Observatory.enable_context(config={"adb": {"enabled": False}}):
        result = AdbLens.on_session_start(Observatory._get_current_context())
    assert result == {"enabled": False}
    assert AdbLens._enabled is False
    Observatory.clear()
