# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Preserve a running diagnostic, then qualify the repaired real mini source."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def process_start(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["prior-job", "native-preparation", "baseline-audit", "gguf", "output", "lock"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--wait-process", action="append", required=True, help="PID:start_ticks")
    p.add_argument("--model", choices=["MG", "Gemma"], required=True)
    p.add_argument("--wait-hours", type=float, default=3)
    p.add_argument("--cache-root", type=Path)
    a = p.parse_args()
    assert 0 < a.wait_hours <= 6
    watched = {int(value.split(":")[0]): value.split(":")[1] for value in a.wait_process}
    assert len(watched) == len(a.wait_process)
    a.output.mkdir(parents=True, exist_ok=False)

    def state(status, **extra):
        value = dict(status=status, PID=os.getpid(), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     real_mini_qualified=False, full_models_paused=True, watched_processes=watched, **extra)
        tmp = a.output / "progress.pending.json"
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(a.output / "progress.json")

    try:
        deadline = time.monotonic() + a.wait_hours * 3600
        previous = None
        while True:
            live = [pid for pid, start in watched.items() if process_start(pid) == start]
            if live != previous:
                state("waiting-existing-diagnostic", live_original_processes=live)
                previous = live
            if not live:
                break
            if time.monotonic() >= deadline:
                state("observation-budget-exhausted-diagnostic-not-restarted", live_original_processes=live)
                return
            time.sleep(15)

        evidence = a.output / "prior-diagnostic-evidence"
        evidence.mkdir()
        files = {}
        for relative in ["launch.json", "controller.log", "memcheck.log", "validation/progress.json",
                         "validation/result.json", "validation/ring/artifact-audit.json"]:
            src = a.prior_job / relative
            if not src.is_file():
                continue
            dst = evidence / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files[relative] = dict(SHA256=hashlib.sha256(dst.read_bytes()).hexdigest(), bytes=dst.stat().st_size)
        (evidence / "manifest.json").write_text(json.dumps(dict(prior_job=str(a.prior_job), files=files,
            prior_files_unchanged=True, original_processes_terminal=True), indent=2) + "\n")
        state("starting-repaired-real-mini-qualification", prior_evidence_preserved=True)
        script = Path(__file__).with_name("run_qualified_ring_recovery.py")
        argv = [sys.executable, "-B", "-u", str(script), "--model", a.model]
        for name in ["native-preparation", "baseline-audit", "gguf", "lock"]:
            argv += ["--" + name, str(getattr(a, name.replace("-", "_")))]
        argv += ["--output", str(a.output / "qualification")]
        if a.cache_root:
            argv += ["--cache-root", str(a.cache_root)]
        env = dict(os.environ)
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        with (a.output / "qualification.log").open("x") as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=25200)
        state("real-mini-data-collected-review-required", performance_acceptance_pending=True)
    except BaseException as error:
        state("failed-preserve-evidence", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
