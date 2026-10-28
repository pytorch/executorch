"""Run original mini numerical and lifecycle checks on an isolated RMS export."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main(args):
    job = args.job
    assert job.is_dir() and not (job / "effective-config.json").exists()
    rc = 1

    def state(status, **extra):
        (job / "progress.json").write_text(json.dumps(dict(status=status,
            mini_only=True, qualified=False, PID=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)))

    try:
        state("waiting-for-isolated-mini-export")
        handle = json.loads((args.export_job / "launch-receipt.json").read_text())
        proc = Path("/proc") / str(handle["PID"])
        while not (args.export_job / "exit-receipt.json").exists():
            assert proc.exists(), "Export handle disappeared without a terminal receipt"
            fields = (proc / "stat").read_text().split(") ", 1)[1].split()
            assert int(fields[19]) == handle["start_ticks"] and fields[0] != "Z"
            assert hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest() == handle["command_SHA256"]
            time.sleep(15)
        export_rc = json.loads((args.export_job / "exit-receipt.json").read_text())["returncode"]
        control_diagnostic = None
        if args.diagnostic_unmatched_export_controls:
            assert export_rc == 1
            failure = (args.export_job / "failure.txt").read_text()
            assert 'assert all(item["function_AST_SHA256"] in matched for item in controls[:4])' in failure
            required = json.loads((args.export_job / "input-RMS-control-required.json").read_text())["control"]
            receipts = [json.loads(line) for line in
                (args.export_job / "reduction-layout-actual-control-receipt.jsonl").read_text().splitlines()]
            assert any(all(item[key] == required[key] for key in
                ["function_AST_SHA256", "kwargs", "num_warps", "num_stages"]) for item in receipts)
            controls = json.loads((args.export_job / "Gemma-reference-reduction-layouts.json").read_text())
            matched = {item["function_AST_SHA256"] for item in receipts}
            control_diagnostic = dict(export_returncode=export_rc,
                export_control_gate_passed=False, original_failure=failure,
                input_RMS_control_verified=True,
                unmatched_reference_ASTs=[item["function_AST_SHA256"] for item in controls
                    if item["function_AST_SHA256"] not in matched],
                strict_numerical_thresholds_unchanged=True,
                cannot_qualify_full_export=True)
            (job / "export-control-diagnostic.json").write_text(json.dumps(control_diagnostic, indent=2))
        else:
            assert export_rc == 0
        exported = json.loads((args.export_job / "data/exports/result.json").read_text())
        artifact = exported["arms"][0]
        cfg = json.loads((args.template / "effective-config.json").read_text())
        cfg["PTE"] = str(Path(artifact["artifact"]) / "model.pte")
        cfg["model"] = "Gemma-mini81-input-RMS-reference"
        assert hashlib.sha256(Path(cfg["PTE"]).read_bytes()).hexdigest() == artifact["artifacts"]["model.pte"]["SHA256"]
        assert hashlib.sha256(Path(cfg["PTD"]).read_bytes()).hexdigest() == cfg["PTD_SHA256"]
        (job / "effective-config.json").write_text(json.dumps(cfg, indent=2))
        for name in ["mini_preflight.py", "run_mini81_KV_invariants_v2.py"]:
            (job / name).write_bytes((args.template / name).read_bytes())
        (job / "source-bindings.json").write_text(json.dumps({
            name: hashlib.sha256((job / name).read_bytes()).hexdigest()
            for name in ["mini_preflight.py", "run_mini81_KV_invariants_v2.py"]}, indent=2))
        state("original-five-point-numerical-and-clone-check")
        out = job / "strict-five-point"
        out.mkdir()
        spec = importlib.util.spec_from_file_location("original_preflight", job / "mini_preflight.py")
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        result = preflight.main(cfg, out)
        state("original-independent-session-lifecycle-check", numeric_pass=result["original_numeric_gate_pass"])
        inv = job / "independent-KV"
        inv.mkdir()
        art = inv / "artifact"
        art.mkdir()
        for name, key in [("baseline.pte", "baseline_PTE"), ("ring.pte", "PTE"), ("aoti_cuda_blob.ptd", "PTD")]:
            (art / name).symlink_to(cfg[key])
        (inv / "validation-config.json").write_text(json.dumps(dict(
            model=cfg["model"], artifact=str(art), probe=cfg["probe"],
            lib=cfg["native_lib"], lock=cfg["lock"])))
        with (job / "lifecycle.log").open("x") as log:
            child = subprocess.run([args.python, "-B", "-u",
                str(job / "run_mini81_KV_invariants_v2.py"), str(inv)],
                stdout=log, stderr=subprocess.STDOUT)
        (job / "result.json").write_text(json.dumps(dict(strict_result=result,
            original_numeric_gate_pass=result["original_numeric_gate_pass"],
            lifecycle_returncode=child.returncode, mini_only=True,
            export_control_diagnostic=control_diagnostic,
            export_control_gate_passed=control_diagnostic is None,
            performance_qualification_passed=False, full_model_unchanged=True), indent=2))
        state("mini-original-numerical-and-lifecycle-results-ready",
            numeric_pass=result["original_numeric_gate_pass"], lifecycle_returncode=child.returncode)
        rc = 0
    except BaseException:
        (job / "failure.txt").write_text(traceback.format_exc())
        state("mini-qualification-incomplete-error-preserved")
    finally:
        (job / "exit-receipt.json").write_text(json.dumps(dict(returncode=rc,
            full_model_unchanged=True, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--export-job", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--diagnostic-unmatched-export-controls", action="store_true",
        help="Run unchanged numerical checks on a saved export whose control audit failed; cannot qualify the export")
    main(parser.parse_args())
