"""Isolate a mini Gemma input RMS launch configuration against its reference.

Preserve weights, runtime, ring semantics and existing reduction controls.
Export completion is not numerical or performance qualification.
"""

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--baseline-kernel", type=Path, required=True)
    parser.add_argument("--ring-kernel", type=Path, required=True)
    args = parser.parse_args()
    job, template, base = args.job, args.template, args.base
    assert job.is_dir() and not (job / "export_mini81.py").exists()
    rc = 1

    def state(status, **extra):
        pending = job / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, qualified=False,
            mini_only=True, PID=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)))
        pending.replace(job / "progress.json")

    try:
        assert shutil.disk_usage(base).free >= 3 * 2**30
        state("preparing-isolated-input-RMS-control")
        for name in ["reference_generated_reduction_layout.py", "reuse_exact_ptd_export.py"]:
            (job / name).write_bytes((template / name).read_bytes())
        spec = importlib.util.spec_from_file_location("reference_layout", job / "reference_generated_reduction_layout.py")
        layout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layout)
        baseline_source = args.baseline_kernel.read_text()
        assert baseline_source == args.ring_kernel.read_text()
        key = layout.fingerprint(baseline_source)
        assert key == "79d2c45207e650d4bbfabd4f9db5573212d2ff7714a76ff7abf6bd72cfecf658"
        controls = json.loads((template / "Gemma-reference-reduction-layouts.json").read_text())
        assert len(controls) == 4
        assert key not in [row["function_AST_SHA256"] for row in controls]
        row = dict(M=2048, function_AST_SHA256=key,
            generated_function_source=baseline_source,
            kwargs=dict(XBLOCK=8, R0_BLOCK=512), num_warps=4, num_stages=1)
        controls.append(row)
        (job / "Gemma-reference-reduction-layouts.json").write_text(json.dumps(controls, indent=2))
        (job / "input-RMS-control-required.json").write_text(json.dumps(dict(
            control=row, prior_controls_preserved=4, same_generated_function=True,
            source_SHA256=hashlib.sha256(baseline_source.encode()).hexdigest(),
            no_activation_replacement=True), indent=2))

        exporter = (template / "export_mini81.py").read_text().replace(str(template), str(job))
        ast.parse(exporter)
        (job / "export_mini81.py").write_text(exporter)
        argv = json.loads((template / "export-command.json").read_text())
        argv = [arg.replace(str(template), str(job)) for arg in argv]
        cache = job / "export-cache"
        assert not cache.exists()
        cache.mkdir(parents=True)
        argv[argv.index("--cache-root") + 1] = str(cache)
        (job / "export-command.json").write_text(json.dumps(argv, indent=2))
        (job / "data/intermediate-weights").mkdir(parents=True)
        env = dict(os.environ)
        env["ET_DIAGNOSTIC_WEIGHT_STORAGE_DIR"] = str(job / "data/intermediate-weights")
        origin = base / "root-normal-Gemma-source-complete-qualification-20261028-v6"
        env["PYTHONPATH"] = str(origin / "code")
        env["ET_INT4_REFERENCE_EXPORT_CONFIGS"] = str(base / "root-authorized-ring-source-host-pool-validation-20261027-v1/controls/Gemma-baseline-INT4-config-manifest.json")
        for name in ["ET_INT5_REFERENCE_EXPORT_CONFIGS", "ET_INT6_REFERENCE_EXPORT_CONFIGS",
                     "ET_CUDA_RING_JOURNAL_BUFFERS", "ET_CUDA_RING_ARCHIVE_BATCH_ROWS"]:
            env.pop(name, None)
        state("exporting-mini-with-reference-input-RMS-config", cache=str(cache))
        with (job / "export.log").open("x") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)

        receipts = [json.loads(line) for line in (job / "reduction-layout-actual-control-receipt.jsonl").read_text().splitlines()]
        assert any(item["function_AST_SHA256"] == key and item["kwargs"] == row["kwargs"]
                   and item["num_warps"] == 4 for item in receipts), "Input RMS control did not match"
        matched = {item["function_AST_SHA256"] for item in receipts}
        assert all(item["function_AST_SHA256"] in matched for item in controls[:4])
        result = json.loads((job / "data/exports/result.json").read_text())
        assert len(result["arms"]) == 1
        assert result["arms"][0]["metadata"]["get_offgraph_kv_max_seqs"] == 81
        assert result["arms"][0]["metadata"]["get_max_prefill_chunk"] == 2048
        (job / "result.json").write_text(json.dumps(dict(export=result,
            actual_reference_config_receipts=receipts, numerical_qualification_passed=False,
            performance_qualification_passed=False, full_model_unchanged=True), indent=2))
        state("mini-input-RMS-export-complete-qualification-pending")
        rc = 0
    except BaseException:
        (job / "failure.txt").write_text(traceback.format_exc())
        state("mini-input-RMS-export-failed")
    finally:
        (job / "exit-receipt.json").write_text(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            numerical_qualification_passed=False, full_model_unchanged=True)))


if __name__ == "__main__":
    main()
