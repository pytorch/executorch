"""Qualify an isolated working-set retention experiment on real MG/Gemma minis.

All artifact paths are read-only. Measurements use the existing four-round
protocol, device argmax, selected logits and controlled actual token counts.
No full models or combined optimization are launched by this controller.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "source-scripts", "mg-audit", "gemma-audit",
                 "output", "data-output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.data_output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    commands = []
    manifest = dict(experiment="prefill-working-set-retention", full_models_tested=False,
                    candidate_accepted=False, runtime_option="ET_CUDA_IDLE_KEEP_PREFILL_WORKING_SET=1",
                    idle_spare_bytes=256 * 2**20, inherited_sources={}, models={})

    def state(status, **extra):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, PID=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_models_tested=False, candidate_accepted=False, **extra), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    def run(label, argv, env=None):
        record = dict(label=label, argv=[str(x) for x in argv])
        commands.append(record)
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (args.output / (label + ".log")).open("x") as log:
            subprocess.run(argv, env=env, check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=10800)

    try:
        copied = args.output / "scripts"
        copied.mkdir()
        for path in args.source_scripts.iterdir():
            if path.is_file() and path.suffix == ".py":
                shutil.copy2(path, copied / path.name)
                manifest["inherited_sources"][str(path)] = sha(path)
        for name in ("run_real_mini_four_arms.py", "run_allocator_lifecycle.py"):
            assert (copied / name).is_file()
        for model, audit_path in (("MG", args.mg_audit), ("Gemma", args.gemma_audit)):
            audit = json.loads(audit_path.read_text())
            assert audit["model"] == model and audit["arm"] == "baseline"
            assert audit["retained_original_layers"] == list(range(4 if model == "MG" else 6))
            assert audit["prefill_budget"] == (4096 if model == "MG" else 2048)
            assert audit["BF16_KV"] and audit["metadata"]["get_offgraph_kv_sequence_info_abi"] == 3
            manifest["models"][model] = dict(audit=str(audit_path), audit_SHA256=sha(audit_path),
                                            GGUF_SHA256=audit["GGUF_SHA256"], artifacts=audit["artifacts"])
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        state("building-working-set-candidate")
        native = args.output / "native"
        run("build", [str(args.python), "-B", str(Path(__file__).with_name("build_prefill_workset_retention.py")),
                      "--parent", str(args.parent), "--output", str(native), "--lock", str(args.lock)])
        assert json.loads((native / "native-gate.json").read_text())["native_tests_passed"]
        env = dict(os.environ, ET_CUDA_IDLE_KEEP_PREFILL_WORKING_SET="1")
        env.pop("ET_CUDA_IDLE_SPARE_BYTES", None)
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        jobs = {}
        for model, audit in (("MG", args.mg_audit), ("Gemma", args.gemma_audit)):
            for stage, script, extra in (
                ("pair", "run_real_mini_four_arms.py", ["--arms", "allocator"]),
                ("measure", "run_allocator_lifecycle.py", ["--measure"]),
                ("memory", "run_allocator_lifecycle.py", []),
                ("graph", "run_allocator_lifecycle.py", ["--graph-trace"]),
            ):
                state("running-real-mini", model=model, stage=stage)
                label = model + "-" + stage
                destination = args.data_output / label
                argv = [str(args.python), "-B", "-u", str(copied / script),
                        "--native-preparation", str(native), "--baseline-audit", str(audit),
                        "--output", str(destination), "--lock", str(args.lock), *extra]
                if script == "run_allocator_lifecycle.py":
                    argv += ["--executable-directory", str(copied)]
                run(label, argv, env=env)
                assert (destination / "result.json").is_file()
                jobs[label] = dict(path=str(destination),
                                   result_SHA256=sha(destination / "result.json"))
                (args.output / "completed-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        state("complete-data-independent-review-required", jobs=jobs,
              qualification="No acceptance until correctness, cold/repeated burst latency and memory are reviewed")
    except BaseException as error:
        state("failed-preserve-evidence", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
