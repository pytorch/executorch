"""Run the deferred mini/full experiment only after independent main acceptance.

Waiting and orchestration use CPU. Every GPU child is serialized through the
existing task lock. Main servers and artifacts are never changed or promoted.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--CPU-build-progress", type=Path, required=True)
    parser.add_argument("--accepted-main-review", type=Path, required=True)
    parser.add_argument("--native-controls-progress", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--fixed-full-artifact", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument("--completed-CUDA-tests", type=Path,
                        help="Reuse only verified tests for identical binaries and source spec")
    parser.add_argument("--reuse-mini-export-job", type=Path,
                        help="Reuse unchanged audited O2 mini artifacts; rerun both four-round probes")
    args = parser.parse_args()
    assert not args.job_dir.exists()
    args.job_dir.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(args.source.parent),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MAX_JOBS="1",
    )

    def state(status, **fields):
        pending = args.job_dir / "progress.pending.json"
        pending.write_text(
            json.dumps(
                dict(
                    status=status,
                    controller_pid=os.getpid(),
                    candidate_promoted=False,
                    **fields,
                ),
                indent=2,
            )
            + "\n"
        )
        pending.replace(args.job_dir / "progress.json")

    def wait_file(path, accepted):
        while True:
            if path.exists():
                value = json.loads(path.read_text())
                assert value.get("status") not in ["failed", "cancelled", "aborted"]
                if value.get("status") == accepted:
                    return value
            time.sleep(30)

    def run(argv, log_name, gpu=False):
        with (args.job_dir / log_name).open("x") as log:
            command = [str(item) for item in argv]
            if gpu:
                with args.lock.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    assert not subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip()
                    child_env = env | {"CUDA_VISIBLE_DEVICES": "0"}
                    subprocess.run(
                        command,
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=21600,
                    )
                    deadline = time.monotonic() + 120
                    while subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip():
                        assert time.monotonic() < deadline
                        time.sleep(1)
            else:
                subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=21600,
                )

    try:
        state("waiting-independent-main-acceptance", GPU_tests_started=False)
        accepted = wait_file(args.accepted_main_review, "main-independently-accepted")
        assert (
            accepted["independently_accepted"]
            and accepted["correctness_protocol_verified"]
        )
        assert [
            accepted[key]
            for key in [
                "performance_cells",
                "full_quality_cells",
                "smoke_quality_cells",
            ]
        ] == [100, 40, 60]
        assert accepted["performance_request_errors"] == 0
        state("waiting-native-controls-completion", GPU_tests_started=False)
        wait_file(args.native_controls_progress, "completed")
        build = json.loads(args.CPU_build_progress.read_text())
        spec = json.loads((package / "runtime_kv_columns_source.json").read_text())
        assert (
            build["status"] == "CPU-build-completed"
            and build["patch_sha256"] == spec["patch_sha256"]
        )
        for name, record in spec["changed_files"].items():
            assert digest(args.source / name) == record["after_sha256"]
        for path, sha in build["binaries"].items():
            assert digest(Path(path)) == sha
        probe = Path(
            next(
                path
                for path in build["binaries"]
                if Path(path).name == "muse_glimmer_runtime_kv_probe"
            )
        )
        test = Path(
            next(
                path
                for path in build["binaries"]
                if Path(path).name == "test_cuda_kv_cache"
            )
        )
        reused = None
        if args.completed_CUDA_tests:
            reused = json.loads(args.completed_CUDA_tests.read_text())
            assert reused['actual_CUDA_tests_passed']
            assert reused['binary_sha256'] == digest(test)
            assert reused['source_spec_sha256'] == digest(package / 'runtime_kv_columns_source.json')
            state('reusing-identical-verified-CUDA-tests', GPU_tests_started=False)
        else:
            state("running-actual-CUDA-tests", GPU_tests_started=True)
            run([test, "--gtest_color=no"], "CUDA-cache-tests.log", gpu=True)
            run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "unittest",
                    "executorch.backends.cuda.tests.test_offgraph_kv_batched_sequence",
                ],
                "CUDA-Python-tests.log",
                gpu=True,
            )
        (args.job_dir / "CUDA-tests-completed.json").write_text(
            json.dumps(
                dict(
                    actual_CUDA_tests_passed=True,
                    binary_sha256=digest(test),
                    source_spec_sha256=digest(
                        package / "runtime_kv_columns_source.json"
                    ),
                    reused_verified_identical_tests=bool(reused),
                )
            )
            + "\n"
        )

        # The probe's C++ child inherits visibility. Its driver owns the task
        # GPU lock before starting that child; the initial wait stays CPU-only.
        env["CUDA_VISIBLE_DEVICES"] = "0"

        def export(role, mini_review=None):
            if args.reuse_mini_export_job and role in ["mini-fixed", "mini-runtime"]:
                artifact = args.reuse_mini_export_job / role
                audit_path = artifact / "runtime-column-artifact-audit.json"
                audit = json.loads(audit_path.read_text())
                assert audit["role"] == role
                assert audit["source_spec_sha256"] == digest(package / "runtime_kv_columns_source.json")
                assert audit["GGUF_sha256"] == digest(args.gguf)
                assert audit["AOTI_wrapper_optimization"] == "O2"
                assert audit["main_acceptance_proof_sha256"] == digest(args.accepted_main_review)
                for name, record in audit["artifacts"].items():
                    item = artifact / name
                    assert item.stat().st_size == record["bytes"] and digest(item) == record["sha256"]
                (args.job_dir / (role + "-reused-artifact.json")).write_text(
                    json.dumps(dict(artifact=str(artifact), audit_sha256=digest(audit_path),
                                    source_spec_sha256=audit["source_spec_sha256"],
                                    reuse_scope="Unchanged PTE/PTD and model; diagnostic warmup-count correction only",
                                    original_artifacts_modified=False), indent=2) + "\n")
                return artifact
            artifact = args.job_dir / role
            command = [
                sys.executable,
                "-B",
                package / "export_runtime_kv_model.py",
                "--source",
                args.source,
                "--CPU-build-progress",
                args.CPU_build_progress,
                "--accepted-main-review",
                args.accepted_main_review,
                "--gguf",
                args.gguf,
                "--role",
                role,
                "--output",
                artifact,
                "--job-dir",
                args.job_dir / (role + "-export"),
                "--lock",
                args.lock,
            ]
            if mini_review:
                command += ["--accepted-mini-review", mini_review]
            run(command, role + "-export-controller.log")
            return artifact

        def diagnose(role, arm, artifact, mini_review=None):
            job = args.job_dir / (role + "-" + arm + "-diagnostic")
            command = [
                sys.executable,
                "-B",
                package / "run_runtime_kv_probe.py",
                "--probe",
                probe,
                "--artifact",
                artifact,
                "--CPU-build-progress",
                args.CPU_build_progress,
                "--expected-PTE-sha256",
                digest(artifact / "model.pte"),
                "--expected-weights-sha256",
                digest(artifact / "aoti_cuda_blob.ptd"),
                "--arm",
                arm,
                "--role",
                role,
                "--accepted-main-review",
                args.accepted_main_review,
                "--job-dir",
                job,
                "--lock",
                args.lock,
                "--nsys",
                args.nsys,
            ]
            if mini_review:
                command += ["--accepted-mini-review", mini_review]
            run(command, role + "-" + arm + "-diagnostic-controller.log")
            return job

        def review(role, fixed_artifact, runtime_artifact, fixed_job, runtime_job):
            weights = args.job_dir / (role + "-weight-payload-review.json")
            run(
                [
                    sys.executable,
                    "-B",
                    package / "review_runtime_kv_weight_payload.py",
                    "--source",
                    args.source,
                    "--fixed",
                    fixed_artifact / "aoti_cuda_blob.ptd",
                    "--runtime",
                    runtime_artifact / "aoti_cuda_blob.ptd",
                    "--output",
                    weights,
                ],
                role + "-weights-review.log",
            )
            sqlite = runtime_job / "round-1/cuda-graphs.sqlite"
            run(
                [
                    args.nsys,
                    "export",
                    "--type=sqlite",
                    "--output=" + str(sqlite),
                    runtime_job / "round-1/cuda-graphs.nsys-rep",
                ],
                role + "-trace-export.log",
            )
            graph = args.job_dir / (role + "-actual-graph-review.json")
            run(
                [
                    sys.executable,
                    "-B",
                    package / "review_runtime_kv_graph_trace.py",
                    "--runtime-job",
                    runtime_job,
                    "--sqlite",
                    sqlite,
                    "--output",
                    graph,
                ],
                role + "-graph-review.log",
            )
            result = args.job_dir / (role + "-independent-pair-review.json")
            run(
                [
                    sys.executable,
                    "-B",
                    package / "review_runtime_kv_probe_pair.py",
                    "--fixed",
                    fixed_job,
                    "--runtime",
                    runtime_job,
                    "--source-spec",
                    package / "runtime_kv_columns_source.json",
                    "--graph-review",
                    graph,
                    "--weight-payload-review",
                    weights,
                    "--output",
                    result,
                ],
                role + "-pair-review.log",
            )
            return result

        state("exporting-mini-pair", GPU_tests_started=True)
        mini_fixed, mini_runtime = export("mini-fixed"), export("mini-runtime")
        fixed_job = diagnose("mini", "fixed", mini_fixed)
        runtime_job = diagnose("mini", "runtime", mini_runtime)
        mini_review = review("mini", mini_fixed, mini_runtime, fixed_job, runtime_job)
        state(
            "mini-independently-reviewed",
            mini_review=str(mini_review),
            GPU_tests_started=True,
        )
        full_runtime = export("full-runtime", mini_review)
        fixed_job = diagnose("full", "fixed", args.fixed_full_artifact, mini_review)
        runtime_job = diagnose("full", "runtime", full_runtime, mini_review)
        full_review = review(
            "full", args.fixed_full_artifact, full_runtime, fixed_job, runtime_job
        )
        state(
            "full-diagnostic-completed-pending-main-server-review",
            full_review=str(full_review),
            main_results_replaced=False,
            automatic_server_compaction_integrated=False,
        )
    except Exception as error:
        state("failed", error_type=type(error).__name__, detail=str(error)[-1800:])
        raise


if __name__ == "__main__":
    main()
