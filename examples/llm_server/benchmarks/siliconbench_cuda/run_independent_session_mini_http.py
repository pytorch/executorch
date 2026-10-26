"""Measure both mini artifacts through the official Muse Glimmer HTTP server.

Retain named sessions while concurrency grows; delete the long owner before
the final c1 step. Native logs and NVML samples retain allocation evidence.
Natural generated lengths must be reviewed before performance qualification.
"""

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

from trace_gemma_runtime_http_memory import NVML


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-job", "python", "tokenizer", "lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--wait-minutes", type=int, default=0)
    parser.add_argument("--memory-worker-proof", type=Path)
    args = parser.parse_args()
    assert 1024 <= args.port <= 65535
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    deadline = time.monotonic() + 60 * args.wait_minutes
    while True:
        preparation = json.loads((args.preparation_job / "progress.json").read_text())
        if preparation["status"] == "mini-artifacts-ready":
            break
        assert preparation["status"] != "failed", preparation
        handle = Path("/proc") / str(preparation["controller_pid"])
        assert handle.exists(), "Preparation handle missing; do not restart"
        command = (handle / "cmdline").read_bytes()
        assert (b"prepare_independent_session_mini.py" in command or
                b"export_independent_session_mini_from_native.py" in command)
        (args.output / "progress.json").write_text(json.dumps(dict(
            status="waiting-on-confirmed-live-mini-preparation", controller_pid=os.getpid(),
            preparation_pid=preparation["controller_pid"], preparation_status=preparation["status"],
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), candidate_accepted=False), indent=2) + "\n")
        if time.monotonic() >= deadline:
            (args.output / "progress.json").write_text(json.dumps(dict(
                status="observation-timeout", controller_pid=os.getpid(),
                preparation_still_running=True, preparation_pid=preparation["controller_pid"]), indent=2) + "\n")
            return
        time.sleep(30)
    assert preparation["status"] == "mini-artifacts-ready"
    gate_path = args.preparation_job / "native-gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["native_GPU_tests_pass"] and gate["test_count"] >= 29
    source = Path(gate["source"])
    worker = args.preparation_job / "model-build/muse_glimmer_batching_worker"
    assert digest(worker) == gate["binaries"][str(worker)]
    memory_proof = None
    if args.memory_worker_proof:
        memory_proof = json.loads(args.memory_worker_proof.read_text())
        assert memory_proof["native_gate_SHA256"] in (
            digest(gate_path), gate.get("imported_native_gate_SHA256"))
        assert memory_proof["original_worker_SHA256"] == digest(worker)
        assert memory_proof["source"] == gate["source"]
        assert memory_proof["official_control_flow_preserved"] and memory_proof["no_inference_kernel_change"]
        for name, expected in memory_proof["unchanged_SDK_libraries"].items():
            assert digest(Path(name)) == expected, name
        worker = Path(memory_proof["instrumented_worker"])
        assert digest(worker) == memory_proof["instrumented_worker_SHA256"]
    for name, expected in gate["unchanged_controls"].items():
        assert digest(source / name) == expected, name
    assert args.python.is_file() and (args.tokenizer / "tokenizer.json").is_file()
    cache_config = (source / "extension/llm/cache/cache.h").read_text()
    assert "int initial_capacity = 512;" in cache_config
    factory = (source / "examples/models/muse-glimmer/runtime/backend_factory.cpp").read_text()
    assert "CudaExecutorConfig executor_config;" in factory
    explicit_initial = "executor_config.initial_capacity" in factory
    if explicit_initial:
        assert "executor_config.initial_capacity = config.initial_kv_capacity;" in factory
        bootstrap = (source / "examples/models/muse-glimmer/runtime/runners/batching_bootstrap.cpp").read_text()
        assert 'DEFINE_int32(initial_kv_capacity, -1,' in bootstrap
        assert "backend.initial_kv_capacity = FLAGS_initial_kv_capacity;" in bootstrap
        serve = (source / "examples/models/muse-glimmer/serving/serve_batching.py").read_text()
        assert '"--initial-kv-capacity"' in serve
    artifacts, audits = {}, {}
    for arm in ("global", "independent"):
        artifact = args.preparation_job / ("mini-" + arm + "-artifact")
        audit = json.loads((artifact / "independent-mini-artifact-audit.json").read_text())
        assert audit["mini_is_not_full_model"] and audit["retained_original_layer_indices"] == [0, 1, 2, 3]
        assert audit["runtime_columns"] and audit["independent_capacity"] == (arm == "independent")
        assert audit["source_spec_sha256"] == gate["source_spec_SHA256"]
        for name in ("model.pte", "aoti_cuda_blob.ptd"):
            assert digest(artifact / name) == audit["artifacts"][name]["sha256"]
        artifacts[arm], audits[arm] = artifact, audit

    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           candidate_accepted=False, full_model_export_started=False,
                                           **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    def http(path, payload=None, method=None, timeout=600):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request("http://127.0.0.1:" + str(args.port) + path,
                                     data=body, method=method,
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    records = []
    state("waiting-GPU-lock")
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                            "--format=csv,noheader,nounits"], text=True).strip()
        nvml = NVML()
        for number in range(1, 5):
            for prefix in ("off", "on"):
                order = ("global", "independent") if number % 2 else ("independent", "global")
                for arm in order:
                    with socket.socket() as check:
                        assert check.connect_ex(("127.0.0.1", args.port)) != 0, "Preserve existing port owner"
                    directory = args.output / ("round-" + str(number)) / prefix / arm
                    directory.mkdir(parents=True)
                    stage = "before-loading"
                    stop = threading.Event()
                    sampling_errors = []
                    samples = directory / "memory.jsonl"

                    def sample():
                        try:
                            with samples.open("x") as file:
                                while not stop.is_set():
                                    file.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),
                                                               stage=stage, GPU_used_bytes=nvml.used())) + "\n")
                                    file.flush()
                                    stop.wait(.05)
                        except BaseException as error:
                            sampling_errors.append(str(error))

                    sampler = threading.Thread(target=sample, daemon=True)
                    sampler.start()
                    artifact = artifacts[arm]
                    command = [str(args.python), "-B", "-m",
                               "executorch.examples.models.muse_glimmer.serving.serve_batching",
                               "--worker-bin", str(worker), "--model-path", str(artifact / "model.pte"),
                               "--data-path", str(artifact / "aoti_cuda_blob.ptd"),
                               "--tokenizer-path", str(args.tokenizer), "--hf-tokenizer", str(args.tokenizer),
                               "--backend", "cuda", "--model-id", "mg-four-layer-mini",
                               "--max-context", "131072", "--max-sessions", "16",
                               "--max-decode-sequences", "16", "--max-inflight-requests", "16",
                               "--prefix-cache-entries", "4" if prefix == "on" else "0",
                               "--prefix-checkpoint-interval", "512" if prefix == "on" else "0",
                               "--tool-parser", "none", "--host", "127.0.0.1", "--port", str(args.port)]
                    if explicit_initial:
                        command.extend(["--initial-kv-capacity", "512"])
                    (directory / "launch.json").write_text(json.dumps(dict(argv=command), indent=2) + "\n")
                    env = {**os.environ, "PYTHONPATH": str(source.parent), "CUDA_VISIBLE_DEVICES": "0"}
                    env.pop("ET_CUDA_BENCH_MEMORY_TRACE", None)
                    if memory_proof:
                        env["ET_CUDA_BENCH_MEMORY_TRACE"] = "1"
                    child = None
                    try:
                        with (directory / "server.log").open("x") as log, (directory / "requests.jsonl").open("x") as raw:
                            stage = "loading"
                            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                     env=env, start_new_session=True, stdin=subprocess.DEVNULL)
                            state("starting-official-HTTP-server", round=number, prefix=prefix, arm=arm,
                                  server_pid=child.pid)
                            deadline = time.monotonic() + 1800
                            while True:
                                assert child.poll() is None, "Server terminated; inspect its preserved log"
                                try:
                                    with http("/health", timeout=3) as response:
                                        assert json.load(response)["status"] == "ok"
                                    break
                                except Exception:
                                    assert time.monotonic() < deadline, "HTTP readiness timeout"
                                    stop.wait(2)
                            stage = "ready"
                            stop.wait(2)
                            active, stage_records = [], []
                            raw_lock = threading.Lock()

                            def request(barrier, owner):
                                count = 12200 if owner == 0 else 128 + owner * 43
                                content = (" orange" * count if owner == 0 else " blue" * count)
                                content += "\nContinue with the word green."
                                body = dict(model="mg-four-layer-mini", session_id="mini-" + str(owner),
                                            messages=[dict(role="user", content=content)],
                                            stream=True, stream_options=dict(include_usage=True),
                                            max_tokens=64, temperature=0, top_p=1)
                                barrier.wait()
                                started = time.monotonic_ns()
                                events, first_token = [], None
                                with http("/v1/chat/completions", body) as response:
                                    for line in response:
                                        text = line.decode().strip()
                                        if text == "data: [DONE]":
                                            break
                                        if not text.startswith("data: "):
                                            continue
                                        event = json.loads(text[6:])
                                        assert "error" not in event, event
                                        events.append(event)
                                        choices = event.get("choices", [])
                                        if choices and choices[0].get("delta", {}).get("content") and first_token is None:
                                            first_token = time.monotonic_ns()
                                ended = time.monotonic_ns()
                                usages = [event["usage"] for event in events if event.get("usage")]
                                assert len(usages) == 1 and usages[0]["completion_tokens"] > 0, events
                                result = dict(owner=owner, body=body, events=events, usage=usages[0],
                                              start_ns=started, end_ns=ended,
                                              first_content_ns=first_token,
                                              seconds=(ended - started) / 1e9,
                                              TTFT_seconds=(first_token - started) / 1e9 if first_token else None)
                                with raw_lock:
                                    raw.write(json.dumps(result) + "\n")
                                    raw.flush()
                                return {key: value for key, value in result.items() if key not in ("body", "events")}

                            for index, concurrency in enumerate([1, 2, 4, 8, 16, 1]):
                                if index == 5:
                                    survivor = 5
                                    stage = "delete-long-and-short-owners"
                                    for owner in active:
                                        if owner != survivor:
                                            with http("/v1/sessions/mini-" + str(owner), method="DELETE") as response:
                                                assert json.load(response)["closed"]
                                    active = [survivor]
                                    stage = "after-deletion-before-safe-forward"
                                    stop.wait(2)
                                else:
                                    active = list(range(concurrency))
                                stage = "stage-" + str(index) + "-c" + str(concurrency) + "-active"
                                state("running-HTTP-lifecycle", round=number, prefix=prefix, arm=arm,
                                      concurrency=concurrency, server_pid=child.pid)
                                barrier = threading.Barrier(concurrency)
                                started = time.monotonic_ns()
                                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                                    results = list(pool.map(lambda owner: request(barrier, owner), active))
                                ended = time.monotonic_ns()
                                tokens = sum(row["usage"]["completion_tokens"] for row in results)
                                stage_records.append(dict(stage=index, concurrency=concurrency,
                                                          active_owners=active.copy(), requests=results,
                                                          start_ns=started, end_ns=ended,
                                                          output_tokens=tokens,
                                                          aggregate_TPS=tokens / ((ended - started) / 1e9)))
                                stage = "stage-" + str(index) + "-c" + str(concurrency) + "-idle"
                                stop.wait(3)
                            records.append(dict(round=number, prefix=prefix, arm=arm, stages=stage_records,
                                                requests_sha256=digest(directory / "requests.jsonl")))
                    finally:
                        stage = "shutdown"
                        if child is not None and child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=120)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait(timeout=120)
                        stop.wait(1)
                        stop.set()
                        sampler.join(timeout=10)
                        assert not sampler.is_alive() and not sampling_errors, sampling_errors
                        deadline = time.monotonic() + 120
                        while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                                        "--format=csv,noheader,nounits"], text=True).strip():
                            assert time.monotonic() < deadline, "Owned worker GPU teardown incomplete"
                            time.sleep(1)
                    records[-1]["memory_samples_sha256"] = digest(samples)
                    records[-1]["server_log_sha256"] = digest(directory / "server.log")

    aggregates = []
    for prefix in ("off", "on"):
        for arm in ("global", "independent"):
            selected = [row for row in records if row["prefix"] == prefix and row["arm"] == arm]
            assert [row["round"] for row in selected] == [1, 2, 3, 4]
            for index, concurrency in enumerate([1, 2, 4, 8, 16, 1]):
                values = [row["stages"][index]["aggregate_TPS"] for row in selected]
                aggregates.append(dict(prefix=prefix, arm=arm, stage=index, concurrency=concurrency,
                                       mean_TPS=statistics.mean(values[1:]),
                                       sample_SD_TPS=statistics.stdev(values[1:]), first_discarded_TPS=values[0]))
    result = dict(actual_official_HTTP_server=True, mini_only=True, context=131072,
                  initial_KV_capacity=512, concurrency_sequence=[1, 2, 4, 8, 16, 1],
                  prefix_modes=["off", "on"], prefix_entries_on=4, prefix_checkpoint_interval_on=512,
                  four_rounds_first_discarded=True, alternating_arm_order=True,
                  NVML_source="entire device, sampled every 50 ms", rounds=records, aggregates=aggregates,
                  artifact_audits=audits, native_gate_sha256=digest(gate_path),
                  worker_sha256=digest(worker), candidate_promoted=False,
                  initial_KV_set_by="explicit supported CLI" if explicit_initial else "verified native default",
                  memory_worker_proof=memory_proof,
                  native_allocator_sampling_enabled=memory_proof is not None,
                  workload_equivalence_and_memory_review_pending=True,
                  model_performance_qualified=False,
                  limitations="Named-session lifecycle diagnostic. Natural outputs are retained, not forced. "
                              "Prefix ON retains checkpoint owners after named-session deletion. "
                              "This does not replace full-model SiliconBench or task F1.")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    state("HTTP-lifecycle-completed-pending-independent-review", server_runs=len(records))
    print(json.dumps(dict(server_runs=len(records), candidate_promoted=False)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            if output.is_dir():
                (output / "failure.json").write_text(json.dumps(dict(
                    status="failed", controller_pid=os.getpid(), error=str(error),
                    traceback=traceback.format_exc()), indent=2) + "\n")
        raise
