"""Exercise one actual CUDA executor through live-owner growth and reclamation.

This is a matched teacher-forced diagnostic, not SiliconBench HTTP throughput.
Each of four fresh processes follows 1/2/4/8/16/1. First-round logits are saved;
performance and sampled memory use the last three rounds. This script never
accepts graph replay or promotes an optimization without independent evidence.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import threading
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        while data := file.read(16 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--CPU-build-progress", type=Path, required=True)
    parser.add_argument("--expected-PTE-sha256", required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--arm", choices=["fixed", "runtime"], required=True)
    parser.add_argument("--role", choices=["mini", "full"], required=True)
    parser.add_argument("--accepted-main-review", type=Path, required=True)
    parser.add_argument("--accepted-mini-review", type=Path)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    accepted = json.loads(args.accepted_main_review.read_text())
    assert (
        accepted["independently_accepted"] and accepted["correctness_protocol_verified"]
    )
    assert [
        accepted[k]
        for k in ["performance_cells", "full_quality_cells", "smoke_quality_cells"]
    ] == [100, 40, 60]
    assert accepted["performance_request_errors"] == 0
    if args.role == "full":
        assert args.accepted_mini_review is not None
        mini = json.loads(args.accepted_mini_review.read_text())
        assert all(
            mini[k]
            for k in [
                "numerical_checks_pass",
                "graph_recapture_replay_pass",
                "physical_growth_and_reclamation_pass",
            ]
        )
    build = json.loads(args.CPU_build_progress.read_text())
    assert build["status"] == "CPU-build-completed"
    assert digest(args.probe) == build["binaries"][str(args.probe)]
    assert digest(args.artifact / "model.pte") == args.expected_PTE_sha256
    assert digest(args.artifact / "aoti_cuda_blob.ptd") == args.expected_weights_sha256
    assert args.probe.is_file() and not args.job_dir.exists()
    args.job_dir.mkdir(parents=True)
    records = []

    def state(status, **fields):
        path = args.job_dir / "progress.pending.json"
        path.write_text(
            json.dumps(
                dict(
                    status=status,
                    controller_pid=os.getpid(),
                    candidate_accepted=False,
                    **fields,
                ),
                indent=2,
            )
            + "\n"
        )
        path.replace(args.job_dir / "progress.json")

    def gpu_pids():
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()

    def sampling(stop, path):
        with path.open("x") as file:
            while not stop.is_set():
                start = time.monotonic_ns()
                sampled = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                file.write(
                    json.dumps(
                        dict(
                            monotonic_ns=start,
                            exit=sampled.returncode,
                            GPU_used_MiB=(
                                float(sampled.stdout.strip())
                                if sampled.returncode == 0
                                else None
                            ),
                        )
                    )
                    + "\n"
                )
                file.flush()
                stop.wait(0.25)

    state("waiting-GPU-lock")
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not gpu_pids(), "Do not overlap any other GPU owner"
        try:
            for round_number in range(1, 5):
                directory = args.job_dir / f"round-{round_number}"
                directory.mkdir()
                stop = threading.Event()
                sampler = threading.Thread(
                    target=sampling,
                    args=(stop, directory / "memory.jsonl"),
                    daemon=True,
                )
                sampler.start()
                command = [
                    str(args.probe),
                    "--pte=" + str(args.artifact / "model.pte"),
                    "--data_path=" + str(args.artifact / "aoti_cuda_blob.ptd"),
                    "--output_dir=" + str(directory),
                    "--max_sessions=21",
                    "--max_context=131072",
                    "--initial_kv_capacity=512",
                ]
                child = None
                try:
                    with (directory / "stderr.log").open("x") as log, (
                        directory / "protocol.jsonl"
                    ).open("x") as protocol:
                        child = subprocess.Popen(
                            command,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=log,
                            text=True,
                            bufsize=1,
                        )
                        selector = selectors.DefaultSelector()
                        selector.register(child.stdout, selectors.EVENT_READ)

                        def receive(timeout=1800):
                            assert selector.select(
                                timeout
                            ), "Timed out waiting for actual executor"
                            line = child.stdout.readline()
                            assert (
                                line
                            ), f"Actual executor terminated with code {child.poll()}"
                            value = json.loads(line)
                            protocol.write(
                                json.dumps(
                                    {
                                        "response": value,
                                        "monotonic_ns": time.monotonic_ns(),
                                    }
                                )
                                + "\n"
                            )
                            protocol.flush()
                            assert not value.get("failed"), value
                            return value

                        ready = receive()
                        assert ready["ready"] and ready["snapshot"]["pid"] == child.pid
                        state(
                            "running-actual-executor",
                            round=round_number,
                            GPU_pid=child.pid,
                        )
                        serial = 0

                        def request(value):
                            nonlocal serial
                            serial += 1
                            value["tag"] = f"command-{serial}"
                            protocol.write(
                                json.dumps(
                                    {
                                        "request": value,
                                        "monotonic_ns": time.monotonic_ns(),
                                    }
                                )
                                + "\n"
                            )
                            protocol.flush()
                            child.stdin.write(json.dumps(value) + "\n")
                            child.stdin.flush()
                            reply = receive()
                            assert reply["op"] == value["op"]
                            return reply

                        def step(ids, tokens, positions, save=False):
                            reply = request(
                                dict(
                                    op="step",
                                    seq_ids=ids,
                                    tokens=tokens,
                                    select=positions,
                                    argmax=True,
                                    save_logits=save and round_number == 1,
                                )
                            )
                            for part in reply.get("logits", []):
                                part["sha256"] = digest(part["file"])
                            return reply

                        def token(owner, position):
                            return 100 + (owner * 101 + position * 37) % 149000

                        lengths = {}
                        owner = request({"op": "new"})["seq_id"]
                        lengths[owner] = 0

                        def fill(owners, target, save=False):
                            last = None
                            while any(lengths[id] < target for id in owners):
                                remaining = [
                                    id for id in owners if lengths[id] < target
                                ]
                                budget = max(1, 2048 // len(remaining))
                                ids, tokens, select = [], [], []
                                for id in remaining:
                                    size = min(budget, target - lengths[id])
                                    ids.extend([id] * size)
                                    tokens.extend(
                                        token(id, p)
                                        for p in range(lengths[id], lengths[id] + size)
                                    )
                                    select.append(len(tokens) - 1)
                                    lengths[id] += size
                                last = step(
                                    ids,
                                    tokens,
                                    select,
                                    save=save
                                    and all(lengths[id] == target for id in owners),
                                )
                            return last

                        # Prefix checkpoint, cloned hit/scratch, cancellation and reuse.
                        fill([owner], 511, save=True)
                        fill([owner], 512, save=True)
                        prefix = request(dict(op="clone", seq_id=owner, upto=512))[
                            "seq_id"
                        ]
                        scratch = request(dict(op="clone", seq_id=prefix, upto=512))[
                            "seq_id"
                        ]
                        lengths[scratch] = 512
                        fill([scratch], 525, save=True)
                        request(dict(op="remove", seq_id=scratch))
                        del lengths[scratch]
                        request(dict(op="remove", seq_id=prefix))
                        if args.arm == "runtime":
                            request(dict(op="compact", spare_columns=0))
                        for boundary in [513, 2047, 2048, 2049, 4097, 8193, 12289]:
                            fill([owner], boundary, save=True)
                        active = [owner]
                        stages = []
                        for index, concurrency in enumerate([1, 2, 4, 8, 16, 1]):
                            if index == 5:
                                survivor = active[5]
                                for id in active:
                                    if id != survivor:
                                        request(dict(op="remove", seq_id=id))
                                        del lengths[id]
                                active = [survivor]
                                if args.arm == "runtime":
                                    request(dict(op="compact", spare_columns=0))
                            else:
                                added = []
                                while len(active) < concurrency:
                                    id = request({"op": "new"})["seq_id"]
                                    lengths[id] = 0
                                    active.append(id)
                                    added.append(id)
                                if added:
                                    fill(added, 12289, save=True)
                            before = request({"op": "metrics"})["snapshot"]
                            start = time.monotonic_ns()
                            for cycle in range(64):
                                tokens = [token(id, lengths[id]) for id in active]
                                last = step(
                                    active,
                                    tokens,
                                    list(range(len(active))),
                                    save=cycle == 63,
                                )
                                for id in active:
                                    lengths[id] += 1
                            end = time.monotonic_ns()
                            after = request({"op": "metrics"})["snapshot"]
                            if args.arm == "runtime":
                                assert after["physical_columns"] == concurrency + 1
                                assert after["live_sequences"] == concurrency
                            stages.append(
                                dict(
                                    stage=index,
                                    active_owners=active.copy(),
                                    concurrency=concurrency,
                                    before=before,
                                    after=after,
                                    decode_steps=64,
                                    teacher_forced_output_tokens=64 * concurrency,
                                    diagnostic_wall_seconds=(end - start) / 1e9,
                                    diagnostic_tokens_per_second=64
                                    * concurrency
                                    / ((end - start) / 1e9),
                                    monotonic_start_ns=start,
                                    monotonic_end_ns=end,
                                )
                            )
                        if args.arm == "runtime":
                            assert (
                                stages[-1]["after"]["live_KV_bytes"]
                                < stages[-2]["after"]["live_KV_bytes"]
                            )
                            assert (
                                stages[-1]["after"]["device_used_bytes"]
                                < stages[-2]["after"]["device_used_bytes"]
                            )
                        child.stdin.close()
                        assert child.wait(timeout=120) == 0
                        selector.close()
                    records.append(
                        dict(
                            round=round_number,
                            ready=ready,
                            stages=stages,
                            protocol_sha256=digest(directory / "protocol.jsonl"),
                        )
                    )
                finally:
                    if child and child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=120)
                    stop.set()
                    sampler.join(timeout=30)
                deadline = time.monotonic() + 120
                while gpu_pids():
                    assert (
                        time.monotonic() < deadline
                    ), "Wait for owned executor teardown"
                    time.sleep(1)
            aggregates = []
            for index, concurrency in enumerate([1, 2, 4, 8, 16, 1]):
                values = [
                    r["stages"][index]["diagnostic_tokens_per_second"] for r in records
                ]
                aggregates.append(
                    dict(
                        stage=index,
                        concurrency=concurrency,
                        mean=statistics.mean(values[1:]),
                        sample_stddev=statistics.stdev(values[1:]),
                        first_discarded=values[0],
                    )
                )
            result = dict(
                arm=args.arm,
                role=args.role,
                rounds=records,
                performance=aggregates,
                probe_sha256=digest(args.probe),
                model_PTE_sha256=digest(args.artifact / "model.pte"),
                weights_blob_sha256=digest(args.artifact / "aoti_cuda_blob.ptd"),
                physical_growth_and_reclamation_observed=args.arm == "runtime",
                numerical_pair_independent_review_pending=True,
                graph_recapture_replay_pass=False,
                actual_graph_trace_review_pending=True,
                candidate_accepted=False,
                HTTP_primary_comparison=False,
                limitations="Matched teacher-forced actual executor diagnostic; times include sampling/completion, IPC and memory snapshots. Quarter-second device sampling can miss transient peaks; it is not SiliconBench HTTP TPS.",
            )
            (args.job_dir / "result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            state("completed-diagnostic-pending-independent-review", rounds=4)
        except Exception as error:
            state("failed", error_type=type(error).__name__, detail=str(error)[-2000:])
            raise


if __name__ == "__main__":
    main()
