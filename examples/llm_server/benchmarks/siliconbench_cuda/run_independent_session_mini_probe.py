"""Exercise geometry-checked mini arms with heterogeneous session histories.

Model numerical comparison and graph trace review remain independent gates.
No full-model artifacts or main-result replacement are accepted here.
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
from independent_session_mini_profiles import PROFILES, DEFAULT_PROFILE, FULL_PROFILES, validate_mini_audit, validate_full_audit


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
    parser.add_argument("--native-gate", type=Path, required=True)
    parser.add_argument("--expected-PTE-sha256", required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--arm", choices=["global", "independent"], required=True)
    parser.add_argument("--mini-profile", choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument('--full-model-profile',choices=FULL_PROFILES,
                        help='Validate actual full geometry rather than a mini artifact')
    parser.add_argument("--automatic-reclaim", action="store_true",
                        help="Require native automatic contraction; never issue compact operations")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--same-artifact-pitch-control", action="store_true",
                        help="Establish declared pitch in both arms before runtime-only compaction")
    parser.add_argument('--diagnostic-initial-KV-capacity',type=int,choices=[512,16384],
                        help='Same independent PTE allocation control; never a main preset')
    parser.add_argument(
        "--nsys",
        type=Path,
        help="Profile only discarded round 1 for actual CUDA graph evidence",
    )
    args = parser.parse_args()
    assert args.diagnostic_initial_KV_capacity is None or args.arm=='independent'
    initial_capacity=args.diagnostic_initial_KV_capacity or 512
    build = json.loads(args.native_gate.read_text())
    assert build['native_GPU_tests_pass'] and build['test_count'] >= 29
    assert digest(args.probe) == build['binaries'][str(args.probe)]
    assert digest(args.artifact / 'model.pte') == args.expected_PTE_sha256
    assert digest(args.artifact / 'aoti_cuda_blob.ptd') == args.expected_weights_sha256
    audit_name='independent-full-artifact-audit.json' if args.full_model_profile else 'independent-mini-artifact-audit.json'
    audit = json.loads((args.artifact / audit_name).read_text())
    geometry = (validate_full_audit(audit,args.full_model_profile,args.arm=='independent') if args.full_model_profile else validate_mini_audit(audit, args.mini_profile, args.arm == 'independent'))
    if args.full_model_profile:
        assert args.mini_profile==('gemma4-first6' if args.full_model_profile=='gemma4-48layers' else DEFAULT_PROFILE)
    assert audit['source_spec_sha256'] == build['source_spec_SHA256']
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
                    "--initial_kv_capacity="+str(initial_capacity),
                ]
                if args.nsys and round_number == 1:
                    command = [
                        str(args.nsys),
                        "profile",
                        "--trace=cuda,nvtx",
                        "--cuda-graph-trace=graph",
                        "--cuda-memory-usage=true",
                        "--cuda-flush-interval=1000",
                        "--sample=none",
                        "--cpuctxsw=none",
                        "-o",
                        str(directory / "cuda-graphs"),
                        *command,
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
                            while True:
                                line = child.stdout.readline()
                                assert (
                                    line
                                ), f"Actual executor terminated with code {child.poll()}"
                                try:
                                    value = json.loads(line)
                                except ValueError:
                                    log.write(line)
                                    log.flush()
                                    assert selector.select(
                                        timeout
                                    ), "Profiler output without executor reply"
                                    continue
                                if isinstance(value, dict) and (
                                    "snapshot" in value or "failed" in value
                                ):
                                    break
                                log.write(line)
                                log.flush()
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
                        assert ready["ready"]
                        executor_pid = ready["snapshot"]["pid"]
                        if executor_pid != child.pid:
                            assert args.nsys and round_number == 1
                            ancestor = executor_pid
                            for _ in range(16):
                                status = (
                                    Path("/proc") / str(ancestor) / "status"
                                ).read_text()
                                ancestor = int(
                                    next(
                                        line.split()[1]
                                        for line in status.splitlines()
                                        if line.startswith("PPid:")
                                    )
                                )
                                if ancestor == child.pid:
                                    break
                            assert (
                                ancestor == child.pid
                            ), "Profiled executor must be owned descendant"
                        state(
                            "running-actual-executor",
                            round=round_number,
                            GPU_pid=executor_pid,
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
                                budget = max(1, geometry['max_step'] // len(remaining))
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
                        extra_owners = []
                        if args.same_artifact_pitch_control:
                            extra_owners = [request({"op": "new"})["seq_id"] for _ in range(20)]
                        fill([owner], 511, save=True)
                        if args.same_artifact_pitch_control:
                            assert request({"op":"metrics"})["snapshot"]["physical_columns"] == 22
                            for extra in extra_owners:
                                request(dict(op="remove", seq_id=extra))
                            if args.arm == "independent" and not args.automatic_reclaim:
                                request(dict(op="compact", spare_columns=0))
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
                        if args.arm == "independent" and not args.automatic_reclaim:
                            request(dict(op="compact", spare_columns=0))
                        boundaries = [513, 2047, 2048, 2049, 4097, 8193, 12289]
                        if args.mini_profile == 'gemma4-first6':
                            boundaries = [513, 1023, 1024, 1025, 2047, 2048, 2049, 4097, 8193, 12289]
                        for boundary in boundaries:
                            fill([owner], boundary, save=True)
                            if boundary == 8193:
                                # Clone actual wrapped model caches, including a
                                # checkpoint one token behind the current owner.
                                prefix = request(
                                    dict(op="clone", seq_id=owner, upto=8192)
                                )["seq_id"]
                                scratch = request(
                                    dict(op="clone", seq_id=prefix, upto=8192)
                                )["seq_id"]
                                lengths[scratch] = 8192
                                fill([scratch], 8205, save=True)
                                request(dict(op="remove", seq_id=scratch))
                                del lengths[scratch]
                                request(dict(op="remove", seq_id=prefix))
                                if args.arm == "independent" and not args.automatic_reclaim:
                                    request(dict(op="compact", spare_columns=0))
                        active = [owner]
                        stages = []
                        for index, concurrency in enumerate([1, 2, 4, 8, 16, 1]):
                            lifecycle_start = time.monotonic_ns()
                            if index == 5:
                                survivor = active[5]
                                for id in active:
                                    if id != survivor:
                                        request(dict(op="remove", seq_id=id))
                                        del lengths[id]
                                active = [survivor]
                                if args.arm == "independent" and not args.automatic_reclaim:
                                    request(dict(op="compact", spare_columns=0))
                            else:
                                added = []
                                while len(active) < concurrency:
                                    id = request({"op": "new"})["seq_id"]
                                    lengths[id] = 0
                                    active.append(id)
                                    added.append(id)
                                if added:
                                    for added_id in added:
                                        fill([added_id], 511 + added_id * 67, save=True)
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
                            if args.arm == "independent":
                                assert after["physical_columns"] == concurrency + 1
                                assert after["live_sequences"] == concurrency
                            stages.append(
                                dict(
                                    stage=index,
                                    active_owners=active.copy(),
                                    concurrency=concurrency,
                                    before=before,
                                    after=after,
                                    session_lengths={str(id):lengths[id] for id in active},
                                    decode_steps=64,
                                    teacher_forced_output_tokens=64 * concurrency,
                                    diagnostic_wall_seconds=(end - start) / 1e9,
                                    diagnostic_tokens_per_second=64
                                    * concurrency
                                    / ((end - start) / 1e9),
                                    lifecycle_start_ns=lifecycle_start,
                                    lifecycle_end_ns=time.monotonic_ns(),
                                    monotonic_start_ns=start,
                                    monotonic_end_ns=end,
                                )
                            )
                        if args.arm == "independent":
                            assert (
                                stages[-1]["after"]["live_KV_bytes"]
                                < stages[-2]["after"]["live_KV_bytes"]
                            )
                            assert (
                                stages[-1]["after"]["device_used_bytes"]
                                < stages[-2]["after"]["device_used_bytes"]
                            )
                        # Separate post-shrink numerical/graph checks for all seven
                        # static methods. Padded rows must use the current dummy column.
                        method_checks = []
                        survivor = active[0]
                        for count, width in [
                            (1, 1),
                            (2, 2),
                            (3, 4),
                            (7, 8),
                            (13, 16),
                            (32, 32),
                            (63, 64),
                        ]:
                            # A never-used method requires three eager warmups,
                            # then capture, then an independent replay. Methods
                            # recaptured after KV movement need fewer calls.
                            for replay in range(5):
                                before = request({"op": "metrics"})["snapshot"]
                                result = step(
                                    [survivor] * count,
                                    [
                                        token(survivor, lengths[survivor] + i)
                                        for i in range(count)
                                    ],
                                    [count - 1],
                                    save=replay == 4,
                                )
                                lengths[survivor] += count
                                after = result["snapshot"]
                                previous = {
                                    row["method"]: row["calls"]
                                    for row in before["method_calls"]
                                }
                                delta = {
                                    row["method"]: row["calls"]
                                    - previous.get(row["method"], 0)
                                    for row in after["method_calls"]
                                    if row["calls"] != previous.get(row["method"], 0)
                                }
                                assert delta == {f"forward_{width}": 1}, delta
                                method_checks.append(
                                    dict(
                                        actual_tokens=count,
                                        static_width=width,
                                        replay_index=replay,
                                        method_delta=delta,
                                        snapshot=after,
                                    )
                                )
                        child.stdin.close()
                        # nsys finalizes its trace after the actual executor exits.
                        assert (
                            child.wait(
                                timeout=600 if args.nsys and round_number == 1 else 120
                            )
                            == 0
                        )
                        selector.close()
                    records.append(
                        dict(
                            round=round_number,
                            ready=ready,
                            stages=stages,
                            post_shrink_method_checks=method_checks,
                            trace_requested=bool(args.nsys and round_number == 1),
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
            sampled_memory = []
            for record in records:
                directory = args.job_dir / f"round-{record['round']}"
                samples = [
                    json.loads(line)
                    for line in (directory / "memory.jsonl").read_text().splitlines()
                ]
                measured = [row["GPU_used_MiB"] for row in samples if row["exit"] == 0]
                assert measured and all(row["exit"] == 0 for row in samples)
                sampled_memory.append(
                    dict(
                        round=record["round"],
                        samples=len(samples),
                        whole_process_sampled_peak_MiB=max(measured),
                        raw_samples_sha256=digest(directory / "memory.jsonl"),
                    )
                )
            peak_values = [
                row["whole_process_sampled_peak_MiB"] for row in sampled_memory
            ]
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
                mini_profile=args.mini_profile,
                full_model_profile=args.full_model_profile,
                model_geometry=geometry,
                same_artifact_pitch_control=args.same_artifact_pitch_control,
                initial_KV_capacity=initial_capacity,
                same_independent_PTE_initial_capacity_diagnostic=args.diagnostic_initial_KV_capacity is not None,
                heterogeneous_histories=True,
                automatic_reclaim_requested=args.automatic_reclaim,
                native_gate_sha256=digest(args.native_gate),
                role="full-qualification" if args.full_model_profile else "mini",
                rounds=records,
                performance=aggregates,
                probe_sha256=digest(args.probe),
                model_PTE_sha256=digest(args.artifact / "model.pte"),
                weights_blob_sha256=digest(args.artifact / "aoti_cuda_blob.ptd"),
                sampled_memory_rounds=sampled_memory,
                whole_process_sampled_peak_MiB=dict(
                    mean=statistics.mean(peak_values[1:]),
                    sample_stddev=statistics.stdev(peak_values[1:]),
                    first_discarded=peak_values[0],
                ),
                physical_growth_and_reclamation_observed=args.arm == "independent",
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
