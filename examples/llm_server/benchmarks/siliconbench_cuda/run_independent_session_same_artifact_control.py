"""Compare exact logits before and after lifecycle changes using each unchanged artifact.

All comparisons use the same forward8, weights and teacher-forced tokens.
This isolates cache correctness from independently selected GEMM autotuning.
It does not qualify HTTP throughput or the full models.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import traceback
from independent_session_mini_profiles import PROFILES, DEFAULT_PROFILE, FULL_PROFILES, validate_mini_audit, validate_full_audit
from qualified_rewind_probe import qualified_rewind_probe


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-job", "lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wait-minutes", type=int, default=0)
    parser.add_argument('--mini-profile', choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument('--full-model-profile',choices=FULL_PROFILES)
    parser.add_argument('--arm',choices=('both','global','independent'),default='both',
                        help='Explicit diagnostic scope; default mini/full collection remains both arms')
    parser.add_argument('--rewind-probe-job',type=Path,
                        help='Also verify actual native rewind on each cloned model owner')
    parser.add_argument("--after-logit-collision-job", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           full_model_qualified=False, **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    records = []
    try:
        if args.after_logit_collision_job:
            previous = args.after_logit_collision_job
            until = time.monotonic() + 60 * args.wait_minutes
            while True:
                old = json.loads((previous / "progress.json").read_text())
                handle = Path("/proc") / str(old["controller_pid"])
                if old["status"] == "failed" and not handle.exists():
                    assert "preserve prior logits" in old["error"], old
                    break
                assert handle.exists()
                assert b"run_independent_session_same_artifact_control.py" in (handle / "cmdline").read_bytes()
                state("waiting-on-confirmed-owned-diagnostic", previous_pid=old["controller_pid"])
                if time.monotonic() >= until:
                    state("observation-timeout", previous_still_running=True)
                    return
                time.sleep(20)
        deadline = time.monotonic() + 60 * args.wait_minutes
        while True:
            preparation = json.loads((args.preparation_job / "progress.json").read_text())
            if preparation["status"] == ("full-artifacts-ready-for-qualification" if args.full_model_profile else "mini-artifacts-ready"):
                break
            assert not args.full_model_profile,'Full pair must already be gated and ready'
            assert preparation["status"] != "failed"
            handle = Path("/proc") / str(preparation["controller_pid"])
            assert handle.exists()
            command = (handle / "cmdline").read_bytes()
            assert (b"prepare_independent_session_mini.py" in command or
                    b"export_independent_session_mini_from_native.py" in command or
                    b"prepare_independent_session_shared_autotune_mini.py" in command or
                    (args.mini_profile == 'gemma4-first6' and
                     b"prepare_gemma_independent_session_mini_pair.py" in command))
            state("waiting-on-confirmed-live-preparation", preparation_pid=preparation["controller_pid"])
            if time.monotonic() >= deadline:
                state("observation-timeout", preparation_still_running=True)
                return
            time.sleep(30)
        gate = json.loads((args.preparation_job / "native-gate.json").read_text())
        assert gate["native_GPU_tests_pass"] and gate["test_count"] >= 29
        probe = args.preparation_job / "model-build/muse_glimmer_runtime_kv_probe"
        assert digest(probe) == gate["binaries"][str(probe)]
        probe_proof=None
        if args.rewind_probe_job:
            probe,probe_proof=qualified_rewind_probe(args.rewind_probe_job,
                args.preparation_job/'native-gate.json',probe,state,args.wait_minutes)
        for name, expected in gate["unchanged_controls"].items():
            assert digest(Path(gate["source"]) / name) == expected
        artifacts = {}
        if args.full_model_profile:
            assert preparation['mini_gate_passed'] and preparation['all_full_artifact_hashes_verified']
        artifact_prefix='full-' if args.full_model_profile else 'mini-'
        audit_name='independent-full-artifact-audit.json' if args.full_model_profile else 'independent-mini-artifact-audit.json'
        selected_arms=('global','independent') if args.arm=='both' else (args.arm,)
        for arm in selected_arms:
            artifact = args.preparation_job / (artifact_prefix + arm + "-artifact")
            audit = json.loads((artifact / audit_name).read_text())
            geometry = (validate_full_audit(audit,args.full_model_profile,arm=='independent') if args.full_model_profile else validate_mini_audit(audit, args.mini_profile, arm == 'independent'))
            assert audit["independent_capacity"] == (arm == "independent")
            assert audit["source_spec_sha256"] == gate["source_spec_SHA256"]
            for name in ("model.pte", "aoti_cuda_blob.ptd"):
                assert digest(artifact / name) == audit["artifacts"][name]["sha256"]
            artifacts[arm] = artifact
        import numpy as np

        state("waiting-GPU-lock")
        with args.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                                "--format=csv,noheader"], text=True).strip()
            for number in range(1, 5):
                for arm in (selected_arms if number%2 else tuple(reversed(selected_arms))):
                    directory = args.output / ("round-" + str(number)) / arm
                    directory.mkdir(parents=True)
                    artifact = artifacts[arm]
                    argv = [str(probe), "--pte=" + str(artifact / "model.pte"),
                            "--data_path=" + str(artifact / "aoti_cuda_blob.ptd"),
                            "--output_dir=" + str(directory), "--max_sessions=21",
                            "--max_context=131072", "--initial_kv_capacity=512"]
                    state("running-same-artifact-control", round=number, arm=arm)
                    child = None
                    try:
                        with (directory / "stderr.log").open("x") as log, (directory / "protocol.jsonl").open("x") as protocol:
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
                            env.pop('ET_CUDA_BENCH_MEMORY_TRACE',None)
                            child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                     stderr=log, text=True, bufsize=1,env=env)
                            selector = selectors.DefaultSelector()
                            selector.register(child.stdout, selectors.EVENT_READ)

                            def receive():
                                while True:
                                    assert selector.select(1800), "Owned executor response timeout"
                                    line = child.stdout.readline()
                                    assert line, "Owned executor exited"
                                    try:
                                        reply = json.loads(line)
                                    except ValueError:
                                        log.write(line)
                                        continue
                                    if isinstance(reply, dict) and ("snapshot" in reply or "failed" in reply):
                                        protocol.write(json.dumps(dict(response=reply)) + "\n")
                                        protocol.flush()
                                        assert not reply.get("failed"), reply
                                        return reply
                                    log.write(line)

                            assert receive()["ready"]
                            serial = 0

                            def request(value):
                                nonlocal serial
                                serial += 1
                                value["tag"] = "command-" + str(serial)
                                protocol.write(json.dumps(dict(request=value)) + "\n")
                                protocol.flush()
                                child.stdin.write(json.dumps(value) + "\n")
                                child.stdin.flush()
                                result = receive()
                                assert result["op"] == value["op"]
                                return result

                            def fill(owner, current, target):
                                while current < target:
                                    end = min(target, current + geometry['max_step'])
                                    request(dict(op="step", seq_ids=[owner] * (end - current),
                                                 tokens=[100 + (37 * p) % 149000 for p in range(current, end)],
                                                 select=[end - current - 1], argmax=True, save_logits=False))
                                    current = end

                            def saved_step(owner, history):
                                before = request(dict(op="metrics"))["snapshot"]
                                result = request(dict(op="step", seq_ids=[owner] * 8,
                                                      tokens=[100 + (37 * p) % 149000 for p in range(history, history + 8)],
                                                      select=[7], argmax=True, save_logits=True))
                                counts = {v["method"]: v["calls"] for v in before["method_calls"]}
                                delta = {v["method"]: v["calls"] - counts[v["method"]]
                                         for v in result["snapshot"]["method_calls"]
                                         if v["calls"] != counts[v["method"]]}
                                assert delta == {"forward_8": 1}
                                assert len(result["logits"]) == 1
                                part = result["logits"][0]
                                path = Path(part["file"])
                                assert path.resolve().is_relative_to(directory.resolve())
                                array = np.fromfile(path, dtype="<f4").reshape(part["shape"])
                                assert np.isfinite(array).all()
                                return array, dict(file=str(path), SHA256=digest(path),
                                                   snapshot=result["snapshot"], argmax=result["argmax"])

                            cells = []
                            for history in (512, 8192):
                                owner = request(dict(op="new"))["seq_id"]
                                fill(owner, 0, history)
                                checkpoint = request(dict(op="clone", seq_id=owner, upto=history))["seq_id"]
                                reference, reference_record = saved_step(owner, history)
                                for phase in ("clone", "after-other-owner-growth", "after-owner-deletion"):
                                    if phase == "after-other-owner-growth":
                                        fill(owner, history + 8, 12300)
                                        others = []
                                        for _ in range(14):
                                            other = request(dict(op="new"))["seq_id"]
                                            fill(other, 0, 64)
                                            others.append(other)
                                    elif phase == "after-owner-deletion":
                                        for other in [owner, *others]:
                                            request(dict(op="remove", seq_id=other))
                                    clone = request(dict(op="clone", seq_id=checkpoint, upto=history))["seq_id"]
                                    actual, actual_record = saved_step(clone, history)
                                    assert actual.shape == reference.shape
                                    difference = float(np.abs(actual.astype(np.float64) - reference).max())
                                    equal = bool(np.array_equal(actual, reference))
                                    cells.append(dict(history=history, phase=phase, bitwise_equal=equal,
                                                      max_abs=difference, reference=reference_record, actual=actual_record))
                                    (directory / "checks.partial.json").write_text(json.dumps(cells, indent=2) + "\n")
                                    assert equal, cells[-1]
                                    if probe_proof is not None:
                                        rewound=request(dict(op='rewind',seq_id=clone,position=history))
                                        assert rewound['rewound'] and rewound['position']==history
                                        repeated,repeated_record=saved_step(clone,history)
                                        assert repeated.shape==reference.shape
                                        equal_rewind=bool(np.array_equal(repeated,reference))
                                        cells.append(dict(history=history,phase=phase+'-actual-rewind',
                                            bitwise_equal=equal_rewind,
                                            max_abs=float(np.abs(repeated.astype(np.float64)-reference).max()),
                                            reference=reference_record,actual=repeated_record,
                                            rewind_response=rewound))
                                        (directory/'checks.partial.json').write_text(json.dumps(cells,indent=2)+'\n')
                                        assert equal_rewind,cells[-1]
                                        rejected=request(dict(op='rewind',seq_id=clone,position=131073,
                                                              expect_success=False))
                                        assert not rejected['rewound'] and rejected['position']==history+8
                                    request(dict(op="remove", seq_id=clone))
                                request(dict(op="remove", seq_id=checkpoint))
                            child.stdin.close()
                            assert child.wait(timeout=120) == 0
                            selector.close()
                        records.append(dict(round=number, arm=arm, checks=cells,
                                            protocol_SHA256=digest(directory / "protocol.jsonl"),
                                            PTE_SHA256=digest(artifact / "model.pte")))
                        (args.output / "results.partial.json").write_text(json.dumps(records, indent=2) + "\n")
                    finally:
                        if child is not None and child.poll() is None:
                            child.kill()
                            child.wait(timeout=120)
        (args.output / "result.json").write_text(json.dumps(dict(
            same_artifact_same_forward_same_tokens=True, all_bitwise_checks_pass=True,
            mini_profile=args.mini_profile, full_model_profile=args.full_model_profile, model_geometry=geometry,
            collected_arms=list(selected_arms), default_two_arm_collection=args.arm=='both',
            actual_rewind_checked=probe_proof is not None,rewind_probe_proof=probe_proof,
            independent_exports_not_compared_here=True, CUDA_graph_mode_enabled=True,
            actual_graph_trace_review_separate=True, performance_qualified=False,
            full_model_qualified=False, rounds=records), indent=2) + "\n")
        state("same-artifact-lifecycle-logits-passed", checks=sum(len(r["checks"]) for r in records))
    except Exception as error:
        state("failed", error=str(error)[-3000:], traceback=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
