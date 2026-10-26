"""Measure actual mini prefill/decode/mixed calls with matched teacher forcing.

Use synchronized native intervals before snapshots and IPC. Cold/recapture and
KV-growth calls remain in raw data and are reported separately from stable calls.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time
import traceback
from independent_session_mini_profiles import PROFILES, DEFAULT_PROFILE, validate_mini_audit
from qualified_rewind_probe import qualified_rewind_probe


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-job", "output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wait-minutes", type=int, default=240)
    parser.add_argument('--mini-profile', choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument('--rewind-probe-job',type=Path)
    parser.add_argument('--native-sampler',choices=['off','on'],default='off')
    parser.add_argument('--nsys',type=Path,
                        help='Trace round1 only; rounds2-4 remain unprofiled for performance')
    parser.add_argument('--diagnostic-initial-KV-capacity',type=int,choices=(512,16384),default=512,
                        help='Numerical/performance allocation control only; default512 is unchanged')
    args = parser.parse_args()
    assert args.native_sampler=='off' or args.rewind_probe_job is not None
    assert not args.output.exists()
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           candidate_accepted=False, **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    records = []
    try:
        deadline = time.monotonic() + 60 * args.wait_minutes
        while True:
            progress = json.loads((args.preparation_job / "progress.json").read_text())
            if progress["status"] == "mini-artifacts-ready":
                break
            assert progress["status"] != "failed", progress
            pid = progress["controller_pid"]
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            assert (b"prepare_independent_session_mini.py" in command or
                    b"export_independent_session_mini_from_native.py" in command or
                    b"prepare_independent_session_shared_autotune_mini.py" in command or
                    (args.mini_profile == 'gemma4-first6' and
                     b"prepare_gemma_independent_session_mini_pair.py" in command))
            state("waiting-on-confirmed-live-mini-preparation", preparation_pid=pid)
            if time.monotonic() >= deadline:
                state("observation-timeout", preparation_pid=pid, preparation_still_running=True)
                return
            time.sleep(30)
        gate_path = args.preparation_job / "native-gate.json"
        gate = json.loads(gate_path.read_text())
        assert gate["native_GPU_tests_pass"] and gate["test_count"] >= 29
        probe = args.preparation_job / "model-build/muse_glimmer_runtime_kv_probe"
        assert digest(probe) == gate["binaries"][str(probe)]
        probe_proof=None
        if args.rewind_probe_job:
            probe,probe_proof=qualified_rewind_probe(args.rewind_probe_job,gate_path,probe,state,args.wait_minutes)
        audits = {}
        for arm in ("global", "independent"):
            artifact = args.preparation_job / ("mini-" + arm + "-artifact")
            audit = json.loads((artifact / "independent-mini-artifact-audit.json").read_text())
            geometry = validate_mini_audit(audit, args.mini_profile, arm == 'independent')
            assert audit["source_spec_sha256"] == gate["source_spec_SHA256"]
            assert audit["independent_capacity"] == (arm == "independent")
            for name in ("model.pte", "aoti_cuda_blob.ptd"):
                assert digest(artifact / name) == audit["artifacts"][name]["sha256"]
            audits[arm] = audit
        cases = [("decode" + str(c), [1] * c) for c in (1,2,4,8,16)]
        cases += [("prefill64", [64]), ("prefill512", [512]),
                  ("decode8-prefill64", [1]*8+[64]),
                  ("decode8-prefill512", [1]*8+[512])]
        if geometry['max_step'] == 512:
            # Keep this qualification a single mixed forward, within the
            # actual artifact limit. The MG roofline's520 case is separate.
            cases[-1] = ('decode8-prefill504', [1]*8+[504])
        state("waiting-GPU-lock")
        with args.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                                "--format=csv,noheader,nounits"], text=True).strip()
            for number in range(1,5):
                for arm in (("global","independent") if number%2 else ("independent","global")):
                    directory = args.output / ("round-" + str(number)) / arm
                    directory.mkdir(parents=True)
                    artifact = args.preparation_job / ("mini-" + arm + "-artifact")
                    argv = [str(probe), "--pte=" + str(artifact / "model.pte"),
                            "--data_path=" + str(artifact / "aoti_cuda_blob.ptd"),
                            "--output_dir=" + str(directory), "--max_sessions=21",
                            "--max_context=131072", "--initial_kv_capacity="+str(args.diagnostic_initial_KV_capacity)]
                    native_argv=list(argv)
                    trace=bool(args.nsys and number==1)
                    if trace:
                        assert args.nsys.is_file()
                        argv=[str(args.nsys),'profile','--trace=cuda,nvtx',
                              '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                              '--force-overwrite=false','--output='+str(directory/'cold-profile'),*argv]
                    (directory/'executor-command.json').write_text(json.dumps(dict(
                        command=argv,native_command=native_argv,profiled_discarded_round=trace,
                        initial_KV_capacity=args.diagnostic_initial_KV_capacity),indent=2)+'\n')
                    child = None
                    try:
                        with (directory / "stderr.log").open("x") as log, (directory / "protocol.jsonl").open("x") as raw:
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
                            env.pop('ET_CUDA_BENCH_MEMORY_TRACE',None)
                            if args.native_sampler=='on':env['ET_CUDA_BENCH_MEMORY_TRACE']='1'
                            child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                     stderr=log, text=True, bufsize=1,env=env)
                            selector = selectors.DefaultSelector()
                            selector.register(child.stdout, selectors.EVENT_READ)

                            def receive():
                                until=time.monotonic()+1800
                                while True:
                                    assert selector.select(max(0,until-time.monotonic())), "Native executor observation timeout"
                                    line = child.stdout.readline()
                                    assert line, "Native executor terminated: " + str(child.poll())
                                    try: response=json.loads(line)
                                    except ValueError:
                                        log.write(line);log.flush();continue
                                    break
                                raw.write(json.dumps(dict(response=response)) + "\n")
                                raw.flush()
                                assert not response.get("failed"), response
                                return response

                            ready = receive()
                            assert ready["ready"]
                            native_pid=ready["snapshot"]["pid"]
                            if not trace: assert native_pid==child.pid
                            else:
                                assert (Path('/proc')/str(native_pid)/'cmdline').read_bytes().split(b'\0')[0]==str(probe).encode()
                            state("running-matched-mini-model-latency", round=number, arm=arm,
                                  GPU_pid=native_pid,launcher_pid=child.pid,profiled_discarded_round=trace)
                            serial = 0

                            def request(body):
                                nonlocal serial
                                serial += 1
                                body["tag"] = "latency-" + str(serial)
                                raw.write(json.dumps(dict(request=body)) + "\n")
                                raw.flush()
                                child.stdin.write(json.dumps(body) + "\n")
                                child.stdin.flush()
                                response = receive()
                                assert response["op"] == body["op"]
                                return response

                            def token(owner, position):
                                return 100 + (owner*101 + position*37) % 149000

                            case_records = []
                            for label, widths in cases:
                                owners = [request(dict(op="new"))["seq_id"] for _ in widths]
                                positions = [0] * len(owners)
                                ids = [owner for owner in owners for _ in range(1024)]
                                values = [token(i,p) for i in range(len(owners)) for p in range(1024)]
                                # Native execute_raw splits this teacher-forced setup at the exported width.
                                for offset in range(0,len(ids),geometry['max_step']):
                                    size = min(geometry['max_step'],len(ids)-offset)
                                    request(dict(op="step",seq_ids=ids[offset:offset+size],
                                                 tokens=values[offset:offset+size],select=[size-1],argmax=True))
                                positions = [1024] * len(owners)
                                before = request(dict(op="metrics"))["snapshot"]
                                calls = []
                                for cycle in range(16):
                                    seq_ids, tokens, select = [], [], []
                                    for i,(owner,width) in enumerate(zip(owners,widths)):
                                        seq_ids.extend([owner]*width)
                                        tokens.extend(token(i,p) for p in range(positions[i],positions[i]+width))
                                        select.append(len(tokens)-1)
                                        positions[i] += width
                                    reply = request(dict(op="step",seq_ids=seq_ids,tokens=tokens,select=select,
                                                         argmax=True,save_logits=number==1 and cycle==15))
                                    current = reply["snapshot"]
                                    previous_counts = {row['method']:row['calls'] for row in before['method_calls']}
                                    delta = {row['method']:row['calls']-previous_counts.get(row['method'],0)
                                             for row in current['method_calls']
                                             if row['calls'] != previous_counts.get(row['method'],0)}
                                    method = 'forward_'+str(next((m for m in (1,2,4,8,16,32,64) if m >= sum(widths)), 'others'))
                                    assert delta == {method:1}, (label, delta)
                                    changed = any(current[key] != before[key] for key in
                                                  ("live_KV_bytes","row_growth_count","column_resize_count"))
                                    calls.append(dict(cycle=cycle,native_seconds=reply["elapsed_seconds"],
                                                      storage_changed=changed, before=before,after=current,
                                                      selected_argmax=reply["argmax"],logits=reply["logits"],
                                                      actual_method=method, actual_model_forwards=1))
                                    before = current
                                stable = [call["native_seconds"] for call in calls[4:] if not call["storage_changed"]]
                                assert stable, "No stable calls; preserve and inspect raw growth data"
                                case_records.append(dict(case=label,widths=widths,history_per_owner=1024,
                                                         calls=calls,cold_first_four_seconds=sum(call["native_seconds"] for call in calls[:4]),
                                                         mean_all_measured_native_ms=1000*statistics.mean(call["native_seconds"] for call in calls[4:]),
                                                         mean_stable_native_ms=1000*statistics.mean(stable),stable_calls=len(stable)))
                                for owner in owners:
                                    request(dict(op="remove",seq_id=owner))
                            child.stdin.close()
                            assert child.wait(timeout=120) == 0
                            selector.close()
                        if trace:
                            report=directory/'cold-profile.nsys-rep';assert report.is_file()
                            with (directory/'nsys-export.log').open('x') as out:
                                subprocess.run([str(args.nsys),'export','--type=sqlite',
                                    '--output='+str(directory/'cold-profile.sqlite'),str(report)],
                                    check=True,stdout=out,stderr=subprocess.STDOUT,timeout=300)
                        records.append(dict(round=number,arm=arm,cases=case_records,
                                            protocol_sha256=digest(directory / "protocol.jsonl"),
                                            native_PID=native_pid,actual_command_SHA256=digest(directory/'executor-command.json'),
                                            profiled_discarded_round=trace))
                        (args.output / "results.partial.json").write_text(json.dumps(records,indent=2) + "\n")
                    finally:
                        if child is not None and child.poll() is None:
                            child.terminate()
                            try: child.wait(timeout=120)
                            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=120)
                        deadline = time.monotonic() + 120
                        while subprocess.check_output(["nvidia-smi","--query-compute-apps=pid",
                                                        "--format=csv,noheader,nounits"],text=True).strip():
                            assert time.monotonic() < deadline, "Owned executor teardown incomplete"
                            time.sleep(1)
        aggregates = []
        for arm in ("global","independent"):
            selected = [row for row in records if row["arm"] == arm]
            assert [row["round"] for row in selected] == [1,2,3,4]
            for index,(label,widths) in enumerate(cases):
                aggregates.append(dict(arm=arm,case=label,widths=widths,statistics={
                    key: dict(mean=statistics.mean(row["cases"][index][key] for row in selected[1:]),
                              sample_SD=statistics.stdev(row["cases"][index][key] for row in selected[1:]),
                              first_discarded=selected[0]["cases"][index][key])
                    for key in ("cold_first_four_seconds","mean_all_measured_native_ms","mean_stable_native_ms")}))
        (args.output / "result.json").write_text(json.dumps(dict(
            mini_only=True,mini_profile=args.mini_profile,model_geometry=geometry,
            rewind_probe_proof=probe_proof,native_sampler_mode=args.native_sampler,
            initial_KV_capacity=args.diagnostic_initial_KV_capacity,
            actual_profiled_rounds=[row['round'] for row in records if row['profiled_discarded_round']],
            performance_rounds_unprofiled=all(not row['profiled_discarded_round'] for row in records if row['round']>1),
            rounds=records,aggregates=aggregates,artifact_audits=audits,
            native_gate_sha256=digest(gate_path),probe_sha256=digest(probe),
            fixed_teacher_forced_inputs=True,native_intervals_exclude_IPC_and_snapshots=True,
            native_intervals_include_host_preparation_argmax_and_CUDA_synchronization=True,
            four_rounds_first_discarded=True,alternating_arm_order=True,
            numerical_and_route_review_pending=True,model_performance_qualified=False,
            candidate_promoted=False),indent=2) + "\n")
        state("model-latency-collected-pending-independent-review",cases=9,processes=len(records))
    except Exception as error:
        state("failed",error=str(error),traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
