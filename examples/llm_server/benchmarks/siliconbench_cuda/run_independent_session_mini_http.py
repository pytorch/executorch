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
from independent_session_mini_profiles import DEFAULT_PROFILE, PROFILES, validate_mini_audit


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
    parser.add_argument('--mini-profile',choices=PROFILES,default=DEFAULT_PROFILE)
    parser.add_argument('--full-model-profile', choices=['gemma4-48layers'])
    parser.add_argument('--full-native-job',type=Path)
    parser.add_argument('--Gemma-source-preparation-job',type=Path)
    parser.add_argument('--http-python-dependencies-job',type=Path,
                        help='Verified isolated HTTP dependencies; no change to model environment')
    parser.add_argument("--disable-native-memory-sampling", action="store_true",
                        help="Use the verified memory worker binary with its sampler disabled")
    args = parser.parse_args()
    gemma=args.mini_profile=='gemma4-first6'
    assert not gemma or (args.Gemma_source_preparation_job is not None and args.memory_worker_proof is not None)
    assert not args.disable_native_memory_sampling or args.memory_worker_proof is not None
    assert 1024 <= args.port <= 65535
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    full_profile=args.full_model_profile
    if full_profile:
        from independent_session_full_http_binding import verified_gemma_full_http
        verified=verified_gemma_full_http(args,package)
        gate_path,gate,source,worker=(verified[k] for k in ('gate_path','gate','source','worker'))
        memory_proof,dependency_path,Gemma_model_source_binding=(verified[k] for k in
            ('memory_proof','dependency_path','Gemma_model_source_binding'))
        artifacts,audits,explicit_initial=(verified[k] for k in ('artifacts','audits','explicit_initial'))
    else:
        deadline = time.monotonic() + 60 * args.wait_minutes
        while True:
            preparation = json.loads((args.preparation_job / "progress.json").read_text())
            if preparation["status"] == "mini-artifacts-ready":
                break
            assert preparation["status"] != "failed", preparation
            handle = Path("/proc") / str(preparation["controller_pid"])
            try:
                command = (handle / "cmdline").read_bytes()
            except FileNotFoundError:
                command = b''
            confirmed = (b"prepare_independent_session_mini.py" in command or
                    b"export_independent_session_mini_from_native.py" in command or
                    b"prepare_independent_session_shared_autotune_mini.py" in command or
                    (gemma and b"prepare_gemma_independent_session_mini_pair.py" in command))
            if not confirmed:
                # The exporter can commit its terminal state and exit between
                # our state read and /proc read. Re-read authoritative state;
                # never restart or accept a reused PID based on stale metadata.
                refreshed = json.loads((args.preparation_job / "progress.json").read_text())
                if refreshed['status'] == 'mini-artifacts-ready':
                    preparation = refreshed
                    break
                assert refreshed['status'] != 'failed', refreshed
                raise RuntimeError('Preparation handle no longer matches its running state; do not restart')
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
        package=Path(__file__).resolve().parent
        worker = args.preparation_job / "model-build/muse_glimmer_batching_worker"
        assert digest(worker) == gate["binaries"][str(worker)]
        memory_proof = None
        if args.memory_worker_proof:
            memory_proof = json.loads(args.memory_worker_proof.read_text())
            root_gate_path=source.parent/'native-gate.json'
            root_gate_SHA256=None
            if root_gate_path.is_file():
                root_gate=json.loads(root_gate_path.read_text())
                for key in ('source','source_spec_SHA256','native_tests_xml_SHA256','unchanged_controls'):
                    assert root_gate[key]==gate[key],key
                for name,expected in root_gate['binaries'].items():
                    assert gate['binaries'][name]==expected and digest(Path(name))==expected,name
                root_gate_SHA256=digest(root_gate_path)
            assert memory_proof["native_gate_SHA256"] in (
                digest(gate_path), gate.get("imported_native_gate_SHA256"),
                root_gate_SHA256)
            if not gemma:
                assert memory_proof["original_worker_SHA256"] == digest(worker)
            else:
                assert memory_proof['existing_Gemma_llm_server_adapter_preserved']
                assert memory_proof['scheduler_stops_tokenizer_transport_unchanged']
                assert memory_proof['original_source_and_SDK_unchanged']
                assert memory_proof['original_Gemma_worker_source_SHA256']==digest(package/'gemma_runtime_columns_worker.cpp')
            assert memory_proof["source"] == gate["source"]
            assert (gemma or memory_proof["official_control_flow_preserved"]) and memory_proof["no_inference_kernel_change"]
            for name, expected in memory_proof["unchanged_SDK_libraries"].items():
                assert digest(Path(name)) == expected, name
            worker = Path(memory_proof["instrumented_worker"])
            assert digest(worker) == memory_proof["instrumented_worker_SHA256"]
        for name, expected in gate["unchanged_controls"].items():
            assert digest(source / name) == expected, name
        dependency_path = None
        if args.http_python_dependencies_job:
            dependency_job=args.http_python_dependencies_job
            assert json.loads((dependency_job/'progress.json').read_text())['status']==(
                'Gemma-HTTP-Python-dependencies-ready' if gemma else 'MG-HTTP-Python-dependencies-ready')
            dependency_proof=json.loads((dependency_job/'proof.json').read_text())
            dependency_path=Path(dependency_proof['dependency_path'])
            assert dependency_path.resolve().is_relative_to(dependency_job.resolve())
            assert dependency_proof['installed_only_into_new_owned_target']
            assert dependency_proof['base_torch_triton_versions_unchanged'] and dependency_proof['native_SDK_and_models_unchanged']
            assert dependency_proof['official_server_import_pass']
            if gemma:
                assert dependency_proof['server_source_SHA256']==digest(package/'serve_gemma_runtime_columns.py')
                assert dependency_proof['Gemma_source_preparation_SHA256']==digest(args.Gemma_source_preparation_job/'source-preparation.json')
            else:
                assert dependency_proof['native_gate_SHA256']==root_gate_SHA256
                assert dependency_proof['server_source_SHA256']==digest(source/'examples/models/muse-glimmer/serving/serve_batching.py')
            for name,expected in dependency_proof['files_SHA256'].items():
                path=dependency_path/name
                assert path.resolve().is_relative_to(dependency_path.resolve()) and digest(path)==expected,name
        Gemma_model_source_binding = None
        if gemma:
            from prepare_gemma_canonical_tokenizer import FILES
            for name,(expected,count) in FILES.items():
                assert (args.tokenizer/name).stat().st_size==count and digest(args.tokenizer/name)==expected,name
            receipt_path=args.Gemma_source_preparation_job/'source-preparation.json'
            receipt=json.loads(receipt_path.read_text())
            receipt_SHA256=digest(receipt_path)
            if receipt['original_native_gate']['source']==gate['source']:
                assert receipt_SHA256==gate['Gemma_source_preparation_SHA256']
                Gemma_model_source_binding=dict(original_model_and_native_source_binding=True)
            else:
                # The allocator control reuses the exact compiled Gemma models
                # against a new ABI-compatible native pool. Its Python export
                # receipt still describes the preserved original model source.
                growth_path=package/'independent_session_growth_allocator_source.json'
                growth=json.loads(growth_path.read_text())
                assert gate['allocator_growth_spec_SHA256']==digest(growth_path)
                assert growth['allocator_policy_only']
                assert growth['weight_math_ABI_graph_invalidation_and_sliding_unchanged']
                expected_pool_SHA256=growth['after_SHA256']
                extra_native_source_hashes={}
                extra_native_original_hashes={}
                source_changes=growth
                if gate.get('stable_fastpath_spec_SHA256'):
                    fast_path=package/'independent_session_stable_repartition_fastpath_source.json'
                    fast=json.loads(fast_path.read_text())
                    assert digest(fast_path)==gate['stable_fastpath_spec_SHA256']
                    assert fast['allocator_growth_spec_SHA256']==digest(growth_path)
                    assert fast['before_SHA256']==growth['after_SHA256'] and fast['file']==growth['file']
                    assert fast['model_weights_kernels_ABI_graph_policy_and_sliding_unchanged']
                    expected_pool_SHA256=fast['after_SHA256']
                    source_changes=dict(patches=[growth,fast])
                if gate.get('stable_metadata_uploads_spec_SHA256'):
                    uploads_path=package/'independent_session_stable_metadata_uploads_source.json'
                    uploads=json.loads(uploads_path.read_text())
                    assert digest(uploads_path)==gate['stable_metadata_uploads_spec_SHA256']
                    assert uploads['stable_fastpath_spec_SHA256']==gate['stable_fastpath_spec_SHA256']
                    assert uploads['native_host_metadata_only'] and uploads['model_weights_kernels_ABI_graph_policy_and_sliding_unchanged']
                    assert source_changes==dict(patches=[growth,fast])
                    source_changes=dict(patches=[growth,fast,uploads])
                    extra_native_source_hashes[uploads['file']]=uploads['after_SHA256']
                    extra_native_original_hashes[uploads['file']]=uploads['before_SHA256']
                if gate.get('flat_migration_copy_spec_SHA256'):
                    copies_path=package/'independent_session_flat_migration_copy_source.json'
                    copies=json.loads(copies_path.read_text())
                    assert digest(copies_path)==gate['flat_migration_copy_spec_SHA256']
                    assert copies['before_SHA256']==expected_pool_SHA256
                    assert copies['file']==growth['file'] and copies['native_copy_scheduling_only']
                    assert copies['model_weights_kernels_ABI_graph_policy_and_sliding_unchanged']
                    assert source_changes==dict(patches=[growth,fast,uploads])
                    source_changes=dict(patches=[growth,fast,uploads,copies])
                    expected_pool_SHA256=copies['after_SHA256']
                if gate.get('empty_owner_reuse_spec_SHA256'):
                    empty_path=package/'independent_session_empty_replacement_source.json'
                    empty=json.loads(empty_path.read_text())
                    assert digest(empty_path)==gate['empty_owner_reuse_spec_SHA256']
                    assert digest(package/'independent_session_empty_replacement.patch')==empty['patch_SHA256']
                    changes=empty['variants']['Gemma-stable-metadata']
                    assert len(changes)==2 and changes[0]['file']==growth['file']
                    assert changes[0]['before_SHA256']==expected_pool_SHA256
                    source_changes=dict(patches=source_changes['patches']+changes)
                    expected_pool_SHA256=changes[0]['after_SHA256']
                    extra_native_source_hashes[changes[1]['file']]=changes[1]['after_SHA256']
                    extra_native_original_hashes[changes[1]['file']]=changes[1]['before_SHA256']
                if 'actual_native_source_change' in gate:
                    assert gate['actual_native_source_change']==source_changes
                    assert gate['preserved_original_native_gate_SHA256']==receipt['original_native_gate_SHA256']
                # Older qualified allocator builds predate those descriptive
                # fields. Verify the actual preserved sources, binaries and XML;
                # never edit their existing receipts or infer compatibility from
                # the common ABI reference alone.
                capacity_path=package/'independent_session_capacity_source.json'
                capacity=json.loads(capacity_path.read_text())
                assert digest(capacity_path)==growth['original_session_source_spec_SHA256']
                original_gate_path=Path(receipt['SDK']).parent.parent/'native-gate.json'
                assert digest(original_gate_path)==receipt['original_native_gate_SHA256']
                original_gate=json.loads(original_gate_path.read_text())
                assert original_gate==receipt['original_native_gate']
                assert gate['test_count']==original_gate['test_count']==74
                for name,record in capacity['files'].items():
                    assert digest(Path(original_gate['source'])/name)==record['after_SHA256'],name
                    expected=expected_pool_SHA256 if name==growth['file'] else record['after_SHA256']
                    if name in extra_native_source_hashes:
                        assert record['after_SHA256']==extra_native_original_hashes[name]
                        expected=extra_native_source_hashes[name]
                    assert digest(source/name)==expected,name
                assert gate['unchanged_controls']==original_gate['unchanged_controls']
                for verified_gate in (gate,original_gate):
                    for name,expected in verified_gate['binaries'].items():assert digest(Path(name))==expected,name
                xml=source.parent/'native-tests.xml'
                assert digest(xml)==gate['native_tests_xml_SHA256']
                import xml.etree.ElementTree as ET
                tests=list(ET.parse(xml).getroot().iter('testcase'))
                assert len(tests)==74 and not any(
                    t.find('failure') is not None or t.find('error') is not None or t.find('skipped') is not None
                    for t in tests)
                assert gate['source_spec_SHA256']==receipt['original_native_gate']['source_spec_SHA256']
                assert digest(source/growth['file'])==expected_pool_SHA256
                original_mini=Path(preparation.get('original_mini_preparation',
                    str((args.preparation_job/'mini-independent-artifact').resolve().parent)))
                assert (args.preparation_job/'mini-global-artifact').resolve().parent==original_mini.resolve()
                original_model_gate=json.loads((original_mini/'native-gate.json').read_text())
                assert original_model_gate['Gemma_source_preparation_SHA256']==receipt_SHA256
                assert original_model_gate['source']==receipt['original_native_gate']['source']
                for arm in ('global','independent'):
                    original=original_mini/('mini-'+arm+'-artifact')
                    artifact=args.preparation_job/('mini-'+arm+'-artifact')
                    audit=json.loads((original/'independent-mini-artifact-audit.json').read_text())
                    assert audit['Gemma_source_preparation_SHA256']==receipt_SHA256
                    for name,record in audit['artifacts'].items():
                        assert digest(original/name)==digest(artifact/name)==record['sha256']
                Gemma_model_source_binding=dict(original_compiled_models_and_Python_source_reused=True,
                    allocator_only_native_control_verified=True,
                    allocator_growth_spec_SHA256=digest(growth_path),
                    actual_native_source_change=source_changes,
                    original_native_gate_SHA256=digest(original_gate_path),
                    actual_native_tests_XML_SHA256=digest(xml),
                    actual_sources_binaries_and_tests_reverified=True,
                    original_receipts_preserved=True,
                    original_model_native_source=original_model_gate['source'],
                    current_native_source=gate['source'],model_source_preparation_SHA256=receipt_SHA256)
            assert receipt['original_source_and_binaries_unchanged']
            source=Path(receipt['new_source'])
            for name,expected in receipt['source_files'].items():assert digest(source/name)==expected,name
        assert args.python.is_file() and (args.tokenizer / "tokenizer.json").is_file()
        cache_config = (source / "extension/llm/cache/cache.h").read_text()
        assert "int initial_capacity = 512;" in cache_config
        factory = (source / "examples/models/muse-glimmer/runtime/backend_factory.cpp").read_text()
        assert "CudaExecutorConfig executor_config;" in factory
        explicit_initial = gemma or "executor_config.initial_capacity" in factory
        if explicit_initial and not gemma:
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
            validate_mini_audit(audit,args.mini_profile,arm=='independent')
            assert audit["source_spec_sha256"] == gate["source_spec_SHA256"]
            for name in ("model.pte", "aoti_cuda_blob.ptd"):
                assert digest(artifact / name) == audit["artifacts"][name]["sha256"]
            artifacts[arm], audits[arm] = artifact, audit
    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           candidate_accepted=False, full_model_export_started=False,
                                           full_model_profile=full_profile,
                                           **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    def http(path, payload=None, method=None, timeout=600):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request("http://127.0.0.1:" + str(args.port) + path,
                                     data=body, method=method,
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    prefix_entries_on=1 if full_profile else 4
    prefix_interval_on=512
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
                    model_id='gemma4-first6-mini' if gemma else 'mg-four-layer-mini'
                    if full_profile:model_id='gemma4-12b-it'
                    command = ([str(args.python),'-B',str(package/'serve_gemma_runtime_columns.py')]
                        if gemma else [str(args.python), "-B", "-m",
                               "executorch.examples.models.muse_glimmer.serving.serve_batching",
                        ]) + ["--worker-bin", str(worker), "--model-path", str(artifact / "model.pte"),
                               "--data-path", str(artifact / "aoti_cuda_blob.ptd"),
                               "--tokenizer-path", str(args.tokenizer), "--hf-tokenizer", str(args.tokenizer),
                               "--model-id", model_id,
                               "--max-context", "131072", "--max-sessions", "16",
                               "--max-decode-sequences", "16", "--max-inflight-requests", "16",
                               "--prefix-cache-entries", str(prefix_entries_on) if prefix == "on" else "0",
                               "--prefix-checkpoint-interval", str(prefix_interval_on) if prefix == "on" else "0",
                               "--tool-parser", "none", "--host", "127.0.0.1", "--port", str(args.port)]
                    if explicit_initial:
                        command.extend(["--initial-kv-capacity", "512"])
                    if not gemma:
                        command.extend(['--backend','cuda'])
                    else:
                        command.extend(['--raw-tool-text'])
                    (directory / "launch.json").write_text(json.dumps(dict(argv=command), indent=2) + "\n")
                    env = {**os.environ, "PYTHONPATH": str(source.parent), "CUDA_VISIBLE_DEVICES": "0"}
                    if dependency_path is not None:
                        env['PYTHONPATH']=str(dependency_path)+os.pathsep+str(source.parent)
                    env.pop("ET_CUDA_BENCH_MEMORY_TRACE", None)
                    if memory_proof and not args.disable_native_memory_sampling:
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
                                body = dict(model=model_id, session_id="mini-" + str(owner),
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
    result = dict(actual_official_HTTP_server=True, mini_only=not full_profile, context=131072,
                  full_model_profile=full_profile, full_model_qualified=False,
                  mini_profile=args.mini_profile,uses_existing_Gemma_llm_server_adapter=gemma,
                  initial_KV_capacity=512, concurrency_sequence=[1, 2, 4, 8, 16, 1],
                  prefix_modes=["off", "on"], prefix_entries_on=prefix_entries_on, prefix_checkpoint_interval_on=prefix_interval_on,
                  four_rounds_first_discarded=True, alternating_arm_order=True,
                  NVML_source="entire device, sampled every 50 ms", rounds=records, aggregates=aggregates,
                  artifact_audits=audits, native_gate_sha256=digest(gate_path),
                  worker_sha256=digest(worker), candidate_promoted=False,
                  initial_KV_set_by="explicit supported CLI" if explicit_initial else "verified native default",
                  memory_worker_proof=memory_proof,Gemma_model_source_binding=Gemma_model_source_binding,
                  native_allocator_sampling_enabled=memory_proof is not None and not args.disable_native_memory_sampling,
                  native_sampler_explicitly_disabled=args.disable_native_memory_sampling,
                  workload_equivalence_and_memory_review_pending=True,
                  model_performance_qualified=False,
                  limitations="Named-session lifecycle diagnostic. Natural outputs are retained, not forced. "
                              "Prefix ON retains checkpoint owners after named-session deletion. "
                              "This does not replace full-model SiliconBench or task F1.",
                  fresh_global_export_is_a_layout_control_not_the_previous_accepted_PTE=bool(full_profile))
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
                failure = dict(
                    status="failed", controller_pid=os.getpid(), error=str(error),
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    traceback=traceback.format_exc())
                (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                pending=output/'progress.pending.json'
                pending.write_text(json.dumps(failure,indent=2)+'\n')
                pending.replace(output/'progress.json')
        raise
