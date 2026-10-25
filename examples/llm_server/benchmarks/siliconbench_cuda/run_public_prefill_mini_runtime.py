"""Compare original/public MG through the actual CUDA executor API.

Every measured forward starts with exactly 1K private history for only its
active owners. History rebuilding and saved-logit copies are untimed. This is
four-layer feasibility evidence by default. A 52-layer trial still requires
separate full-model F1 and HTTP acceptance.
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
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while raw := f.read(8 * 1024 * 1024):
            h.update(raw)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workflow-job', type=Path, required=True)
    p.add_argument('--CPU-build-progress', type=Path, required=True)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--accepted-main-review', type=Path, required=True)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--nsys', type=Path)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repetitions', type=int, default=12)
    p.add_argument('--expected-n-layers', type=int, choices=[4, 52], default=4)
    p.add_argument('--runtime-sessions', type=int, choices=[16, 32], default=32)
    p.add_argument('--history-tokens', type=int, choices=[0,1024], default=1024,
                   help='Zero-history numerical control distinguishes direct kernel differences from changed cached history; default is the original 1K protocol')
    p.add_argument('--workflow-pair-filename', default='mini-pair-exports.json')
    p.add_argument('--numerical-only', action='store_true',
                   help='Collect first-pair complete logits; no performance acceptance or timing averages')
    p.add_argument('--continue-after-numerical-screen-failure', action='store_true',
                   help='Retain the failed RMS screen and collect broader F1 elsewhere; never promote here')
    a = p.parse_args()
    assert a.warmup >= 5 and a.repetitions >= 4 and not a.job_dir.exists()
    a.job_dir.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    def state(status, **fields):
        temp = a.job_dir / 'progress.pending.json'
        temp.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            mini_model=a.expected_n_layers != 52, full_model_or_HTTP_accepted=False,
            new_KA=False, **fields), indent=2) + '\n')
        temp.replace(a.job_dir / 'progress.json')
    state('waiting-existing-mini-exports')
    try:
        accepted = json.loads(a.accepted_main_review.read_text())
        assert accepted['independently_accepted'] and accepted['correctness_protocol_verified']
        assert [accepted[k] for k in ['performance_cells', 'full_quality_cells', 'smoke_quality_cells']] == [100, 40, 60]
        deadline = time.monotonic() + 20000
        assert a.workflow_pair_filename in ['mini-pair-exports.json', 'model-pair-exports.json']
        while not (a.workflow_job / a.workflow_pair_filename).exists():
            s = json.loads((a.workflow_job / 'progress.json').read_text())
            assert s['status'] != 'failed', s
            os.kill(int(s['controller_pid']), 0)
            assert time.monotonic() < deadline, 'Observation expired; do not restart exports'
            time.sleep(15)
        pair = json.loads((a.workflow_job / a.workflow_pair_filename).read_text())
        build = json.loads(a.CPU_build_progress.read_text())
        assert build['status'] == 'CPU-build-completed'
        probe = Path(next(k for k in build['binaries'] if Path(k).name == 'muse_glimmer_runtime_kv_probe'))
        assert sha(probe) == build['binaries'][str(probe)]
        paths = {}
        for arm in ['original', 'public']:
            paths[arm] = Path(pair['artifacts'][arm]['path'])
            audit = pair['artifacts'][arm]['audit']
            assert audit['n_layers'] == a.expected_n_layers and audit['logical_context'] == 131072
            if a.expected_n_layers == 52:
                assert a.runtime_sessions == 16 and audit['max_kv_seqs'] == 21
            assert audit['static_forwards'] == [1, 2, 4, 8, 16, 32, 64]
            assert audit['selected_logits'] == [64, 202048]
            for name, item in audit['artifacts'].items():
                assert sha(paths[arm] / name) == item['SHA256']
        assert pair['artifacts']['original']['audit']['GGUF_SHA256'] == pair['artifacts']['public']['audit']['GGUF_SHA256']
        weight_review = a.job_dir / 'same-weight-payload-review.json'
        subprocess.run([sys.executable, '-B', str(package / 'review_runtime_kv_weight_payload.py'),
            '--source', str(a.et_source), '--fixed', str(paths['original'] / 'aoti_cuda_blob.ptd'),
            '--runtime', str(paths['public'] / 'aoti_cuda_blob.ptd'), '--output', str(weight_review)], check=True)
        assert json.loads(weight_review.read_text())['all_quantized_and_nonquantized_storage_sha256_equal']
        largest_decode = a.runtime_sessions
        mixes = [('d1', 1, 0), ('d8', 8, 0), (f'd{largest_decode}', largest_decode, 0),
                 ('p64', 0, 64), ('p128', 0, 128), ('p512', 0, 512),
                 ('p2048', 0, 2048), ('d8p64', 8, 64), ('d8p512', 8, 512)]
        records = []
        def review_numeric():
            import numpy as np
            numeric = []
            for index, (name, _, _) in enumerate(mixes):
                left_record = next(r for r in records if r['round'] == 1 and r['arm'] == 'original')['results'][index]['numeric']
                right_record = next(r for r in records if r['round'] == 1 and r['arm'] == 'public')['results'][index]['numeric']
                assert len(left_record['logits']) == len(right_record['logits']) == 1
                left, right = left_record['logits'][0], right_record['logits'][0]
                assert left['shape'] == right['shape'] and left['selected_indices'] == right['selected_indices']
                lv = np.fromfile(left['file'], dtype='<f4'); rv = np.fromfile(right['file'], dtype='<f4')
                assert np.isfinite(lv).all() and np.isfinite(rv).all()
                rms = float(np.sqrt(np.mean((lv.astype(np.float64)-rv)**2)) / max(np.sqrt(np.mean(lv.astype(np.float64)**2)), 1e-8))
                numeric.append(dict(mix=name, relative_RMS=rms,
                    max_abs=float(np.max(np.abs(lv-rv))),
                    argmax_identical=left_record['argmax'] == right_record['argmax'],
                    numerical_pass=rms < .005))
            return numeric
        state('waiting-GPU-lock')
        with a.lock.open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            for round_number in ([1] if a.numerical_only else range(1, 5)):
                order = ['original', 'public'] if round_number % 2 else ['public', 'original']
                for arm in order:
                    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                    directory = a.job_dir / f'round-{round_number}-{arm}'
                    directory.mkdir()
                    command = [str(probe), '--pte=' + str(paths[arm] / 'model.pte'),
                        '--data_path=' + str(paths[arm] / 'aoti_cuda_blob.ptd'),
                        '--output_dir=' + str(directory), '--max_sessions=' + str(a.runtime_sessions),
                        '--max_context=131072', '--initial_kv_capacity=512']
                    if a.nsys and round_number == 1:
                        command = [str(a.nsys), 'profile', '--trace=cuda,nvtx',
                            '--cuda-graph-trace=graph', '--sample=none', '--cpuctxsw=none',
                            '-o', str(directory / 'actual-CUDA'), *command]
                    with (directory / 'stderr.log').open('x') as log, (directory / 'protocol.jsonl').open('x') as protocol:
                        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log, text=True, bufsize=1)
                        selector = selectors.DefaultSelector()
                        selector.register(child.stdout, selectors.EVENT_READ)
                        state('running-actual-mini-executor', round=round_number, arm=arm, child_pid=child.pid)
                        serial = 0
                        def receive():
                            while True:
                                assert selector.select(1800), 'Actual executor reply timeout'
                                line = child.stdout.readline()
                                assert line, f'Actual executor stopped: {child.poll()}'
                                try:
                                    reply = json.loads(line)
                                except ValueError:
                                    log.write(line); log.flush(); continue
                                if not isinstance(reply, dict) or not ('snapshot' in reply or 'failed' in reply):
                                    log.write(line); log.flush(); continue
                                assert not reply.get('failed'), reply
                                return reply
                        def request(body):
                            nonlocal serial
                            serial += 1; body = dict(body, tag=f'command-{serial}')
                            before = time.monotonic_ns()
                            protocol.write(json.dumps(dict(request=body, monotonic_ns=before)) + '\n'); protocol.flush()
                            child.stdin.write(json.dumps(body) + '\n'); child.stdin.flush()
                            reply = receive(); after = time.monotonic_ns()
                            assert reply['op'] == body['op']
                            protocol.write(json.dumps(dict(response=reply, monotonic_ns=after)) + '\n'); protocol.flush()
                            return reply, (after - before) / 1e9, body['tag']
                        try:
                            ready = receive(); assert ready['ready']
                            protocol.write(json.dumps(dict(response=ready, monotonic_ns=time.monotonic_ns())) + '\n')
                            owners = []
                            results = []
                            def token(ordinal, pos):
                                return 100 + (ordinal * 101 + pos * 37) % 149000
                            def setup(count):
                                for owner in owners:
                                    request(dict(op='remove', seq_id=owner))
                                owners.clear()
                                for ordinal in range(count):
                                    reply, _, _ = request(dict(op='new')); owner = reply['seq_id']; owners.append(owner)
                                    for start in range(0,a.history_tokens,512):
                                        request(dict(op='step', seq_ids=[owner] * 512,
                                            tokens=[token(ordinal, pos) for pos in range(start, start + 512)],
                                            select=[], argmax=False, save_logits=False))
                                snap, _, _ = request(dict(op='metrics'))
                                assert snap['snapshot']['live_sequences'] == count
                            for name, decode, prefill in mixes:
                                count = decode + int(prefill > 0)
                                def forward(save=False):
                                    ids, tokens, selected = [], [], []
                                    for ordinal in range(decode):
                                        ids.append(owners[ordinal]); tokens.append(token(ordinal, a.history_tokens)); selected.append(len(tokens)-1)
                                    if prefill:
                                        ids.extend([owners[decode]] * prefill)
                                        tokens.extend(token(decode, pos) for pos in range(a.history_tokens, a.history_tokens + prefill))
                                        selected.append(len(tokens)-1)
                                    before, _, _ = request(dict(op='metrics'))
                                    reply, wall, tag = request(dict(op='step', seq_ids=ids, tokens=tokens,
                                        select=selected, argmax=True, save_logits=save))
                                    old = {m['method']: m['calls'] for m in before['snapshot']['method_calls']}
                                    delta = {m['method']: m['calls']-old[m['method']] for m in reply['snapshot']['method_calls']}
                                    assert sum(delta.values()) == 1, 'Unexpected raw-forward split'
                                    width = 1
                                    while width < len(tokens): width *= 2
                                    expected = f'forward_{width}' if len(tokens) <= 64 else 'forward_others'
                                    assert delta[expected] == 1
                                    assert len(reply['argmax']) == count
                                    return dict(tag=tag, executor_seconds=reply['elapsed_seconds'], API_IPC_seconds=wall,
                                        actual_method_call_deltas=delta, argmax=reply['argmax'],
                                        logits=reply['logits'], snapshot=reply['snapshot'])
                                setup(count); numeric = forward(save=round_number == 1)
                                for _ in range(0 if a.numerical_only else a.warmup): setup(count); forward()
                                samples = []
                                for _ in range(0 if a.numerical_only else a.repetitions): setup(count); samples.append(forward())
                                assert all(s['argmax'] == numeric['argmax'] for s in samples), 'Rebuilt identical histories must produce stable greedy outputs'
                                results.append(dict(mix=name, active_owners=count, decode=decode, prefill=prefill,
                                    history_tokens=a.history_tokens, numeric=numeric, timed=samples,
                                    mean_executor_seconds=statistics.mean(s['executor_seconds'] for s in samples) if samples else None,
                                    mean_API_IPC_seconds=statistics.mean(s['API_IPC_seconds'] for s in samples) if samples else None))
                            child.stdin.close(); assert child.wait(timeout=300) == 0
                        finally:
                            if child.poll() is None:
                                child.terminate()
                                try: child.wait(timeout=120)
                                except subprocess.TimeoutExpired: child.kill(); child.wait()
                            selector.close()
                    records.append(dict(round=round_number, arm=arm, ready=ready,
                        results=results, protocol_SHA256=sha(directory / 'protocol.jsonl')))
                    if round_number == 1 and len(records) == 2:
                        preflight = dict(raw=records, numerical=review_numeric(),
                            numerical_gate=.005, full_model_F1_or_HTTP_accepted=False,
                            model_promoted=False, first_discarded_round_only=True,
                            numerical_only=a.numerical_only,
                            screen_failure_retained=a.continue_after_numerical_screen_failure)
                        (a.job_dir / 'numerical-preflight.json').write_text(json.dumps(preflight, indent=2)+'\n')
                        if not a.continue_after_numerical_screen_failure:
                            assert all(item['numerical_pass'] for item in preflight['numerical']), preflight['numerical']
        # Changed kernels may change accumulation order: retain the established
        # 0.5% RMS numerical gate; report greedy differences instead of hiding them.
        numeric = review_numeric()
        performance = []
        for index, (name, _, _) in ([] if a.numerical_only else enumerate(mixes)):
            arms = {}
            for arm in ['original', 'public']:
                values = [r['results'][index]['mean_executor_seconds'] for r in records if r['arm'] == arm]
                arms[arm] = dict(last3_mean_seconds=statistics.mean(values[1:]),
                    last3_sample_SD_seconds=statistics.stdev(values[1:]), first_discarded=values[0])
            performance.append(dict(mix=name, arms=arms,
                public_over_original=arms['public']['last3_mean_seconds']/arms['original']['last3_mean_seconds']))
        report = dict(four_rounds_first_discarded=not a.numerical_only, raw=records, numerical=numeric,
            n_layers=a.expected_n_layers, runtime_sessions=a.runtime_sessions,
            numerical_only=a.numerical_only,history_tokens=a.history_tokens,
            screen_failure_retained=a.continue_after_numerical_screen_failure,
            mini_numeric_pass=all(r['numerical_pass'] for r in numeric),
            performance=performance, actual_executor_API=True,
            same_weight_payload_review_SHA256=sha(weight_review),
            numerical_gate=.005, full_model_F1_or_HTTP_accepted=False,
            model_promoted=False, timing='Reported run_step + greedy + CUDA completion; history rebuild and logit saves are untimed. Includes CPU API overhead, not GPU-only or HTTP TTFT.')
        (a.job_dir / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
        if not a.continue_after_numerical_screen_failure:
            assert report['mini_numeric_pass'], numeric
        state('completed-mini-runtime-pending-full-model-validation')
    except Exception as e:
        state('failed', error_type=type(e).__name__, detail=str(e)[-2000:])
        raise


if __name__ == '__main__':
    main()
