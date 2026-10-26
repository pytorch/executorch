"""Locate the first numerical divergence with identical fresh-owner token inputs.

Run existing immutable mini artifacts. No source, weights, kernel policy,
acceptance threshold or full-model result is changed. Every requested width is
one actual raw forward, followed by one teacher-forced decode. Keep all four
rounds and their saved FP32 logits for independent review.
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
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-job', 'output', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--baseline-repeat-preparation-job', type=Path,
        help='Compare two existing global-layout exports to isolate compiler variation')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous diagnostics'
    args.output.mkdir(parents=True)
    def state(status, **fields):
        pending = args.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate_accepted=False, full_model_qualified=False, **fields), indent=2)+'\n')
        pending.replace(args.output / 'progress.json')
    try:
        preparation = json.loads((args.preparation_job / 'progress.json').read_text())
        assert preparation['status'] == 'mini-artifacts-ready'
        gate = json.loads((args.preparation_job / 'native-gate.json').read_text())
        assert gate['native_GPU_tests_pass'] and gate['test_count'] == 74
        probe = args.preparation_job / 'model-build/muse_glimmer_runtime_kv_probe'
        assert digest(probe) == gate['binaries'][str(probe)]
        artifacts, audits = {}, {}
        for arm in ('global', 'independent'):
            repeat = arm == 'independent' and args.baseline_repeat_preparation_job is not None
            if repeat:
                other_gate = json.loads((args.baseline_repeat_preparation_job/'native-gate.json').read_text())
                assert other_gate['native_GPU_tests_pass'] and other_gate['source_spec_SHA256']==gate['source_spec_SHA256']
                other_probe = args.baseline_repeat_preparation_job/'model-build/muse_glimmer_runtime_kv_probe'
                assert digest(other_probe)==digest(probe)
            artifact = (args.baseline_repeat_preparation_job/'mini-global-artifact' if repeat else
                        args.preparation_job / ('mini-'+arm+'-artifact'))
            audit = json.loads((artifact / 'independent-mini-artifact-audit.json').read_text())
            assert audit['mini_is_not_full_model'] and audit['retained_original_layer_indices'] == [0,1,2,3]
            assert audit['independent_capacity'] == (arm == 'independent' and not repeat)
            assert audit['metadata']['get_max_prefill_chunk'] == 2048
            for name in ('model.pte', 'aoti_cuda_blob.ptd'):
                assert digest(artifact/name) == audit['artifacts'][name]['sha256']
            artifacts[arm], audits[arm] = artifact, audit
        weight_review = args.output/'weight-payload-review.json'
        state('verifying-every-original-weight-storage')
        package = Path(__file__).resolve().parent
        with (args.output/'weight-payload-review.log').open('x') as log:
            subprocess.run([sys.executable,'-B',str(package/'review_runtime_kv_weight_payload.py'),
                '--source',gate['source'],'--fixed',str(artifacts['global']/'aoti_cuda_blob.ptd'),
                '--runtime',str(artifacts['independent']/'aoti_cuda_blob.ptd'),'--output',str(weight_review)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        assert json.loads(weight_review.read_text())['all_quantized_and_nonquantized_storage_sha256_equal']
        import numpy as np
        widths = [1,2,4,8,16,32,64,65,127,128,255,256,511,512,1024]
        results = []
        state('waiting-GPU-lock')
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
            for number in range(1,5):
                for arm in (('global','independent') if number%2 else ('independent','global')):
                    folder = args.output / f'round-{number}' / arm
                    folder.mkdir(parents=True)
                    artifact = artifacts[arm]
                    command = [str(probe),'--pte='+str(artifact/'model.pte'),
                        '--data_path='+str(artifact/'aoti_cuda_blob.ptd'),'--output_dir='+str(folder),
                        '--max_sessions=21','--max_context=131072','--initial_kv_capacity=512']
                    state('running-fresh-owner-width-diagnostic', round=number, arm=arm)
                    with (folder/'stderr.log').open('x') as stderr, (folder/'protocol.jsonl').open('x') as raw:
                        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=stderr, text=True, bufsize=1)
                        selector = selectors.DefaultSelector()
                        selector.register(child.stdout, selectors.EVENT_READ)
                        serial = 0
                        try:
                            def receive():
                                while True:
                                    assert selector.select(600), 'Owned probe response timeout'
                                    line = child.stdout.readline()
                                    assert line, child.poll()
                                    try: reply = json.loads(line)
                                    except ValueError:
                                        stderr.write(line)
                                        continue
                                    if not isinstance(reply,dict) or not ('snapshot' in reply or 'failed' in reply):
                                        stderr.write(line)
                                        continue
                                    raw.write(json.dumps(dict(response=reply))+'\n'); raw.flush()
                                    assert not reply.get('failed'), reply
                                    return reply
                            def request(body):
                                nonlocal serial
                                serial += 1
                                body = dict(body, tag='width-'+str(serial))
                                raw.write(json.dumps(dict(request=body))+'\n');raw.flush()
                                child.stdin.write(json.dumps(body)+'\n');child.stdin.flush()
                                reply = receive()
                                assert reply['op'] == body['op']
                                return reply
                            assert receive()['ready']
                            cases = []
                            for width in widths:
                                owner = request(dict(op='new'))['seq_id']
                                before = request(dict(op='metrics'))['snapshot']
                                replies = []
                                for phase, positions in (('prefill',range(width)),('decode',range(width,width+1))):
                                    tokens = [100+(p*37)%149000 for p in positions]
                                    reply = request(dict(op='step',seq_ids=[owner]*len(tokens),tokens=tokens,
                                        select=[len(tokens)-1],argmax=True,save_logits=number==1))
                                    replies.append(dict(phase=phase,width=len(tokens),response=reply,
                                        teacher_forced_tokens=tokens, before=before))
                                    before = reply['snapshot']
                                cases.append(dict(width=width,steps=replies))
                                request(dict(op='remove',seq_id=owner))
                            child.stdin.close()
                            assert child.wait(timeout=120) == 0
                        finally:
                            selector.close()
                            if child.poll() is None: child.kill();child.wait()
                    results.append(dict(round=number,arm=arm,cases=cases,
                        protocol_SHA256=digest(folder/'protocol.jsonl')))
        comparisons = []
        first = {row['arm']:row for row in results if row['round']==1}
        for left,right in zip(first['global']['cases'],first['independent']['cases']):
            assert left['width'] == right['width']
            for l,r in zip(left['steps'],right['steps']):
                assert l['teacher_forced_tokens'] == r['teacher_forced_tokens']
                parts = [x['response']['logits'] for x in (l,r)]
                assert all(len(values)==1 for values in parts)
                lp,rp = [values[0] for values in parts]
                assert lp['shape']==rp['shape'] and lp['selected_indices']==rp['selected_indices']
                arrays = [np.fromfile(part['file'],dtype='<f4').reshape(part['shape']) for part in (lp,rp)]
                assert all(np.isfinite(a).all() for a in arrays)
                a,b = arrays
                delta = a.astype(np.float64)-b.astype(np.float64)
                comparisons.append(dict(prefill_width=left['width'],phase=l['phase'],
                    bitwise_equal=bool(np.array_equal(a,b)),strict_allclose=bool(np.allclose(a,b,rtol=1e-5,atol=1e-4)),
                    max_abs=float(np.abs(delta).max()),relative_L2=float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-12)),
                    argmax_equal=l['response']['argmax']==r['response']['argmax'],
                    files=[dict(file=part['file'],SHA256=digest(part['file'])) for part in (lp,rp)]))
        report = dict(rounds=results, comparisons=comparisons, artifacts=audits,
            comparison_is_two_global_exports=args.baseline_repeat_preparation_job is not None,
            arm_semantics={'global':'global reference','independent':
                ('second global export' if args.baseline_repeat_preparation_job else 'independent KV candidate')},
            every_original_weight_storage_SHA256_verified=True,weight_review_SHA256=digest(weight_review),
            probe_SHA256=digest(probe), same_teacher_forced_inputs=True,
            candidate_accepted=False, numerical_cause_not_established=True,
            no_performance_acceptance=True, no_kernel_or_acceptance_policy_changed=True)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        state('width-diagnostic-completed-pending-independent-review', cases=len(widths))
        print(json.dumps(comparisons))
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
