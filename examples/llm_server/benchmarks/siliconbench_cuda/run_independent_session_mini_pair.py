"""Wait on one live mini preparation, run both arms and review raw evidence.

This never exports full models or replaces accepted main results.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--nsys', type=Path, required=True)
    parser.add_argument('--wait-minutes', type=int, default=240)
    parser.add_argument('--automatic-reclaim', action='store_true')
    parser.add_argument('--source-spec', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior attempts'
    args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    source_spec = args.source_spec or package / 'independent_session_capacity_source.json'

    def state(status, **fields):
        value = dict(status=status,pid=os.getpid(),
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     full_model_export_started=False, candidate_accepted=False, **fields)
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(value,indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    def run(argv, name, timeout=21600):
        with (args.output/(name+'.log')).open('x') as log:
            subprocess.run([str(v) for v in argv],stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=timeout)

    try:
        deadline = time.monotonic()+60*args.wait_minutes
        while True:
            progress = json.loads((args.preparation_job/'progress.json').read_text())
            if progress['status']=='mini-artifacts-ready':
                break
            assert progress['status']!='failed', progress
            handle = Path('/proc')/str(progress['controller_pid'])
            assert handle.exists(), 'Preparation handle missing; inspect terminal evidence, never restart here'
            command = (handle/'cmdline').read_bytes().replace(b'\0',b' ')
            assert (b'prepare_independent_session_mini.py' in command or
                    b'export_independent_session_mini_from_native.py' in command or
                    b'prepare_independent_session_shared_autotune_mini.py' in command)
            state('waiting-on-confirmed-live-mini-preparation',
                  preparation_pid=progress['controller_pid'],preparation_status=progress['status'])
            if time.monotonic()>=deadline:
                state('observation-timeout',preparation_pid=progress['controller_pid'],
                      preparation_still_running=True)
                return
            time.sleep(30)
        source = Path(progress['source'])
        native_gate = args.preparation_job/'native-gate.json'
        assert hashlib.sha256(source_spec.read_bytes()).hexdigest() == json.loads(native_gate.read_text())['source_spec_SHA256']
        probe = args.preparation_job/'model-build/muse_glimmer_runtime_kv_probe'
        for arm in ['global','independent']:
            artifact = args.preparation_job/('mini-'+arm+'-artifact')
            audit = json.loads((artifact/'independent-mini-artifact-audit.json').read_text())
            state('running-mini-'+arm)
            run([sys.executable,'-B','-u',package/'run_independent_session_mini_probe.py',
                 '--probe',probe,'--artifact',artifact,'--native-gate',native_gate,
                 '--expected-PTE-sha256',audit['artifacts']['model.pte']['sha256'],
                 '--expected-weights-sha256',audit['artifacts']['aoti_cuda_blob.ptd']['sha256'],
                 '--arm',arm,'--job-dir',args.output/arm,'--lock',args.lock,
                 '--nsys',args.nsys,*(['--automatic-reclaim'] if args.automatic_reclaim else [])],arm+'-probe')
        state('reviewing-model-weight-storages')
        weights = args.output/'weight-payload-review.json'
        run([sys.executable,'-B',package/'review_runtime_kv_weight_payload.py',
             '--source',source,'--fixed',args.preparation_job/'mini-global-artifact/aoti_cuda_blob.ptd',
             '--runtime',args.preparation_job/'mini-independent-artifact/aoti_cuda_blob.ptd',
             '--output',weights],'weight-payload-review',timeout=1800)
        state('reviewing-actual-mini-graph-trace')
        trace = args.output/'independent/round-1/cuda-graphs.nsys-rep'
        sqlite = args.output/'independent/round-1/cuda-graphs.sqlite'
        assert trace.is_file() and not sqlite.exists()
        run([args.nsys,'export','--type=sqlite','--output',sqlite,trace],'graph-export',timeout=1800)
        graph = args.output/'graph-review.json'
        run([sys.executable,'-B',package/'review_independent_session_mini_graph_trace.py',
             '--runtime-job',args.output/'independent','--sqlite',sqlite,'--output',graph],
            'graph-review',timeout=1800)
        state('reviewing-matched-raw-mini-tokens-and-logits')
        run([sys.executable,'-B',package/'review_independent_session_mini_pair.py',
             '--fixed',args.output/'global','--runtime',args.output/'independent',
             '--source-spec',source_spec,
             '--graph-review',graph,'--weight-payload-review',weights,
             '--output',args.output/'mini-review.json'],'raw-review',timeout=1800)
        run([sys.executable,'-B',package/'review_independent_session_native_latency.py',
             '--global-job',args.output/'global','--independent-job',args.output/'independent',
             '--output',args.output/'native-latency-review.json'],'native-latency-review',timeout=1800)
        state('mini-raw-review-completed-performance-and-HTTP-gates-pending',
              raw_review=str(args.output/'mini-review.json'))
    except Exception as error:
        state('failed',detail=str(error)[-2000:],traceback=traceback.format_exc()[-2500:])
        raise


if __name__=='__main__':
    main()
