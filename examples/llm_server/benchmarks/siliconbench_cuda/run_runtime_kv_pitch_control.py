"""Isolate physical pitch from export changes using unchanged mini artifacts.

This is a correctness diagnostic, not a performance experiment. The runtime
artifact is run with two and twenty-two physical columns; the fixed artifact
uses twenty-two. No acceptance threshold or model route is changed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--fixed-artifact', type=Path, required=True)
    p.add_argument('--runtime-artifact', type=Path, required=True)
    p.add_argument('--reference-protocol', type=Path, required=True)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    a = p.parse_args()
    a.job_dir.mkdir(exist_ok=False)
    requests = [json.loads(line)['request'] for line in a.reference_protocol.read_text().splitlines()
                if 'request' in json.loads(line)]
    first_steps = [r for r in requests if r['op'] == 'step'][:2]
    assert [len(r['tokens']) for r in first_steps] == [511, 1]
    import numpy as np
    report = dict(scope=__doc__, probe_sha256=digest(a.probe),
                  reference_protocol_sha256=digest(a.reference_protocol),
                  arms={}, comparisons=[], candidate_accepted=False)
    def state(status, **fields):
        (a.job_dir/'progress.json').write_text(json.dumps(dict(status=status, controller_pid=os.getpid(), **fields))+'\n')
    state('waiting-GPU-lock')
    with a.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        try:
            for label, artifact, owners, columns in [('fixed22',a.fixed_artifact,21,22),
                                                     ('runtime22',a.runtime_artifact,21,22),
                                                     ('runtime2',a.runtime_artifact,1,2)]:
                state('running-physical-pitch-control', arm=label)
                folder=a.job_dir/label;folder.mkdir()
                with (folder/'stderr.log').open('x') as stderr, (folder/'protocol.jsonl').open('x') as protocol:
                    command=[str(a.probe),'--pte='+str(artifact/'model.pte'),
                             '--data_path='+str(artifact/'aoti_cuda_blob.ptd'),'--output_dir='+str(folder),
                             '--max_sessions=21','--max_context=131072','--initial_kv_capacity=512']
                    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                           stderr=stderr,text=True,bufsize=1,env=os.environ|{'CUDA_VISIBLE_DEVICES':'0'})
                    selector=selectors.DefaultSelector();selector.register(child.stdout, selectors.EVENT_READ)
                    try:
                        def read():
                            while True:
                                assert selector.select(300), 'Probe response timeout'
                                line=child.stdout.readline();assert line, child.poll()
                                try:value=json.loads(line)
                                except ValueError:stderr.write(line);continue
                                if not isinstance(value,dict) or not any(k in value for k in ['snapshot','failed']):continue
                                protocol.write(json.dumps({'response':value})+'\n');protocol.flush()
                                assert not value.get('failed'), value
                                return value
                        ready=read();assert ready['ready']
                        serial=0
                        def request(value):
                            nonlocal serial
                            serial+=1;value=dict(value,tag=f'control-{serial}')
                            protocol.write(json.dumps({'request':value})+'\n');protocol.flush()
                            child.stdin.write(json.dumps(value)+'\n');child.stdin.flush()
                            return read()
                        ids=[request({'op':'new'})['seq_id'] for _ in range(owners)]
                        assert ids==list(range(owners))
                        # Runtime storage materializes at the first write, not
                        # at owner admission. Keep empty owners through that
                        # write to establish pitch22, then remove them without
                        # compacting or changing the survivor's history.
                        before=request({'op':'metrics'})['snapshot']
                        saved=[]
                        for index,value in enumerate(first_steps):
                            clean={k:v for k,v in value.items() if k not in ['tag','save']}
                            clean['save_logits']=True
                            answer=request(clean)
                            assert answer['snapshot']['physical_columns']==columns
                            saved.extend(answer['logits'])
                            if index==0:
                                for owner in ids[1:]:request({'op':'remove','seq_id':owner})
                        assert len(saved)==2, 'Both actual forward logits must be saved'
                        report['arms'][label]=dict(model_PTE_sha256=digest(artifact/'model.pte'),
                            weights_PTD_sha256=digest(artifact/'aoti_cuda_blob.ptd'),
                            physical_columns=columns, before=before, selected_logits=saved)
                        child.stdin.close();assert child.wait(timeout=120)==0
                    finally:
                        selector.close()
                        if child.poll() is None:child.kill();child.wait()
            for left,right in [('fixed22','runtime22'),('fixed22','runtime2'),('runtime22','runtime2')]:
                values=[]
                for lpart,rpart in zip(report['arms'][left]['selected_logits'],report['arms'][right]['selected_logits']):
                    assert lpart['shape']==rpart['shape']
                    x=np.fromfile(lpart['file'],dtype='<f4').reshape(lpart['shape'])
                    y=np.fromfile(rpart['file'],dtype='<f4').reshape(rpart['shape'])
                    assert np.isfinite(x).all() and np.isfinite(y).all()
                    d=x-y
                    values.append(dict(shape=lpart['shape'], bitwise_equal=bool(np.array_equal(x,y)),
                        max_abs=float(np.max(np.abs(d))), relative_RMS=float(np.linalg.norm(d)/np.linalg.norm(x)),
                        argmax_equal=bool(np.array_equal(x.argmax(-1),y.argmax(-1)))))
                report['comparisons'].append(dict(left=left,right=right,logit_checks=values))
            (a.job_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            state('completed-diagnostic-no-promotion')
        except Exception as error:
            state('failed', error_type=type(error).__name__, detail=str(error));raise


if __name__=='__main__':main()
