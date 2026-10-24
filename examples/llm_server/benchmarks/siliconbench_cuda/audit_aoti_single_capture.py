"""Validate one-call graph timing against preserved actual-GGUF AOTI packages."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['et-source', 'gguf', 'reuse-report', 'output', 'lock']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--owned-worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    cutoff = datetime.datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))

    def state(status, **fields):
        tmp = a.output/'progress.pending.json'
        tmp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields),indent=2)+'\n')
        tmp.replace(a.output/'progress.json')

    if not a.owned_worker:
        assert not a.output.exists(), 'Preserve prior attempts'
        a.output.mkdir(parents=True)
        (a.output/'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
        state('waiting-GPU-lock')
        child = None
        def terminate(*_):
            raise InterruptedError('Task-owned validation stopped')
        signal.signal(signal.SIGTERM, terminate)
        try:
            with a.lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                remaining = (cutoff-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                assert remaining > 0, 'User tuning cutoff passed'
                with (a.output/'worker.log').open('x') as log:
                    child = subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--owned-worker'],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        assert child.wait(timeout=min(900,remaining)) == 0, 'Inspect preserved validation failure'
                    finally:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try: child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid,signal.SIGKILL);child.wait()
                until = time.monotonic()+60
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    assert time.monotonic()<until, 'Retain lock until CUDA context teardown'
                    time.sleep(1)
        except Exception as e:
            state('failed',error_type=type(e).__name__,detail=str(e)[-1200:])
            raise
        return

    assert json.loads((a.output/'lock-owner.json').read_text())['pid'] == os.getppid()
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='2')
    sys.path.insert(0,str(a.et_source.parent))
    import torch
    from gguf import GGUFReader,GGMLQuantizationType
    from executorch.examples.models.gemma4_31b.batching_gguf_loader import decode_q4_blocks
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from candidates_w4a16_v8.validate import precision_oracle,error_stats
    from aoti_single_capture_timing import time_single_aoti_capture_us,TIMING_POLICY
    try:
        assert torch.cuda.get_device_capability() == (12,0)
        assert digest(a.gguf) == '7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
        report = json.loads(a.reuse_report.read_text())
        record = next(c for c in report['AOTI_cases'] if c['bucket']==8 and c['n']==2048 and c['k']==3840)
        assert record['dynamic_M_min']==5 and record['dynamic_M_max']==8
        models={}
        for label,package in record['packages'].items():
            path=Path(package['path'])
            assert path.resolve().is_relative_to(a.reuse_report.parent.resolve())
            assert path.stat().st_size==package['bytes'] and digest(path)==package['sha256']
            models[label]=torch._inductor.aoti_load_package(str(path))
        assert set(models)=={'reference','candidate'}
        reader=GGUFReader(str(a.gguf))
        tensor=next(t for t in reader.tensors if t.name=='blk.0.attn_k.weight')
        assert tensor.tensor_type==GGMLQuantizationType.Q4_0
        raw=torch.frombuffer(memoryview(tensor.data),dtype=torch.uint8).clone()
        obj=CudaCoalescedInt4Tensor.from_exportable_int4_tensor(decode_q4_blocks(raw,'q4_0',(2048,3840)))
        w=tuple(getattr(obj,key).cuda() for key in ['qdata','scale','scale_step','zero_point','zero_point_step'])
        x=torch.randn((8,3840),dtype=torch.bfloat16,device='cuda',generator=torch.Generator(device='cuda').manual_seed(26001))
        def call(label,xx):
            result=models[label](xx,*w)
            if isinstance(result,(tuple,list)):
                assert len(result)==1;result=result[0]
            return result
        checks=[];samples={'reference':[],'candidate':[]};orders=[]
        state('running-single-capture-validation')
        with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            for m in [5,7,8]:
                xx=x[:m].contiguous();oracle=precision_oracle(xx,w)
                for label in ['reference','candidate']:
                    y=call(label,xx);second=call(label,xx)
                    torch.cuda.synchronize()
                    assert torch.equal(y,second) and torch.isfinite(y).all()
                    stats=error_stats(y,oracle)
                    assert stats['relative_rms']<.005, stats
                    checks.append(dict(m=m,label=label,deterministic=True,**stats))
            for n in range(4):
                order=['reference','candidate'] if n%2==0 else ['candidate','reference'];orders.append(order)
                for label in order:
                    for _ in range(3):call(label,x)
                    samples[label].append(time_single_aoti_capture_us(lambda label=label:call(label,x)))
        means={label:statistics.mean(values[1:]) for label,values in samples.items()}
        result=dict(validated=True,source_report_sha256=digest(a.reuse_report),packages=record['packages'],
            GGUF_sha256=digest(a.gguf),timing_policy=TIMING_POLICY,one_AOTI_invocation_per_capture=True,
            rounds=4,discarded=[1],averaged=[2,3,4],warmups=3,tail_checks=checks,
            samples_us=samples,means_us=means,orders=orders,
            sample_sd_us={label:statistics.stdev(values[1:]) for label,values in samples.items()},
            candidate_over_reference=means['candidate']/means['reference'],
            model_runtime_source_and_packages_unchanged=True,no_server_candidate_enabled=True)
        (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        state('completed',validation_only=True,no_server_candidate_enabled=True)
    except Exception as e:
        (a.output/'exception-traceback.txt').write_text(traceback.format_exc())
        state('failed',error_type=type(e).__name__,detail=str(e)[-1200:])
        raise


if __name__=='__main__':
    main()
