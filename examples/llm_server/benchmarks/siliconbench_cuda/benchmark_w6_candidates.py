"""Measure immutable W6A16 candidates against native ET on actual MG GGUF.

No framework source or model is changed. Activations are synthetic BF16; weights
come from the exact GGUF and native planar packer. Server acceptance is separate.
"""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source',type=Path,required=True)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    p.add_argument('--owned-worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    cutoff=datetime.datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))
    if not a.owned_worker:
        assert not a.output.exists()
        a.output.mkdir(parents=True)
        (a.output/'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
        (a.output/'progress.json').write_text(json.dumps(dict(status='waiting-GPU-lock'))+'\n')
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            remaining=(cutoff-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            assert remaining>0,'User tuning deadline passed; do not start'
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--owned-worker']
            with (a.output/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    assert child.wait(timeout=min(7200,remaining))==0,'Inspect preserved worker results'
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    (a.output/'progress.json').write_text(json.dumps(dict(status='failed',reason='User cutoff or two-hour benchmark bound'))+'\n')
                    raise
            until=time.monotonic()+60
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<until,'CUDA context not released; preserve other work'
                time.sleep(1)
        return
    assert json.loads((a.output/'lock-owner.json').read_text())['pid']==os.getppid()
    sys.path.insert(0,str(a.et_source.parent))
    import numpy as np
    import torch
    import triton
    from gguf import GGUFReader,GGMLQuantizationType
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor
    from executorch.backends.cuda.dp4a_planar_int6_tensor import CudaDp4aPlanarInt6Tensor
    from executorch.backends.cuda.triton.kernels import int6_quantized_gemm as native
    from executorch.backends.cuda.autotune import cuda_graph_timing as clock
    from executorch.backends.cuda.autotune.launch_params import autotune_launch_params
    from torch._library.triton import set_wrap_triton_enabled
    from candidates_w6a16_v6 import CONFIGS,run
    from candidates_w6a16_v6.validate import _oracle
    assert torch.cuda.get_device_capability()==(12,0)
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=='93ea2c0eb97503980f643e38658c25c595c9e7a15a6e08d3ea72066c62b332f3'
    digest=hashlib.sha256()
    with a.gguf.open('rb') as source:
        while block:=source.read(8*1024*1024):digest.update(block)
    assert digest.hexdigest()=='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
    report=dict(weights='Actual same MG Q4_K_M GGUF; these two roles retain Q6_K',activations='Synthetic BF16',
                device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                torch=torch.__version__,triton=triton.__version__,GGUF_sha256=digest.hexdigest(),
                rounds=4,discarded=[1],averaged=[2,3,4],orders=['AB','BA','AB','BA'],
                native_reference='Native W6A16 complete invocation; eager split/tile selection frozen before graph capture; executable-selected config still requires integration review',
                graph_measurement='Raw eager Triton launches, bypassing the per-call higher-order-op autotuner wrapper only within this benchmark',
                A100_acceptance_separate=True,automatic_deployment=False,cases=[])
    def save(status,**fields):
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        pending=a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,worker_pid=os.getpid(),**fields),indent=2)+'\n')
        pending.replace(a.output/'progress.json')
    def bounded():
        assert datetime.datetime.now(datetime.timezone.utc)<cutoff,'User tuning cutoff reached'
        ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
        assert all(int(pid)==os.getpid() for pid in ids if pid.strip()),'Other GPU work detected'
    def check(fn,ref):
        values=[fn() for _ in range(4)];torch.cuda.synchronize()
        assert all(torch.equal(values[1],v) for v in values[2:]),'Nondeterministic output'
        got=values[1].float();target=ref.float();assert bool(torch.isfinite(got).all())
        diff=got-target;relative=float(diff.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-12))
        assert relative<.005,('relative RMS exceeds gate',relative)
        return dict(relative_rms=relative,max_abs=float(diff.abs().max()),deterministic=True)
    def paired(ref,own):
        samples={'reference':[],'candidate':[]}
        for repeat in range(4):
            order=[('reference',ref),('candidate',own)]
            if repeat%2:order.reverse()
            for role,fn in order:
                bounded()
                for _ in range(3):fn()
                samples[role].append(clock.time_with_cuda_graph_us(fn))
        mean={role:statistics.mean(v[1:]) for role,v in samples.items()}
        return dict(samples_us=samples,mean_us=mean,sample_sd_us={role:statistics.stdev(v[1:]) for role,v in samples.items()},candidate_over_reference=mean['candidate']/mean['reference'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    try:
        reader=GGUFReader(str(a.gguf))
        q6=[t for t in reader.tensors if t.tensor_type==GGMLQuantizationType.Q6_K]
        assert len(q6)==52
        with set_wrap_triton_enabled(False),torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            for role,n,k in [('ffn_down',6656,19968),('attn_v',256,6656)]:
                tensor=next(t for t in q6 if t.name.endswith('.'+role+'.weight'))
                assert tuple(map(int,tensor.shape[::-1]))==(n,k)
                raw=torch.from_numpy(np.array(tensor.data,copy=True)).view(torch.uint8).reshape(n,-1)
                gt=ExportableGGUFTensor.from_raw(raw,'q6_k')
                weight=CudaDp4aPlanarInt6Tensor.from_exportable_gguf(gt)
                planes=tuple(v.to('cuda').contiguous() for v in [weight.ql,weight.qh,weight.scale,weight.steps])
                raw_sha=hashlib.sha256(raw.numpy().tobytes()).hexdigest()
                for m in [8,16]:
                    bounded();save('running-native-preparation',role=role,m=m)
                    generator=torch.Generator(device='cuda').manual_seed(25000+m+n)
                    x=torch.randn((m,k),dtype=torch.bfloat16,device='cuda',generator=generator)
                    oracle=_oracle(x,*planes)
                    with autotune_launch_params() as selection:
                        native._launch_w6a16(m,x,*planes,16)
                    assert len(selection.picks)==1
                    pick=selection.picks[0]
                    assert pick.function.endswith('._launch_w6a16') and pick.param=='split_k'
                    split=int(pick.choice)
                    selected=native._BUCKET_KERNELS[m].best_config
                    reference_config=triton.Config(dict(selected.kwargs),num_warps=selected.num_warps,
                                                   num_stages=selected.num_stages,num_ctas=selected.num_ctas)
                    reference=lambda m=m,x=x,planes=planes,split=split,config=reference_config:native._launch_w6a16(m,x,*planes,16,split_k=split,config=config)
                    reference();torch.cuda.synchronize()
                    record=dict(role=role,M=m,N=n,K=k,GGUF_tensor=tensor.name,GGUF_tensor_sha256=raw_sha,
                                reference_numerics=check(reference,oracle),
                                reference_selection=dict(split_k=split,split_times_us=pick.times_us,
                                    config=dict(kwargs=reference_config.kwargs,num_warps=reference_config.num_warps,
                                                num_stages=reference_config.num_stages,num_ctas=reference_config.num_ctas),
                                    eagerly_selected=True,executable_selected_config_unproven=True),candidates=[])
                    report['cases'].append(record)
                    for ident,candidate_config in CONFIGS.items():
                        if candidate_config['bucket']!=m:continue
                        own=lambda ident=ident:run(x,*planes,bucket=m,config=ident)
                        try:
                            numeric=check(own,oracle)
                            result=dict(id=ident,config=candidate_config,numerics=numeric,timing=paired(reference,own))
                        except Exception as error:
                            result=dict(id=ident,error_type=type(error).__name__,detail=str(error)[-500:])
                            if 'capture' in str(error).lower():
                                record['candidates'].append(result)
                                (a.output/'exception-traceback.txt').write_text(traceback.format_exc())
                                raise
                        record['candidates'].append(result);save('running-candidate-pairs',role=role,m=m,candidate=ident)
                    del x,oracle,reference
                del planes,weight,gt,raw
                torch.cuda.empty_cache()
        save('completed',cases=len(report['cases']),candidate_records=sum(len(c['candidates']) for c in report['cases']),server_acceptance_pending=True)
    except Exception as error:
        (a.output/'exception-traceback.txt').write_text(traceback.format_exc())
        save('failed',error_type=type(error).__name__,detail=str(error)[-1000:]);raise


if __name__=='__main__':main()
