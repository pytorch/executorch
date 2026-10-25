"""Check a measured large-M routing policy at dynamic AOTI row boundaries.

Packed original GGUF weights remain explicit runtime tensor inputs. Native
arithmetic and complete fallback use separate numerical oracles; no folded
persistent dense weight or new activation quantization is introduced.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(16*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['et-source','gguf','policy','output','lock']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--role',choices=['ffn_down','ffn_gate_up','attention_out'],required=True)
    p.add_argument('--max-M',type=int,default=1024)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(cases=[],rounds=4,discard_first=True,model_accepted=False,main_results_changed=False)
    def state(status,**extra):
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'progress.json').write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
    state('waiting-GPU-lock')
    try:
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state('waiting-prior-CUDA-context-teardown')
            deadline=time.monotonic()+120
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<deadline,'Existing GPU context remains occupied; leave it untouched'
                time.sleep(1)
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'),TRITON_CACHE_DIR=str(a.output/'triton'),TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1',OMP_NUM_THREADS='2')
            sys.path.insert(0,str(a.et_source.parent))
            import numpy as np
            import torch
            from gguf import GGUFReader,GGMLQuantizationType
            from executorch.extension.llm.export.gguf import ExportableGGUFTensor
            from executorch.extension.llm.export.quant import to_default,fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
            from executorch.backends.cuda.autotune import cuda_graph_timing as clock
            from torch._inductor import aoti_compile_and_package,aoti_load_package
            from public_grouped_w4a16 import materialize
            from roofline_large_w4a16_aoti import register_policy,large_w4a16_op
            assert torch.cuda.get_device_capability()==(12,0)
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
            assert digest(a.gguf)=='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            policy=json.loads(a.policy.read_text());index=register_policy(policy)
            report.update(policy=policy,policy_SHA256=digest(a.policy),GGUF_SHA256=digest(a.gguf),dynamic_range=[65,a.max_M],activation='BF16',weights_are_explicit_inputs=True)
            def portable(raw):
                assert raw.tensor_type==GGMLQuantizationType.Q4_K
                n=int(raw.shape[-1]);data=torch.from_numpy(np.array(raw.data,copy=True)).view(torch.uint8).reshape(n,-1)
                return to_default(raw.name,ExportableGGUFTensor.from_raw(data,'q4_k',torch.bfloat16))
            reader=GGUFReader(str(a.gguf))
            if a.role=='ffn_gate_up':
                raw=[next(t for t in reader.tensors if t.name==n) for n in ['blk.0.ffn_gate.weight','blk.0.ffn_up.weight']]
                weight=fuse_along_output([portable(t) for t in raw]);n,k=39936,6656
            else:
                suffix='.ffn_down.weight' if a.role=='ffn_down' else '.attn_output.weight'
                raw=[next(t for t in reader.tensors if t.name.endswith(suffix) and t.tensor_type==GGMLQuantizationType.Q4_K)]
                n,k=map(int,raw[0].shape[::-1]);weight=portable(raw[0])
            assert tuple(policy['shape'])==(n,k)
            module=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16)
            pack_linear_for_cuda(module,{'weight':weight})
            w=tuple(getattr(module.weight,n).cuda().contiguous() for n in ['qdata','scale','scale_step','zero_point','zero_point_step'])
            hashes=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in w]
            assert hashes==policy['packed_SHA256']
            report['packed_SHA256']=hashes
            class Linear(torch.nn.Module):
                def __init__(self,candidate):super().__init__();self.candidate=candidate
                def forward(self,x,q,sc,ss,z,zs):
                    if self.candidate:return large_w4a16_op(x,q,sc,ss,z,zs,index)
                    return _dequant_matmul(x,q,sc,ss,z,zs,32)
            options={'emulate_precision_casts':True,'max_autotune':True,'max_autotune_gemm_backends':'TRITON','loop_ordering_after_fusion':False,'aot_inductor.compile_wrapper_opt_level':'O2','aot_inductor.embed_kernel_binary':True}
            report['compile_options']=options
            stream=torch.cuda.Stream()
            with torch.inference_mode(),torch.cuda.stream(stream),clock.cuda_graph_timing():
                example=torch.randn((a.max_M,k),device='cuda',dtype=torch.bfloat16)
                dynamic=({0:torch.export.Dim('rows',min=65,max=a.max_M)},{},{},{},{},{})
                packages={}
                for name,candidate in [('fallback',False),('candidate',True)]:
                    state('exporting-dynamic-AOTI',arm=name)
                    ep=torch.export.export(Linear(candidate),(example,*w),dynamic_shapes=dynamic,strict=True)
                    assert not ep.state_dict,'Never fold a dense weight into the package'
                    path=a.output/(name+'.pt2')
                    aoti_compile_and_package(ep,package_path=str(path),inductor_configs=options)
                    packages[name]=aoti_load_package(str(path),run_single_threaded=True,device_index=0)
                    report[name+'_package']=dict(SHA256=digest(path),bytes=path.stat().st_size,range_constraints=str(ep.range_constraints))
                oracle=materialize(w,arithmetic='native-zero-fma-bf16',bk=128)
                boundaries={65,72,128,256,512,520,a.max_M}
                for branch in policy['branches']:boundaries.update([branch['max_M']-1,branch['max_M'],branch['max_M']+1])
                def numeric(y,target):
                    delta=y.float()-target.float()
                    return dict(finite=bool(torch.isfinite(y).all()),relative_RMS=float(delta.square().mean().sqrt()/target.float().square().mean().sqrt().clamp_min(1e-8)),max_abs=float(delta.abs().max()))
                for m in sorted(v for v in boundaries if 65<=v<=a.max_M):
                    state('validating-AOTI-boundary',M=m)
                    x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(92000+n+m))
                    functions={name:lambda package=package:package(x,*w) for name,package in packages.items()}
                    ys=[functions['candidate']().clone() for _ in range(3)];torch.cuda.synchronize()
                    assert all(torch.equal(ys[0],v) for v in ys[1:])
                    selected=next((b for b in policy['branches'] if m<=b['max_M']),None)
                    fallback_target=_dequant_matmul(x,*w,32)
                    target=torch.nn.functional.linear(x,oracle) if selected else fallback_target
                    checks=numeric(ys[0],target);assert checks['finite'] and checks['relative_RMS']<.005
                    fallback_checks=numeric(functions['fallback'](),fallback_target)
                    assert fallback_checks['finite'] and fallback_checks['relative_RMS']<.005
                    samples={name:[] for name in functions}
                    for rnd in range(4):
                        for name in (['fallback','candidate'] if rnd%2==0 else ['candidate','fallback']):
                            for _ in range(3):functions[name]()
                            samples[name].append(clock.time_with_cuda_graph_us(functions[name]))
                    means={name:statistics.mean(v[1:]) for name,v in samples.items()}
                    report['cases'].append(dict(M=m,N=n,K=k,selected_branch=selected,native_or_fallback_oracle=checks,compiled_fallback_oracle=fallback_checks,candidate_vs_fallback=numeric(ys[0],functions['fallback']()),deterministic=True,samples_us=samples,mean_last3_us=means,sample_SD_us={name:statistics.stdev(v[1:]) for name,v in samples.items()},candidate_over_complete_fallback=means['candidate']/means['fallback'],timing_scope='standalone dynamic AOTI linear; full forward still pending'))
                    state('boundary-completed',M=m,completed=len(report['cases']))
                state('completed-pending-full-forward')
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());state('failed');raise


if __name__=='__main__':main()
