"""Validate frozen Down512/520 dense configurations inside dynamic AOTI.

Original public GGUF packing is a runtime input to both packages. Check the
full staged-dequant/GEMM pipeline, exact candidate sizes and nearby fallback
sizes. Four paired AB/BA rounds discard the first; this is not HTTP or F1.
No model dispatch changes, activation quantization or KA task are performed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source',type=Path,required=True)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--dense-screen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=2400)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    report=dict(controller_pid=os.getpid(),cases=[],packages={},
        model_dispatch_changed=False,new_KA=False,activation_dtype='BF16',
        accumulation='FP32',weights_requantized=False,full_model_or_HTTP_accepted=False)
    def state(status,**fields):
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        tmp=a.output/'progress.pending.json'
        tmp.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields))+'\n')
        tmp.replace(a.output/'progress.json')
    state('waiting-GPU-lock')
    try:
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            start=time.monotonic()
            def bounded():
                assert time.monotonic()-start<a.max_seconds
                owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
                assert all(int(v)==os.getpid() for v in owners if v.strip())
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor-cache'),
                TRITON_CACHE_DIR=str(a.output/'triton-cache'),TORCHINDUCTOR_COMPILE_THREADS='1',
                MAX_JOBS='1',OMP_NUM_THREADS='2')
            sys.path.insert(0,str(a.et_source.parent))
            import numpy as np
            import torch
            from gguf import GGUFReader,GGMLQuantizationType
            from executorch.extension.llm.export.gguf import ExportableGGUFTensor
            from executorch.extension.llm.export.quant import to_default
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
            from executorch.backends.cuda.autotune import cuda_graph_timing as clock
            from torch._inductor import aoti_compile_and_package,aoti_load_package
            import torch._inductor.cpp_builder as cpp
            from public_prefill_dense_aoti import down_prefill_op
            assert torch.cuda.get_device_capability()==(12,0)
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
            assert sha(a.gguf)=='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
            screen=json.loads(a.dense_screen.read_text())
            prior=[c for c in screen['results'] if c['role']=='ffn_down' and c['M']==512][0]
            reader=GGUFReader(str(a.gguf))
            raw=next(t for t in reader.tensors if t.name.endswith('.ffn_down.weight') and t.tensor_type==GGMLQuantizationType.Q4_K)
            n,k=map(int,raw.shape[::-1]);assert (n,k)==(6656,19968)
            tensor=to_default(raw.name,ExportableGGUFTensor.from_raw(
                torch.from_numpy(np.array(raw.data,copy=True)).view(torch.uint8).reshape(n,-1),
                'q4_k',torch.bfloat16))
            module=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16)
            pack_linear_for_cuda(module,{'weight':tensor})
            w=tuple(getattr(module.weight,key).cuda().contiguous() for key in
                ['qdata','scale','scale_step','zero_point','zero_point_step'])
            hashes=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in w]
            assert hashes==prior['weight']['packed_SHA256']
            report.update(GGUF_SHA256=sha(a.gguf),packed_weight_SHA256=hashes,
                weight_source=raw.name,dense_screen_SHA256=sha(a.dense_screen),
                kernel_source_SHA256=sha(Path(__file__).with_name('public_prefill_dense_aoti.py')),
                example_M=2048,dynamic_M=[64,2048],candidate_sizes=[512,520],
                frozen_tiles={'512':[64,64,128,4,3],'520':[64,128,128,4,3]})
            class Linear(torch.nn.Module):
                def __init__(self,operation):super().__init__();self.operation=operation
                def forward(self,x,q,sc,ss,z,zs):return self.operation(x,q,sc,ss,z,zs)
            native=lambda x,q,sc,ss,z,zs:_dequant_matmul(x,q,sc,ss,z,zs,32)
            options={'emulate_precision_casts':True,'max_autotune':True,
                'max_autotune_gemm_backends':'TRITON','loop_ordering_after_fusion':False,
                'aot_inductor.compile_wrapper_opt_level':'O2','aot_inductor.embed_kernel_binary':True}
            report['compile_options']=options
            original_build=cpp.CppBuilder.build
            def build(builder):
                flags=re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)',builder.get_command_line())
                assert not flags or flags[-1] not in ['-O0','-O1']
                with (a.output/'actual-wrapper-optimization.jsonl').open('a') as log:
                    log.write(json.dumps(dict(flags=flags,target=builder._name))+'\n')
                return original_build(builder)
            cpp.CppBuilder.build=build
            with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
                example=torch.randn((2048,k),dtype=torch.bfloat16,device='cuda')
                dynamic=({0:torch.export.Dim('M',min=64,max=2048)},{},{},{},{},{})
                compiled={}
                for arm,op in [('reference',native),('candidate',down_prefill_op)]:
                    bounded();state('compiling-dynamic-AOTI',arm=arm)
                    ep=torch.export.export(Linear(op),(example,*w),dynamic_shapes=dynamic,strict=True)
                    assert not ep.state_dict
                    graph='\n'.join(str(g.graph) for g in ep.graph_module.modules() if isinstance(g,torch.fx.GraphModule))
                    (a.output/f'{arm}-exported-graph.txt').write_text(graph)
                    if arm=='candidate':assert 'siliconbench.public_down_prefill_bf16' in graph
                    torch.export.save(ep,str(a.output/f'{arm}.pt2-export'))
                    package=a.output/f'{arm}.pt2'
                    aoti_compile_and_package(ep,package_path=str(package),inductor_configs=options)
                    compiled[arm]=aoti_load_package(str(package),run_single_threaded=True,device_index=0)
                    report['packages'][arm]=dict(SHA256=sha(package),bytes=package.stat().st_size)
                for m in [72,128,256,511,512,513,520,521,2048]:
                    bounded();state('checking-dynamic-AOTI',M=m)
                    x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,
                        generator=torch.Generator(device='cuda').manual_seed(47000+n+m))
                    target=_dequant_matmul(x,*w,32)
                    numeric={};functions={arm:(lambda arm=arm:compiled[arm](x,*w)) for arm in compiled}
                    for arm,fn in functions.items():
                        ys=[fn().clone() for _ in range(3)];torch.cuda.synchronize()
                        assert all(torch.equal(ys[0],y) for y in ys[1:])
                        delta=ys[0].float()-target.float()
                        rms=float(delta.square().mean().sqrt()/target.float().square().mean().sqrt().clamp_min(1e-8))
                        assert bool(torch.isfinite(ys[0]).all()) and rms<.005
                        torch.testing.assert_close(ys[0].float(),target.float(),rtol=.05,atol=4.)
                        numeric[arm]=dict(relative_RMS=rms,max_abs=float(delta.abs().max()),deterministic=True)
                    samples={arm:[] for arm in compiled};orders=[]
                    for rnd in range(4):
                        order=['reference','candidate'] if rnd%2==0 else ['candidate','reference'];orders.append(order)
                        for arm in order:
                            for _ in range(3):functions[arm]()
                            samples[arm].append(clock.time_with_cuda_graph_us(functions[arm]))
                    means={arm:statistics.mean(v[1:]) for arm,v in samples.items()}
                    report['cases'].append(dict(M=m,N=n,K=k,numeric=numeric,samples_us=samples,
                        orders=orders,mean_last3_us=means,sample_SD_us={arm:statistics.stdev(v[1:]) for arm,v in samples.items()},
                        candidate_over_reference=means['candidate']/means['reference'],
                        expected_route='manual-dense-with-staged-unpack' if m in [512,520] else 'original-bulk-fallback',
                        scope='Standalone dynamic AOTI GPU graph; includes dequant; not model or HTTP'))
                    state('case-completed',M=m)
            state('completed-dynamic-AOTI-pending-model-validation',cases=len(report['cases']))
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());state('failed');raise


if __name__=='__main__':main()
