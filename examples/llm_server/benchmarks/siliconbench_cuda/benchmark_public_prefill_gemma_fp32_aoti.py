"""Validate a dynamic-M AOTI conditional before any whole-model integration.

Real original GGUF packing is supplied as runtime inputs to both packages,
preventing a folded permanent dense weight from changing the comparison.
Small bulk M uses the frozen public candidate; larger M keeps the original
ET bulk operation. Static decode methods are outside this experiment.
"""
import argparse
from dataclasses import asdict
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


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        while b:=file.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source',type=Path,required=True)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--screen-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--example-M',type=int,choices=[512],default=512,
                   help="Match the full ET dynamic export example for baseline autotuning")
    p.add_argument('--gate-bk32',action='store_true',
                   help='Test a Gate K tile of 32 without changing its eligible M range')
    p.add_argument('--prepared-groups',action='store_true',
                   help='Test temporary prepared-metadata M128/M512; other sizes use original')
    p.add_argument('--runtime-chunk-range',action='store_true',
                   help='Separate opt-in Gate M496..512 qualification for the actual worker chunk size')
    p.add_argument('--roles',nargs='+',choices=['ffn_gate_up','ffn_down'],
                   default=['ffn_gate_up','ffn_down'])
    p.add_argument('--test-M',nargs='+',type=int,choices=[8,16,64,72,127,128,129,256,495,*range(496,513)],
                   default=[8,16,64,72,127,128,129,256,511,512])
    p.add_argument('--owned-gpu-worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.owned_gpu_worker:
        a.output.mkdir(exist_ok=False)
        (a.output/'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
        def parent_state(status,**fields):
            (a.output/'progress.json').write_text(json.dumps(dict(status=status,
                controller_pid=os.getpid(),GPU_lock_held_through_context_teardown=True,**fields))+'\n')
        parent_state('waiting-GPU-lock')
        try:
            with a.lock.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                    '--format=csv,noheader'],text=True).strip()
                with (a.output/'worker.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                        *sys.argv[1:],'--owned-gpu-worker'],stdout=log,stderr=subprocess.STDOUT)
                    parent_state('running-owned-GPU-worker',child_pid=child.pid)
                    try:
                        assert child.wait(timeout=14400)==0,'Operator worker failed; retain its log'
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                            try:child.wait(timeout=120)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                deadline=time.monotonic()+60
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                    '--format=csv,noheader'],text=True).strip():
                    assert time.monotonic()<deadline,'Keep the lock until CUDA context teardown'
                    time.sleep(1)
                result=json.loads((a.output/'results.json').read_text())
                parent_state('completed',cases=len(result['cases']))
        except BaseException as exc:
            parent_state('failed',error_type=type(exc).__name__,detail=str(exc)[-2000:])
            raise
        return
    assert json.loads((a.output/'lock-owner.json').read_text())['pid']==os.getppid()
    report=dict(scope=__doc__,controller_pid=os.getpid(),cases=[],packages={},
                model_route_changed=False,KA_invoked=False,A100_performance_verified=False,
                full_model_or_HTTP_accepted=False)
    def save(status,**fields):
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'progress.json').write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields))+'\n')
    save('waiting-GPU-lock')
    with a.lock.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('The parent must hold the GPU lock throughout the worker lifetime')
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        try:
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor-cache'),
                              TRITON_CACHE_DIR=str(a.output/'triton-cache'),
                              TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='2',MAX_JOBS='1')
            sys.path.insert(0,str(a.et_source.parent))
            import numpy as np
            import torch
            from gguf import GGUFReader,GGMLQuantizationType
            from executorch.examples.models.gemma4_31b.batching_gguf_loader import decode_q4_blocks
            from executorch.extension.llm.export.quant import to_default,fuse_along_output
            from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
            from public_prefill_gemma_fp32_aoti import reference_fp32 as _dequant_matmul, prepared_prefill_op
            from executorch.backends.cuda.autotune import cuda_graph_timing as clock
            from torch._inductor import aoti_compile_and_package,aoti_load_package
            import torch._inductor.cpp_builder as cpp

            from public_grouped_w4a16 import Config
            if a.prepared_groups:
                assert not a.gate_bk32
                from public_prefill_gemma_fp32_aoti import prepared_prefill_op
            if a.runtime_chunk_range:
                assert a.prepared_groups and not a.gate_bk32
                from public_prefill_gemma_runtime_shapes_aoti import prepared_runtime_prefill_op as prepared_prefill_op
            assert torch.cuda.get_device_capability()==(12,0)
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
            assert digest(a.gguf)=='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
            screen=json.loads(a.screen_results.read_text())
            assert len(screen['cases'])==16 and all(c['numeric_pass'] for c in screen['cases'])
            assert not screen['model_dispatch_changed'] and not screen['full_model_or_HTTP_accepted']
            assert a.prepared_groups and not a.gate_bk32
            report.update(GGUF_sha256=digest(a.gguf),screen_results_sha256=digest(a.screen_results),
                          export_example_M=a.example_M,
                          bulk_dynamic_range=[5,512],small_bulk_limit=128,static_decode_changed=False,
                          arithmetic='Original selected Gemma FP32 reconstruction then BF16; A16; FP32 GEMM accumulation')
            if a.prepared_groups:
                report.update(small_bulk_limit=None,prepared_metadata_candidate_sizes=[128,512],
                              preparation_cost_included=True,no_persistent_extra_metadata=True)
            if a.runtime_chunk_range:
                report.update(runtime_chunk_range=True,
                    prepared_metadata_candidate_sizes={'ffn_gate_up':[128,*range(496,513)],'ffn_down':[128]},
                    scheduler_budget=512,scheduler_reserved_decode_tokens=16,scheduler_prefill_chunk=496,
                    old_exact512_route_kept_as_separate_trial=True)
            reader=GGUFReader(str(a.gguf))
            def portable(t):
                kind={GGMLQuantizationType.Q4_0:'q4_0',GGMLQuantizationType.Q4_1:'q4_1'}[t.tensor_type]
                shape=tuple(map(int,t.shape[::-1]))
                raw=torch.from_numpy(np.array(t.data,copy=True)).view(torch.uint8).reshape(-1)
                return decode_q4_blocks(raw,kind,shape)
            class Linear(torch.nn.Module):
                def __init__(self,conditional,index):super().__init__();self.conditional=conditional
                def forward(self,x,q,sc,ss,z,zs):
                    if self.conditional:return prepared_prefill_op(x,q,sc,ss,z,zs)
                    return _dequant_matmul(x,q,sc,ss,z,zs,32)
            options={'emulate_precision_casts':True,'max_autotune':True,
                     'max_autotune_gemm_backends':'TRITON','loop_ordering_after_fusion':False,
                     'aot_inductor.compile_wrapper_opt_level':'O2',
                     'aot_inductor.embed_kernel_binary':True}
            report['compile_options']=options
            original_build=cpp.CppBuilder.build
            def record_build(builder):
                flags=re.findall(r'(?:^|\s)(-O[0-3s])(?:\s|$)',builder.get_command_line())
                with (a.output/'actual-wrapper-optimization.jsonl').open('a') as log:
                    log.write(json.dumps({'flags':flags,'target':builder._name})+'\n')
                return original_build(builder)
            cpp.CppBuilder.build=record_build
            stream=torch.cuda.Stream()
            with torch.inference_mode(),torch.cuda.stream(stream):
                for role in a.roles:
                    if role=='ffn_gate_up':
                        gate=next(t for t in reader.tensors if t.name=='blk.0.ffn_gate.weight')
                        up=next(t for t in reader.tensors if t.name=='blk.0.ffn_up.weight')
                        tensor=fuse_along_output([portable(gate),portable(up)]);n,k=30720,3840
                    else:
                        down=next(t for t in reader.tensors if t.name=='blk.0.ffn_down.weight')
                        tensor=portable(down);n,k=tuple(map(int,down.shape[::-1]))
                    module=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16)
                    pack_linear_for_cuda(module,{'weight':tensor})
                    w=tuple(getattr(module.weight,key).to('cuda').contiguous() for key in ['qdata','scale','scale_step','zero_point','zero_point_step'])
                    hashes=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in w]
                    assert hashes==next(c for c in screen['cases'] if c['role']==role)['weight']['packed_SHA256']
                    chosen=asdict(Config(bm=64,bn=128,bk=128,stages=2) if role=='ffn_gate_up'
                                  else Config(bm=64,bn=64,bk=128,stages=3))
                    index=0;compiled={}
                    example=torch.randn((a.example_M,k),device='cuda',dtype=torch.bfloat16)
                    dynamic=({0:torch.export.Dim('bulk_M',min=5,max=512)},{},{},{},{},{})
                    for name,conditional in [('reference',False),('conditional',True)]:
                        save('exporting-dynamic-AOTI',role=role,arm=name)
                        ep=torch.export.export(Linear(conditional,index),(example,*w),dynamic_shapes=dynamic,strict=True)
                        assert len(ep.state_dict)==0,'Weights must remain inputs, not folded constants'
                        torch.export.save(ep,str(a.output/f'{role}-{name}.pt2-export'))
                        path=a.output/f'{role}-{name}.pt2'
                        aoti_compile_and_package(ep,package_path=str(path),inductor_configs=options)
                        compiled[name]=aoti_load_package(str(path),run_single_threaded=True,device_index=0)
                        report['packages'][role+'-'+name]=dict(sha256=digest(path),bytes=path.stat().st_size,
                            original_packed_weight_SHA256=hashes,frozen_small_config=chosen)
                    for m in a.test_M:
                        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,
                                      generator=torch.Generator(device='cuda').manual_seed(47000+n+m))
                        target=_dequant_matmul(x,*w,32)
                        own=lambda:compiled['conditional'](x,*w)
                        ref=lambda:compiled['reference'](x,*w)
                        ys=[own().clone() for _ in range(3)];torch.cuda.synchronize()
                        assert all(torch.equal(ys[0],y) for y in ys[1:])
                        delta=ys[0].float()-target.float()
                        rms=float(delta.square().mean().sqrt()/target.float().square().mean().sqrt())
                        assert torch.isfinite(ys[0]).all() and rms<.005
                        torch.testing.assert_close(ys[0].float(),target.float(),rtol=.05,atol=4.)
                        ry=ref();reference_rms=float((ry.float()-target.float()).square().mean().sqrt()/target.float().square().mean().sqrt())
                        assert torch.isfinite(ry).all() and reference_rms<.005
                        torch.testing.assert_close(ry.float(),target.float(),rtol=.05,atol=4.)
                        reference_delta=ys[0].float()-ry.float()
                        direct_rms=float(reference_delta.square().mean().sqrt()/ry.float().square().mean().sqrt().clamp_min(1e-8))
                        assert direct_rms<.005
                        direct_different=int(torch.count_nonzero(ys[0]!=ry))
                        samples={'reference':[],'conditional':[]}
                        for round_number in range(4):
                            order=[('reference',ref),('conditional',own)]
                            if round_number%2:order.reverse()
                            for label,fn in order:
                                for _ in range(3):fn()
                                samples[label].append(clock.time_with_cuda_graph_us(fn))
                        means={key:statistics.mean(values[1:]) for key,values in samples.items()}
                        report['cases'].append(dict(role=role,M=m,N=n,K=k,relative_RMS=rms,deterministic=True,
                            reference_relative_RMS=reference_rms,
                            candidate_vs_compiled_reference_RMS=direct_rms,
                            candidate_vs_compiled_reference_different_elements=direct_different,
                            samples_us=samples,mean_last3_us=means,sample_SD_us={key:statistics.stdev(values[1:]) for key,values in samples.items()},
                            conditional_over_reference=means['conditional']/means['reference'],
                            expected_bulk_route=('prepared-metadata' if m==128 or (role=='ffn_gate_up' and (496<=m<=512 if a.runtime_chunk_range else m==512)) else 'original-dequant-matmul')
                                if a.prepared_groups else ('packed' if m<=128 else 'original-dequant-matmul'),
                            timing_scope='GPU graph timing of standalone AOTI linear; not current whole-model kernels or HTTP'))
                        save('AOTI-case-completed',role=role,M=m)
                    del compiled,w,module,tensor,example,target,x,ys
                    torch._dynamo.reset();torch.cuda.empty_cache()
            save('completed',cases=len(report['cases']))
        except Exception as error:
            (a.output/'exception-traceback.txt').write_text(traceback.format_exc())
            save('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':main()
