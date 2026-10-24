"""Isolated KA v4 versus verified native references on RTX5090.

Real Gemma GGUF roles and synthetic MG dimensions. This does not modify or
deploy a runtime. Search-cache configs are feasible references, not proof of
the executable-selected config or a global optimum. A100 acceptance remains
separate. Every timing pair uses four rounds, discarding the first.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--et-source', type=Path, required=True)
    parser.add_argument('--gemma-gguf', type=Path, required=True)
    parser.add_argument('--native-search', type=Path, required=True)
    parser.add_argument('--native-configs', type=Path, required=True)
    parser.add_argument('--precision-control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--owned-gpu-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.owned_gpu_worker:
        assert not args.output.exists(), 'Preserve every prior attempt'
        args.output.mkdir(parents=True)
        # The parent has no CUDA context. It holds the GPU lock until the child
        # process and its device context are gone, including interpreter teardown.
        (args.output/'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
        (args.output/'progress.json').write_text(json.dumps(dict(status='waiting-GPU-lock',controller_pid=os.getpid()))+'\n')
        with args.lock.open('a') as owner_lock:
            fcntl.flock(owner_lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],
                            '--owned-gpu-worker'],check=True)
            deadline=time.monotonic()+60
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<deadline,'GPU context did not disappear after worker exit'
                time.sleep(1)
        return
    assert json.loads((args.output/'lock-owner.json').read_text())['pid']==os.getppid()
    report = dict(scope=__doc__,rounds=4,discarded=[1],averaged=[2,3,4],warmups=3,
                  model_and_runtime_unchanged=True,deployment_accepted=False,cases=[])

    def save(status, **fields):
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')

    with args.lock.open('a') as lock:
        try:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError('Worker requires a parent that already owns the GPU lock')
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            os.environ['CUDA_VISIBLE_DEVICES']='0'
            os.environ['TRITON_CACHE_DIR']=str(args.output/'triton-cache')
            os.environ['TORCHINDUCTOR_CACHE_DIR']=str(args.output/'inductor-cache')
            sys.path.insert(0,str(args.et_source.parent))
            import torch
            import triton
            from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
            from executorch.backends.cuda.triton.kernels import quantized_gemm_utils as utils
            from executorch.backends.cuda.triton.kernels import quantized_gemm_family as family
            from executorch.backends.cuda.autotune import cuda_graph_timing as clock
            from candidates_w4a16_precision_v4 import CONFIGS,run
            from candidates_w4a16_precision_v4.validate import make_weights,precision_oracle,error_stats
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            assert torch.cuda.get_device_capability()==(12,0)
            started=time.monotonic()

            def idle_except_self():
                ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
                assert all(int(v.strip())==os.getpid() for v in ids if v.strip())
                assert time.monotonic()-started<7200,'Bounded GPU diagnostic exceeded two hours'

            previous=json.loads(args.native_search.read_text())
            report.update(device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,power.limit',
                                                         '--format=csv,noheader'],text=True).strip(),
                          torch=torch.__version__,triton=triton.__version__,source_sha256={})
            assert report['device']==previous['device'] and report['torch']==previous['torch'] and report['triton']==previous['triton']
            assert sum(len(c['native']) for c in previous['cases'])==504
            for name,module in {'int4_quantized_gemm.py':native,'quantized_gemm_utils.py':utils,
                                'quantized_gemm_family.py':family,'cuda_graph_timing.py':clock}.items():
                expected=next(h for f,h in previous['snapshot']['files'].items() if f.endswith('/'+name))
                actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                assert actual==expected,(name,'native source changed')
                report['source_sha256'][name]=actual
            assert hashlib.sha256(args.precision_control.read_bytes()).hexdigest()=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
            spec=importlib.util.spec_from_file_location('verified_precision_reference',args.precision_control)
            corrected=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(corrected)
            here=Path(__file__).resolve().parent
            manifest=json.loads((here/'public-source-manifest.json').read_text())
            for name,proof in manifest['files'].items():
                assert hashlib.sha256((here/name).read_bytes()).hexdigest()==proof['sha256']
            report['public_source_manifest_sha256']=hashlib.sha256((here/'public-source-manifest.json').read_bytes()).hexdigest()
            configs=json.loads(args.native_configs.read_text())
            lookup={(x['bucket'],x['N'],x['K']):x for x in configs['records']}
            report['reference_files_sha256']={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in
                                             [('native_search',args.native_search),('native_configs',args.native_configs),
                                              ('precision_control',args.precision_control)]}

            def checked(fn,ref):
                ys=[fn() for _ in range(4)]
                torch.cuda.synchronize()
                assert all(torch.equal(ys[1],y) for y in ys[2:]),'Nondeterministic measured outputs'
                stats=error_stats(ys[1],ref)
                assert stats['relative_rms']<.005,stats
                return dict(deterministic=True,**stats)

            def paired(reference,candidate):
                samples={'reference':[],'candidate':[]};orders=[]
                for repeat in range(4):
                    idle_except_self()
                    order=[('reference',reference),('candidate',candidate)]
                    if repeat%2:order.reverse()
                    orders.append([role for role,_ in order])
                    for role,fn in order:
                        for _ in range(3):fn()
                        samples[role].append(clock.time_with_cuda_graph_us(fn))
                means={key:statistics.mean(values[1:]) for key,values in samples.items()}
                return dict(samples_us=samples,means_us=means,orders=orders,
                            sample_sd_us={key:statistics.stdev(values[1:]) for key,values in samples.items()},
                            candidate_over_reference=means['candidate']/means['reference'])

            def evaluate(role,m,n,k,w,reference,metadata,check_reference):
                idle_except_self()
                generator=torch.Generator(device='cuda').manual_seed(19000+m+n)
                x=torch.randn((m,k),dtype=torch.bfloat16,device='cuda',generator=generator)
                ref=precision_oracle(x,w)
                record=dict(role=role,m=m,n=n,k=k,reference_metadata=metadata,
                            reference_numerical_gate_required=check_reference,candidates=[])
                report['cases'].append(record)
                nfn=lambda:reference(x)
                if check_reference:record['reference_check']=checked(nfn,ref)
                for ident,cfg in CONFIGS.items():
                    if cfg['bucket']!=m:continue
                    own=lambda ident=ident:run(x,*w,bucket=m,config=ident)
                    record['candidates'].append(dict(id=ident,config=cfg,numerics=checked(own,ref),
                                                     paired_timing=paired(nfn,own)))
                    save('running-pairs',role=role,m=m,candidate=ident)

            with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
                for m,n in [(8,19968),(16,19968),(8,39936),(16,39936)]:
                    k=6656
                    w=make_weights(n,k,18000+m+n,torch.device('cuda'))
                    baseline=next(c['native_winner'] for c in previous['cases'] if c['m']==m and c['n']==n)
                    cfg=baseline['config']
                    ncfg=triton.Config(cfg['kwargs'],num_warps=cfg['num_warps'],num_stages=cfg['num_stages'])
                    fn=lambda x,m=m,w=w,ncfg=ncfg,split=baseline['split']:native._launch_w4a16(m,x,*w,32,split_k=split,config=ncfg)
                    evaluate('MG-synthetic',m,n,k,w,fn,dict(baseline,arithmetic='Legacy staged BF16; timing only'),False)
                    del w
                    torch.cuda.empty_cache()

                from gguf import GGUFReader,GGMLQuantizationType
                from executorch.examples.models.gemma4_31b.batching_gguf_loader import decode_q4_blocks
                from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
                from executorch.extension.llm.export.gguf import ExportableGGUFTensor
                from executorch.examples.models.gemma4_31b import cuda_packers
                assert hashlib.sha256(Path(cuda_packers.__file__).read_bytes()).hexdigest()=='c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
                # Hash GGUF in chunks; avoid copying the whole model into host RAM.
                digest=hashlib.sha256()
                with args.gemma_gguf.open('rb') as file:
                    while chunk:=file.read(16*1024*1024):digest.update(chunk)
                assert digest.hexdigest()=='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
                report['GGUF_sha256']=digest.hexdigest()
                reader=GGUFReader(str(args.gemma_gguf));tensors={t.name:t for t in reader.tensors}

                def load_one(name):
                    tensor=tensors[name];n,k=tuple(reversed(tensor.shape.tolist()))
                    raw=torch.frombuffer(memoryview(tensor.data),dtype=torch.uint8).clone()
                    if tensor.tensor_type==GGMLQuantizationType.Q4_K:
                        portable=ExportableGGUFTensor.from_raw(raw.reshape(n,-1),'q4_k')
                        module=torch.nn.Linear(k,n,bias=False,device='meta')
                        cuda_packers.pack_linear_for_cuda(module,{'weight':portable});obj=module.weight
                    else:
                        assert tensor.tensor_type in [GGMLQuantizationType.Q4_0,GGMLQuantizationType.Q4_1]
                        typ='q4_0' if tensor.tensor_type==GGMLQuantizationType.Q4_0 else 'q4_1'
                        portable=decode_q4_blocks(raw,typ,(n,k))
                        obj=CudaCoalescedInt4Tensor.from_exportable_int4_tensor(portable)
                    weights=tuple(getattr(obj,key).cuda() for key in ['qdata','scale','scale_step','zero_point','zero_point_step'])
                    return weights,n,k

                roles=[('attn_k',['blk.0.attn_k.weight']),('ffn_down',['blk.0.ffn_down.weight']),
                       ('fused_gate_up',['blk.0.ffn_gate.weight','blk.0.ffn_up.weight']),
                       ('vocabulary',['output.weight' if 'output.weight' in tensors else 'token_embd.weight'])]
                for role,names in roles:
                    parts=[load_one(name) for name in names]
                    assert len({part[2] for part in parts})==1
                    n=sum(part[1] for part in parts);k=parts[0][2]
                    w=parts[0][0] if len(parts)==1 else tuple(torch.cat([part[0][i] for part in parts],dim=0) for i in range(5))
                    del parts
                    for m in [8,16]:
                        baseline=lookup[(m,n,k)];cfg=baseline['config']
                        ncfg=triton.Config({key:value for key,value in cfg.items() if key not in ['num_warps','num_stages']},
                                           num_warps=cfg['num_warps'],num_stages=cfg['num_stages'])
                        fn=lambda x,m=m,w=w,ncfg=ncfg,split=baseline['split_k']:corrected._launch(m,x,*w,32,split_k=split,config=ncfg)
                        evaluate('Gemma-real-'+role,m,n,k,w,fn,
                                 dict(baseline,source_tensors=names,kind='Feasible emitted config, executable-selection unproven'),True)
                    del w
                    torch.cuda.empty_cache()
            assert len(report['cases'])==12
            save('completed',candidate_deployment_accepted=False,
                 real_model_F1_and_A100_gate_still_required=True)
        except Exception as exc:
            save('failed',error_type=type(exc).__name__,detail=str(exc)[-1200:])
            raise


if __name__ == '__main__':
    main()
