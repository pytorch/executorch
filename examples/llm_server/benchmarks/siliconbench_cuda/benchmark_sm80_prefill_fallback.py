"""Measure the unchanged SM80 native fallback before/after the public hook.

Synthetic ET-format packed weights; this is a GEMM diagnostic, not model F1,
server throughput, or a new static forward. No kernel generation is invoked.
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.et_source.parent))
    import torch
    import executorch.backends.cuda.quantize_op_dispatch.int4_dispatch as dispatch
    from public_prefill_gemma_export_hook import install_gemma_prefill_port
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    original_hash=digest(dispatch.__file__)
    assert original_hash=='1a3fdadd5dcfe03ef9b6f55f93e4fce9a88b44867839b1316c824f7bd1faaa3e'
    result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='running',
        controller_pid=os.getpid(),synthetic_weights=True,A16=True,model_loaded=False,
        model_correctness_proven=False,server_perf_measured=False,new_KA=False,
        source_SHA256=digest(__file__),original_dispatch_SHA256=original_hash,records=[])
    def save():
        t=a.output/'results.pending.json';t.write_text(json.dumps(result,indent=2)+'\n');t.replace(a.output/'results.json')
    save()
    try:
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Leave other GPU work untouched'
            capability=torch.cuda.get_device_capability();assert capability==(8,0)
            result['device']=torch.cuda.get_device_name();result['capability']=list(capability)
            original=dispatch._dequant_matmul
            baseline=torch.compile(original,fullgraph=True,dynamic=False)
            receipt=install_gemma_prefill_port(dispatch,capability=capability,enabled=True,runtime_chunk_range=True)
            assert not receipt['installed'] and dispatch._dequant_matmul is original
            assert 'public_prefill_gemma_runtime_shapes_aoti' not in sys.modules
            routed=torch.compile(dispatch._dequant_matmul,fullgraph=True,dynamic=False)
            result['hook_receipt']=receipt;result['actual_dispatch_object_identical']=True
            torch.manual_seed(731)
            n,k=30720,3840
            weights=(torch.randint(0,256,(n,k//2),device='cuda',dtype=torch.uint8),
                torch.randint(1,9,(n,k//32),device='cuda',dtype=torch.uint8),
                torch.full((n,k//256),0.015625,device='cuda',dtype=torch.float16),
                torch.full((n,k//32),7,device='cuda',dtype=torch.uint8),
                torch.ones((n,k//256),device='cuda',dtype=torch.float16))
            for m in [8,16,128,496,512]:
                x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
                functions={'original':baseline,'hook_enabled':routed};graphs={};outputs={}
                for name,fn in functions.items():
                    for _ in range(3):fn(x,*weights,32)
                    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):outputs[name]=fn(x,*weights,32)
                    graphs[name]=graph
                torch.cuda.synchronize()
                assert torch.isfinite(outputs['original']).all().item()
                assert torch.equal(outputs['original'],outputs['hook_enabled'])
                samples={name:[] for name in functions};orders=[]
                for round_number in range(4):
                    order=list(functions) if round_number%2==0 else list(reversed(functions));orders.append(order)
                    for name in order:
                        for _ in range(3):graphs[name].replay()
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):graphs[name].replay()
                        end.record();end.synchronize();samples[name].append(start.elapsed_time(end)*1000/20)
                measured={name:dict(all4_us=values,first_discarded_us=values[0],kept3_us=values[1:],mean_us=statistics.mean(values[1:]),sample_SD_us=statistics.stdev(values[1:])) for name,values in samples.items()}
                result['records'].append(dict(M=m,N=n,K=k,scope='native dequant_matmul fallback; not complete static dispatch',
                    dtype='BF16',group_size=32,bitwise_equal=True,CUDA_graphs=True,orders=orders,
                    measurements=measured,hook_over_original=measured['hook_enabled']['mean_us']/measured['original']['mean_us']))
                save();del graphs,outputs;torch.cuda.synchronize()
            assert digest(dispatch.__file__)==original_hash
            result.update(status='completed-pending-independent-fallback-performance-review',original_source_unmodified=True)
            save()
    except BaseException as e:
        result.update(status='failed',error_type=type(e).__name__,detail=str(e)[-1600:]);save();raise


if __name__=='__main__':main()
