"""Probe the enclosing1024-row compile hint on an unselected128-row leaf.

This deliberately bypasses the row predicate to reproduce compiler branch
benchmarking. It is a memory-safety fixture, never model/performance data.
Each invocation uses a separate CUDA process so a failed old probe cannot
poison the corrected invocation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['et-source','gguf','policy','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--old-row-grid',type=Path)
    p.add_argument('--old-row-grid-SHA256')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(M=1024,branch_index=0,normal_runtime_predicate_deliberately_bypassed=True,
        fixture_not_model_acceptance=True,timing_not_performance=True)
    try:
        sys.path.insert(0,str(a.et_source.parent))
        import numpy as np
        import torch
        from gguf import GGUFReader,GGMLQuantizationType
        from executorch.extension.llm.export.gguf import ExportableGGUFTensor
        from executorch.extension.llm.export.quant import to_default
        from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
        from public_grouped_w4a16 import materialize
        from roofline_large_w4a16_aoti import register_policy,native_leaf_op
        assert torch.cuda.get_device_capability()==(12,0)
        assert sha(a.gguf)=='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
        policy=json.loads(a.policy.read_text());assert policy['shape']==[6656,19968]
        branch=policy['branches'][0];assert branch['bucket']==128 and branch['split_k']>1
        assert branch['config']['kwargs']['BLOCK_M']<128
        index=register_policy(policy)
        if a.old_row_grid:
            assert sha(a.old_row_grid)==a.old_row_grid_SHA256
            spec=importlib.util.spec_from_file_location('old_roofline_row_grid_fixture',a.old_row_grid)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install()
            report['old_row_grid_SHA256']=sha(a.old_row_grid)
        reader=GGUFReader(str(a.gguf))
        raw=next(t for t in reader.tensors if t.name.endswith('.ffn_down.weight') and t.tensor_type==GGMLQuantizationType.Q4_K)
        n,k=map(int,raw.shape[::-1]);assert [n,k]==policy['shape']
        blob=torch.from_numpy(np.array(raw.data,copy=True)).view(torch.uint8).reshape(n,-1)
        weight=to_default(raw.name,ExportableGGUFTensor.from_raw(blob,'q4_k',torch.bfloat16))
        module=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16)
        pack_linear_for_cuda(module,{'weight':weight})
        with torch.inference_mode():
            w=tuple(getattr(module.weight,name).cuda().contiguous() for name in ['qdata','scale','scale_step','zero_point','zero_point_step'])
            assert [hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in w]==policy['packed_SHA256']
            x=torch.randn((1024,k),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(62401))
            torch.cuda.synchronize()
            report.update(branch=branch,packed_SHA256=policy['packed_SHA256'])
            y=native_leaf_op(x,*w,index,0);torch.cuda.synchronize()
            oracle=materialize(w,arithmetic='native-zero-fma-bf16',bk=128)
            target=torch.nn.functional.linear(x,oracle)
            delta=y.float()-target.float()
            relative=float(delta.square().mean().sqrt()/target.float().square().mean().sqrt().clamp_min(1e-8))
            report.update(finite=bool(torch.isfinite(y).all()),relative_RMS=relative,shape=list(y.shape))
            assert report['finite'] and relative<.005 and y.shape==(1024,n)
            report['deterministic']=all(torch.equal(y,native_leaf_op(x,*w,index,0)) for _ in range(2))
            assert report['deterministic'];torch.cuda.synchronize()
        report['status']='passed'
    except BaseException as e:
        report.update(status='failed',error=str(e),traceback=traceback.format_exc())
        raise
    finally:(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
