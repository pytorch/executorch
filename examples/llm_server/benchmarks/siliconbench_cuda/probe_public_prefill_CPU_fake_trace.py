"""Check CPU/FakeTensor export routing for a CUDA-targeted prefill operator.

No model/weights/artifacts are changed; no GPU benchmark or promotion occurs.
This catches device guards that incorrectly reject the CPU examples used by
the actual model exporter. A disabled or SM80 route keeps the original object.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--et-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port-kind', choices=['small-bulk','prepared-metadata'], default='small-bulk')
    a = p.parse_args(); assert not a.output.exists()
    sys.path.insert(0,str(a.et_source.parent))
    import torch
    from executorch.backends.cuda.quantize_op_dispatch import int4_dispatch as dispatch
    from public_prefill_export_hook import install_bulk_prefill_port
    if a.port_kind == 'prepared-metadata':
        from public_prefill_export_hook import install_prepared_prefill_port as install_bulk_prefill_port
    target = ('siliconbench.public_prepared_prefill_a16' if a.port_kind == 'prepared-metadata'
              else 'siliconbench.public_bulk_w4a16')
    original = dispatch._dequant_matmul
    off = install_bulk_prefill_port(dispatch,capability=(8,0),enabled=True)
    assert not off['installed'] and dispatch._dequant_matmul is original
    on = install_bulk_prefill_port(dispatch,capability=(12,0),enabled=True)
    assert on['installed']
    class Probe(torch.nn.Module):
        def forward(self,x,q,sc,ss,z,zs):
            return dispatch._dequant_matmul(x,q,sc,ss,z,zs,32)
    n,k = 39936,6656
    args=(torch.empty((2048,k),dtype=torch.bfloat16),
          torch.empty((n,k//2),dtype=torch.uint8),
          torch.empty((n,k//32),dtype=torch.uint8),
          torch.empty((n,k//256),dtype=torch.float16),
          torch.empty((n,k//32),dtype=torch.uint8),
          torch.empty((n,k//256),dtype=torch.float16))
    program=torch.export.export(Probe(),args,
        dynamic_shapes=({0:torch.export.Dim('M',min=64,max=2048)},None,None,None,None,None),strict=True)
    targets=[str(node.target) for node in program.graph.nodes if node.op=='call_function']
    assert sum(target in name for name in targets)==1
    result=dict(CPU_examples=True,strict_FakeTensor_export_pass=True,port_kind=a.port_kind,
                experimental_operator_count=1,graph_targets=targets,
                SM80_original_dispatch_identity=True,CPU_real_eager_not_routed=True,
                GPU_benchmark=False,full_model_or_HTTP_accepted=False,
                hook_SHA256=hashlib.sha256(Path(__file__).with_name('public_prefill_export_hook.py').read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
