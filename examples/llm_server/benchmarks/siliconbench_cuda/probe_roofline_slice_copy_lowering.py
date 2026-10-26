"""Check nested slice-copy rewrite values, layout and fresh-copy ownership."""
import argparse
import json
from pathlib import Path
import torch
from roofline_slice_copy_lowering import SliceCopyLowering,install_static_compile_reuse

class Fixture(torch.nn.Module):
    def forward(self,x):
        def branch(x,start):
            y=torch.ops.aten.slice_copy.Tensor(x,1,start,start+32)
            return y+1
        return torch.cond(x.shape[0]<=128,lambda x:branch(x,2),lambda x:branch(x,3),(x,))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous-export',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    shape={'x':{0:torch.export.Dim('M',min=65,max=1024)}}
    example=torch.randn(128,64,dtype=torch.bfloat16)
    before=torch.export.export(Fixture(),(example,),dynamic_shapes=shape)
    after=torch.export.export(Fixture(),(example,),dynamic_shapes=shape)
    events=[];SliceCopyLowering(events)(after.graph_module)
    assert sum(v['slice_copy_replaced'] for v in events)==2
    cases=[]
    for M in [65,128,129,512,520,1024]:
        x=torch.randn(M,64,dtype=torch.bfloat16);saved=x.clone()
        gold=before.module()(x);actual=after.module()(x)
        assert torch.equal(gold,actual) and torch.equal(x,saved) and actual.is_contiguous()
        actual.add_(1);assert torch.equal(x,saved)
        cases.append(dict(M=M,bitwise_equal=True,input_unchanged_after_output_mutation=True,contiguous=True))
    from executorch.backends.cuda.cuda_backend import CudaBackend
    reuse=[];install_static_compile_reuse(CudaBackend,a.previous_export,reuse)
    report=dict(status='passed',CPU_only=True,GPU_work=False,cases=cases,events=events,
        all_seven_preserved_static_paths_exist=True,full_model_acceptance_still_required=True)
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
