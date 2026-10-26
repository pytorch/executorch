"""Lower nested slice_copy as slice+contiguous clone without a proxy executor.

Retain fresh-copy semantics, including consumers that mutate or take views.
This isolated export pass applies only to the roofline forward_others method.
"""
import contextlib
import contextvars
import hashlib
import json
from pathlib import Path

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.pass_base import PassResult


class SliceCopyLowering:
    def __init__(self,events):self.events=events
    def __call__(self,root):
        changed=0
        for module in root.modules():
            if not isinstance(module,torch.fx.GraphModule):continue
            for node in list(module.graph.nodes):
                edge=node.target is ops.edge.aten.slice_copy.Tensor
                if node.op!='call_function' or not (edge or node.target is torch.ops.aten.slice_copy.Tensor):continue
                target=ops.edge.aten.slice.Tensor if edge else torch.ops.aten.slice.Tensor
                def fake(v):
                    if isinstance(v,torch.fx.Node):return v.meta['val']
                    return v
                with module.graph.inserting_before(node):
                    view=module.graph.call_function(target,node.args,node.kwargs)
                    view.meta=dict(node.meta)
                    view.meta['val']=torch.ops.aten.slice.Tensor(
                        *torch.fx.map_arg(node.args,fake),**torch.fx.map_arg(node.kwargs,fake))
                node.target=ops.edge.aten.clone.default if edge else torch.ops.aten.clone.default
                node.args=(view,);node.kwargs={'memory_format':torch.contiguous_format}
                changed+=1
            module.graph.lint();module.recompile()
        self.events.append(dict(slice_copy_replaced=changed,fresh_contiguous_copy_retained=True,
                                nested_graphs_visited=True,arithmetic_and_weights_changed=False))
        return PassResult(root,bool(changed))


def install(backend,job):
    events=[];original=backend.get_custom_passes
    def passes(cls,specs):
        result=list(original(specs))
        if cls.method_name_from_compile_specs(specs)=='forward_others':
            result.append(SliceCopyLowering(events))
        return result
    backend.get_custom_passes=classmethod(passes)
    return events


def install_static_compile_reuse(backend,previous,events):
    """Reuse only unchanged static delegates from the preserved terminal run.

    The caller preserves the source/GGUF/shape configuration and validates all
    complete-model outputs after export. Never reuse the failed dynamic graph.
    """
    import shlex
    previous=Path(previous)
    assert 'Method forward_others missing fallback kernels' in (previous/'export.log').read_text()
    routes=json.loads((previous/'actual-strict-export-route-audit.json').read_text())
    static=[v for v in routes if not v['dynamic']]
    assert len(static)==7 and all(v['native_large_operator_calls']==0 for v in static)
    records=[json.loads(line) for line in (previous/'actual-compiler-optimization.jsonl').read_text().splitlines()]
    wrappers=[]
    for v in records:
        if not v['target_name'].endswith('.wrapper'):continue
        command=shlex.split(v['command']);cpp=next(Path(s) for s in command if s.endswith('.wrapper.cpp'))
        assert previous/'cache/inductor' in cpp.parents
        if cpp not in wrappers:wrappers.append(cpp)
    assert len(wrappers)==8
    sources={}
    for M,cpp in zip([1,2,4,8,16,32,64],wrappers[:7]):
        so=cpp.with_suffix('.so');blob=cpp.with_name(cpp.name[:-len('.wrapper.cpp')]+'.wrapper_weights.blob')
        assert so.is_file() and blob.is_file(),(so,blob)
        sources['forward_'+str(M)]=(so,blob)
    current=contextvars.ContextVar('roofline_compile_method',default=None)
    original_context=backend.get_extra_aoti_compile_context_manager
    @contextlib.contextmanager
    def context(cls,specs=None):
        token=current.set(cls.method_name_from_compile_specs(specs))
        try:
            with original_context(specs):yield
        finally:current.reset(token)
    backend.get_extra_aoti_compile_context_manager=classmethod(context)
    original_compile=torch._inductor.aot_compile
    def compile(module,args,**kwargs):
        method=current.get()
        if method not in sources:return original_compile(module,args,**kwargs)
        assert not any('siliconbench.roofline_native_large_w4a16' in str(n.target)
                       for gm in module.modules() if isinstance(gm,torch.fx.GraphModule) for n in gm.graph.nodes)
        so,blob=sources[method]
        events.append(dict(reused_static_method=method,SO=str(so),SO_SHA256=hashlib.sha256(so.read_bytes()).hexdigest(),
                           graph_code_SHA256=hashlib.sha256(module.code.encode()).hexdigest(),
                           failed_dynamic_method_not_reused=True,full_output_validation_still_required=True))
        return [str(so),str(blob)]
    torch._inductor.aot_compile=compile
