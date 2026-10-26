"""Lower nested slice_copy as slice+contiguous clone without a proxy executor.

Retain fresh-copy semantics, including consumers that mutate or take views.
This isolated export pass applies only to the roofline forward_others method.
"""

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
