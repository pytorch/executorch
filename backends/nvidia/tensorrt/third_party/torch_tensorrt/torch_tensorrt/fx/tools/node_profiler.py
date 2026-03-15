from typing import Any

import torch
from torch import fx


class NodeProfiler(fx.Interpreter):
    """
    This is basically a variant of shape prop in
    https://github.com/pytorch/pytorch/blob/74849d9188de30d93f7c523d4eeceeef044147a9/torch/fx/passes/shape_prop.py#L65.
    Instead of propagating just the shape, we record all the intermediate node Tensor values.
    This is useful to debug some of lowering pass issue where we want to check a specific
    tensor value. Note that output value can be tuple(Tensor) as well as Tensor.
    """

    def __init__(self, module: fx.GraphModule):
        super().__init__(module)
        self.execution_time = {}
        self.node_map = {}
        self.iter = 100

    def run_node(self, n: fx.Node) -> Any:
        result = super().run_node(n)
        if n.op not in {"call_function", "call_method", "call_module"}:
            return result

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for _ in range(self.iter):
            result = super().run_node(n)

        end_event.record()
        torch.cuda.synchronize()

        self.execution_time[f"{n.name}"] = (
            start_event.elapsed_time(end_event) / self.iter
        )
        self.node_map[n.name] = n
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.
        Args:
            *args (Tensor): the sample input.
        Returns:
            Any: The value returned from executing the Module
        """
        return super().run(*args)
