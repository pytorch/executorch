# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
from typing import Optional, Sequence, Union

import torch
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.node import Argument, Target
from torch.utils import _pytree as pytree


class GraphBuilder(ExportPass):
    """Utility class for creating a graph module with user-specified ops.

    This class allows us to create test graph modules with any ops we want
    directly, rather than relying on decomposition or passes.

    Usage:
        builder = GraphBuilder()
        # To insert placeholders, use builder.placeholder.
        x = builder.placeholder("x", torch.randn(1, 3, 224, 224))
        # To insert an op, use builder.call_operator.
        op = builder.call_operator(
            some_op
            (x, other_args, ...),
        )
        # Insert outputs as a list of ProxyValues using builder.output.
        builder.output([op])
        # Get GraphModule from builder.
        gm = builder.get_graph_module()
    """

    def __init__(self) -> None:
        self.exporter = ExportPass()
        self.tracer: ExportPass.ExportTracer = self.ExportTracer(
            self, torch.fx.graph.CodeGen()
        )
        self.fake_tensor_mode = FakeTensorMode(
            allow_fallback_kernels=False,
            allow_non_fake_inputs=True,
        )
        self.tracer.fake_tensor_mode = self.fake_tensor_mode

        # This will be called to create nodes in tracer.
        self.interpreter = torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )

    # pyre-ignore[14]: Inconsistent override.
    def placeholder(
        self, target: str, fake_tensor: Union[FakeTensor, torch.Tensor]
    ) -> ProxyValue:
        if not isinstance(fake_tensor, FakeTensor):
            fake_tensor = self.fake_tensor_mode.from_tensor(fake_tensor)
        logging.info(f"Creating placeholder {target} => {fake_tensor.shape}")
        placeholder = super().placeholder(target, fake_tensor, NodeMetadata({}))
        return placeholder

    # pyre-ignore[14]: Inconsistent override.
    def output(self, results: list[ProxyValue]) -> ProxyValue:
        logging.info(f"Creating outputs {results}")
        return super().output(results, NodeMetadata({}))

    def get_graph_module(self) -> torch.fx.GraphModule:
        return torch.fx.GraphModule(self.tracer.root, self.tracer.graph)

    def call_operator(
        self,
        op,  # pyre-ignore
        args: tuple[Argument, ...],
        kwargs: Optional[dict[str, Argument]] = None,
        meta: Optional[NodeMetadata] = None,
    ) -> ProxyValue:
        if meta is None:
            meta = NodeMetadata({})
        if kwargs is None:
            kwargs = {}
        return super().call_operator(op, args, kwargs, meta)


def single_op_builder(
    placeholders: Sequence[Union[torch.Tensor, FakeTensor]],
    op: Target,
    args: Sequence[Argument],
    kwargs: Optional[dict[str, Argument]] = None,
) -> torch.fx.GraphModule:
    """Create a graph module with a single op.

    Args:
        placeholders: Placeholders to be used as inputs to the GraphModule.
        op: The op to be inserted.
        args: The args to be passed to the op.
        kwargs: The kwargs to be passed to the op.

    Returns:
        A graph module with a single op
    """
    builder = GraphBuilder()
    op_to_placeholder_dict = {
        p: builder.placeholder(f"p_{i}", p) for i, p in enumerate(placeholders)
    }
    proxy_args, proxy_kwargs = pytree.tree_map_only(
        (torch.Tensor, FakeTensor), lambda x: op_to_placeholder_dict[x], (args, kwargs)
    )
    node = builder.call_operator(op, proxy_args, proxy_kwargs)
    builder.output([node])
    return builder.get_graph_module()
