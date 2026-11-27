# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Find and lower single-input, single-output pointwise regions."""

from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import dataclass

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassResult


@dataclass(frozen=True)
class FusionRegion:
    nodes: tuple[Node, ...]
    input_node: Node


class FuseElementwiseToLUTPass(ExportPass):
    """Replace single-input, single-output int8 pointwise regions with a TABLE.

    The caller supplies a TABLE target with ``(input, table)`` arguments and
    ``table[input + 128]`` semantics, and pure pointwise targets that can be
    evaluated in integer code space. Backend quantization restrictions and
    metadata updates belong in backend specializations. Backends can override
    table generation and replacement for other table formats.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        table_op: Callable,
        supported_ops: Collection[Callable],
        *,
        region_filter: Callable[[FusionRegion], bool] | None = None,
        supported_dtypes: Collection[torch.dtype] = (torch.int8,),
        min_region_size: int = 2,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.table_op = table_op
        self.supported_ops = {*supported_ops, table_op}
        self.region_filter = region_filter
        self.supported_dtypes = supported_dtypes
        self.min_region_size = min_region_size

    def _constant(self, node: Node) -> torch.Tensor | None:
        if node.op != "placeholder" or not is_param_node(self.exported_program, node):
            return None
        signature = self.exported_program.graph_signature
        buffer_target = signature.inputs_to_buffers.get(node.name)
        if buffer_target in signature.buffers_to_mutate.values():
            return None
        return get_param_tensor(self.exported_program, node)

    def _is_supported_node(self, node: Node) -> bool:
        if node.op != "call_function" or node.target not in self.supported_ops:
            return False
        value = node.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype not in self.supported_dtypes
        ):
            return False
        if node.target == self.table_op:
            _, table = node.args
            data = self._constant(table)
            return (
                data is not None and data.dtype == torch.int8 and data.shape == (256,)
            )
        return all(
            data is None or data.numel() == 1
            for arg in node.all_input_nodes
            for data in (self._constant(arg),)
        )

    def _data_inputs(self, node: Node) -> list[Node]:
        return [arg for arg in node.all_input_nodes if self._constant(arg) is None]

    def _single_input_suffix(self, nodes: tuple[Node, ...]) -> FusionRegion | None:
        candidates = set(nodes)
        inputs = Counter(
            arg
            for node in nodes
            for arg in self._data_inputs(node)
            if arg not in candidates
        )
        # A multi-input prefix can still feed a single-input suffix.
        for start in range(len(nodes) - self.min_region_size + 1):
            if len(inputs) == 1:
                return FusionRegion(nodes[start:], next(iter(inputs)))
            node = nodes[start]
            candidates.remove(node)
            for arg in self._data_inputs(node):
                inputs[arg] -= 1
                if inputs[arg] == 0:
                    del inputs[arg]
            users = sum(user in candidates for user in node.users)
            if users:
                inputs[node] = users
        return None

    def _find_fusion_region(self, sink: Node) -> FusionRegion | None:
        candidates: set[Node] = set()
        pending = [sink]
        while pending:
            node = pending.pop()
            if node in candidates or not self._is_supported_node(node):
                continue
            candidates.add(node)
            pending.extend(self._data_inputs(node))

        # An interior result with an outside consumer is a boundary tensor.
        while True:
            exposed = {
                node
                for node in candidates - {sink}
                if any(user not in candidates for user in node.users)
            }
            if not exposed:
                break
            candidates.difference_update(exposed)

        if len(candidates) < self.min_region_size:
            return None
        region = self._single_input_suffix(
            tuple(node for node in sink.graph.nodes if node in candidates)
        )
        if region is None:
            return None
        input_value = region.input_node.meta.get("val")
        if (
            not isinstance(input_value, torch.Tensor)
            or input_value.dtype not in self.supported_dtypes
        ):
            return None
        if any(node.meta["val"].shape != input_value.shape for node in region.nodes):
            return None
        if self.region_filter is not None and not self.region_filter(region):
            return None
        return region

    def _build_lookup_table(self, region: FusionRegion) -> torch.Tensor:
        values = {region.input_node: torch.arange(-128, 128, dtype=torch.int8)}
        for node in region.nodes:
            for arg in node.all_input_nodes:
                constant = self._constant(arg)
                if constant is not None:
                    values[arg] = (
                        constant.reshape(()) if constant.numel() == 1 else constant
                    )
            args = map_arg(node.args, values.__getitem__)
            kwargs = map_arg(node.kwargs, values.__getitem__)
            if node.target == self.table_op:
                input_tensor, table = args
                values[node] = table[input_tensor.to(torch.int64) + 128]
            else:
                values[node] = node.target(*args, **kwargs)
        table = values[region.nodes[-1]]
        if table.dtype != torch.int8 or table.shape != (256,):
            raise ValueError(
                "Configured pointwise operators must produce 256 int8 codes"
            )
        return table

    def _insert_table_constant(self, region: FusionRegion, table: torch.Tensor) -> Node:
        graph = region.input_node.graph
        first_placeholder = next(
            node for node in graph.nodes if node.op == "placeholder"
        )
        with graph.inserting_before(first_placeholder):
            return create_constant_placeholder(
                self.exported_program,
                graph,
                f"b_{region.nodes[-1].name}_fused_table_constant",
                InputKind.BUFFER,
                table,
                persistent_buffer=True,
            )

    def _fuse_region(self, region: FusionRegion) -> None:
        table_node = self._insert_table_constant(
            region, self._build_lookup_table(region)
        )
        sink = region.nodes[-1]
        sink.target = self.table_op
        sink.args = (region.input_node, table_node)
        sink.kwargs = {}

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for sink in reversed(list(graph.nodes)):
            if sink not in graph.nodes or not self._is_supported_node(sink):
                continue
            region = self._find_fusion_region(sink)
            if region is None:
                continue
            old_constants = {
                arg
                for node in region.nodes
                for arg in node.all_input_nodes
                if self._constant(arg) is not None
            }
            self._fuse_region(region)
            for node in reversed(region.nodes[:-1]):
                graph.erase_node(node)
            signature = self.exported_program.graph_signature
            protected = {
                spec.target
                for spec in signature.output_specs
                if spec.target is not None
            }
            for constant in old_constants:
                if not constant.users:
                    target = next(
                        spec.target
                        for spec in signature.input_specs
                        if spec.arg.name == constant.name
                    )
                    if target not in protected:
                        delete_constant_placeholder(self.exported_program, constant)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
