# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.experimental.symbolic_shapes import statically_known_true

_EDGE_GATHER = exir_ops.edge.aten.gather.default
_EDGE_CAST = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
_INT32_MAX = torch.iinfo(torch.int32).max


def _is_compatible_gather_user(boundary: torch.fx.Node, gather: torch.fx.Node) -> bool:
    if gather.target != _EDGE_GATHER:
        return False
    if len(gather.args) != 3 or gather.args[2] is not boundary:
        return False

    values = get_first_fake_tensor(cast(torch.fx.Node, gather.args[0]))
    indices = get_first_fake_tensor(boundary)
    values_shape = tuple(values.shape)
    indices_shape = tuple(indices.shape)
    if len(values_shape) not in (2, 3) or len(indices_shape) not in (2, 3):
        return False

    dim = cast(int, gather.args[1]) % len(values_shape)
    if dim != 1 or not statically_known_true(values_shape[dim] <= _INT32_MAX):
        return False
    if not statically_known_true(values_shape[0] == indices_shape[0]):
        return False
    return len(indices_shape) == 2 or statically_known_true(
        values_shape[-1] == indices_shape[-1]
    )


def is_safe_int32_to_int64_gather_boundary(node: torch.fx.Node) -> bool:
    """Return whether a cast is a removable gather compatibility boundary.

    Args:
        node: Candidate int32-to-int64 cast node.

    Returns:
        True when every user is a compatible, int32-bounded gather.

    """
    if node.target != _EDGE_CAST:
        return False
    if node.kwargs.get("dtype") != torch.int64:
        return False
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return False
    source_tensor = get_first_fake_tensor(source)
    if source_tensor.dtype != torch.int32:
        return False
    if source_tensor.dim_order() != get_first_fake_tensor(node).dim_order():
        return False
    if not node.users:
        return False
    return all(_is_compatible_gather_user(node, gather) for gather in node.users)


class PrepareGatherIndicesPass(ArmPass):
    """Remove safe int64 compatibility boundaries from gather indices.

    Portable ExecuTorch gather requires int64 indices, while TOSA gather uses
    int32. This pass runs during backend preprocessing, after delegation is
    established, and removes only an int32-to-int64 compatibility cast. It
    never narrows an arbitrary int64 tensor.

    Args:
        tosa_spec: Target capabilities used to gate gather
            value dtypes.

    """

    _passes_required_after: set[type[ExportPass]] = set()

    _edge_gather = _EDGE_GATHER

    def __init__(self, tosa_spec: TosaSpecification) -> None:
        super().__init__()
        self.tosa_spec = tosa_spec

    def _values_dtype_supported(self, dtype: torch.dtype) -> bool:
        if dtype in (torch.bool, torch.int8, torch.int16, torch.int32):
            return self.tosa_spec.support_integer()
        floating_profile = (
            self.tosa_spec.support_float() or self.tosa_spec.support_integer()
        )
        if dtype == torch.bfloat16:
            return floating_profile and self.tosa_spec.support_extension("bf16")
        if dtype == torch.float8_e4m3fn:
            return floating_profile and self.tosa_spec.support_extension("fp8e4m3")
        if dtype == torch.float8_e5m2:
            return floating_profile and self.tosa_spec.support_extension("fp8e5m2")
        if dtype in (torch.float16, torch.float32):
            return floating_profile
        return False

    def _is_supported_gather(self, node: torch.fx.Node) -> bool:
        if node.target != self._edge_gather:
            return False
        if len(node.args) != 3 or not isinstance(node.args[2], torch.fx.Node):
            return False

        values = get_first_fake_tensor(cast(torch.fx.Node, node.args[0]))
        indices = get_first_fake_tensor(node.args[2])
        if not self._values_dtype_supported(values.dtype):
            return False
        if indices.dtype != torch.int64 or not is_safe_int32_to_int64_gather_boundary(
            node.args[2]
        ):
            return False

        values_shape = tuple(values.shape)
        indices_shape = tuple(indices.shape)
        if len(values_shape) not in (2, 3):
            return False
        if len(indices_shape) not in (2, 3):
            return False

        dim = cast(int, node.args[1]) % len(values_shape)
        if dim != 1:
            return False
        if values_shape[0] != indices_shape[0]:
            return False
        return len(indices_shape) == 2 or values_shape[-1] == indices_shape[-1]

    @classmethod
    def _int32_input_before_boundary(
        cls, indices: torch.fx.Node
    ) -> torch.fx.Node | None:
        if not is_safe_int32_to_int64_gather_boundary(indices):
            return None
        return cast(torch.fx.Node, indices.args[0])

    def should_run_pass(self, graph_module: torch.fx.GraphModule) -> bool:
        """Return whether the graph contains a removable gather boundary."""
        return any(
            node.op == "call_function" and self._is_supported_gather(node)
            for node in graph_module.graph.nodes
        )

    def call(self, graph_module: torch.fx.GraphModule):
        """Remove safe compatibility casts from delegated gather inputs."""
        graph = graph_module.graph
        modified = False
        for gather in list(graph.nodes):
            if gather.op != "call_function" or not self._is_supported_gather(gather):
                continue

            indices = cast(torch.fx.Node, gather.args[2])
            int32_indices = self._int32_input_before_boundary(indices)
            if int32_indices is None:
                continue

            gather.replace_input_with(indices, int32_indices)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
