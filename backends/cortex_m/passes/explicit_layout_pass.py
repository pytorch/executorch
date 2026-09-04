# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch
from executorch.backends.cortex_m.passes.passes_utils import (
    coerce_int_pair,
    skips_quantized_max_pool2d,
)
from executorch.backends.transforms.canonicalize_view_copy_permute_pass import (
    CanonicalizeViewCopyPermutePass,
)
from executorch.backends.transforms.channels_last_layout import (
    LAYOUT_PERMUTE_COPY,
    PERMUTE_COPY_TARGETS,
)
from executorch.backends.transforms.replace_ops_with_channels_last_variants import (
    ChannelsLastOpSpec,
    ReplaceOpsWithChannelsLastVariants,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torch.fx.node import Target


def _is_rank4(node: Node) -> bool:
    return len(node.meta["val"].shape) == 4


def _has_per_tensor_qparam(node: Node, key: str, index: int) -> bool:
    qparam = node.meta.get(key, {}).get(index)
    return qparam is not None and not getattr(qparam, "per_channel", False)


def _has_input_and_output_qparams(node: Node) -> bool:
    return _has_per_tensor_qparam(node, "input_qparams", 0) and _has_per_tensor_qparam(
        node, "output_qparams", 0
    )


def _supports_avg_pool2d(node: Node) -> bool:
    divisor_override = node.args[6] if len(node.args) > 6 else None
    return (
        _is_rank4(node)
        and _has_input_and_output_qparams(node)
        and divisor_override is None
    )


def _supports_max_pool2d(node: Node) -> bool:
    if not _is_rank4(node) or not _has_per_tensor_qparam(node, "input_qparams", 0):
        return False
    if skips_quantized_max_pool2d(node):
        return False

    dilation = coerce_int_pair(node.args[4] if len(node.args) > 4 else None, (1, 1))
    ceil_mode = bool(node.args[5]) if len(node.args) > 5 else False
    if dilation != (1, 1) or ceil_mode:
        return False

    input_qparams = node.meta["input_qparams"][0]
    output_qparams = node.meta.get("output_qparams", {}).get(0)
    return output_qparams is None or (
        not getattr(output_qparams, "per_channel", False)
        and abs(float(input_qparams.scale) - float(output_qparams.scale)) <= 1e-6
        and int(input_qparams.zp) == int(output_qparams.zp)
    )


_EXPLICIT_LAYOUT_OP_MAP: dict[Target, ChannelsLastOpSpec] = {
    exir_ops.edge.aten.convolution.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.convolution.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=lambda node: _is_rank4(node) and _has_input_and_output_qparams(node),
    ),
    exir_ops.edge.aten.avg_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.avg_pool2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_supports_avg_pool2d,
    ),
    exir_ops.edge.aten.max_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.max_pool2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_supports_max_pool2d,
    ),
}


class CortexMReplaceOpsWithChannelsLastVariants(ReplaceOpsWithChannelsLastVariants):
    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__(exported_program, op_map=dict(_EXPLICIT_LAYOUT_OP_MAP))


class CortexMCanonicalizeViewCopyPermutePass(CanonicalizeViewCopyPermutePass):
    def __init__(self) -> None:
        super().__init__(permute_targets=PERMUTE_COPY_TARGETS)

    def _set_node_op(self, node, target, input_node, arg) -> None:
        super()._set_node_op(node, target, input_node, arg)
        if node.target != LAYOUT_PERMUTE_COPY:
            return
        input_value = input_node.meta.get("val")
        if isinstance(input_value, torch.Tensor):
            dims = [int(dim) % input_value.dim() for dim in arg]
            node.meta["val"] = input_value.new_empty(
                tuple(input_value.shape[dim] for dim in dims)
            )


class ValidateCortexMExplicitLayoutPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.target in _EXPLICIT_LAYOUT_OP_MAP:
                raise RuntimeError(
                    "Cortex-M explicit layout requires every quantized spatial "
                    f"operator to be NHWC-eligible, but {node.target} was not. "
                    "Use the legacy layout pipeline for this model."
                )

            if node.target != LAYOUT_PERMUTE_COPY:
                continue
            input_node = node.args[0]
            dims = node.args[1] if len(node.args) > 1 else None
            input_value = (
                input_node.meta.get("val") if isinstance(input_node, Node) else None
            )
            if (
                not isinstance(input_value, torch.Tensor)
                or input_value.dtype != torch.int8
            ):
                raise RuntimeError(
                    f"Cortex-M layout copy {node.name} must move an int8 tensor."
                )
            rank = input_value.dim()
            if (
                not 1 <= rank <= 4
                or not isinstance(dims, (list, tuple))
                or len(dims) != rank
                or not all(isinstance(dim, int) for dim in dims)
                or sorted(dim % rank for dim in dims) != list(range(rank))
            ):
                raise RuntimeError(
                    f"Cortex-M layout copy {node.name} has invalid permutation "
                    f"{dims!r} for rank {rank}."
                )

        return PassResult(graph_module, False)
