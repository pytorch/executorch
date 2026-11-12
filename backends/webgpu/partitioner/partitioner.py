# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Callable, Dict, final, List, Optional, Tuple

import executorch.backends.vulkan.utils as vk_utils

import torch
from executorch.backends.vulkan.op_registry import (
    allow_node,
    get_op_features,
    OpFeatures,
    OpKey,
    vulkan_supported_ops,
)
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
    VulkanSupportedOperators,
)
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


def _arange_bounds_are_static(node: torch.fx.Node) -> bool:
    # A SymInt start/end/step makes the length dynamic, which the WebGPU kernel
    # does not resize for.
    return not any(isinstance(arg, torch.fx.Node) for arg in node.args[:3])


def _webgpu_features(
    target: OpKey,
    check: Callable[[torch.fx.Node, Callable[[torch.fx.Node], bool]], bool],
) -> OpFeatures:
    """A copy of the Vulkan features of `target` whose input check is
    `check(node, vulkan_check)`."""
    vulkan_features = vulkan_supported_ops[target]
    vulkan_check = vulkan_features.are_node_inputs_supported_fn or allow_node
    features = copy.copy(vulkan_features)
    features.are_node_inputs_supported_fn = lambda node: check(node, vulkan_check)
    return features


def _layer_norm_check(
    node: torch.fx.Node, vulkan_check: Callable[[torch.fx.Node], bool]
) -> bool:
    # The kernel normalizes over the last dim only; leading normalized dims
    # are allowed only when they are 1.
    normalized_shape = node.args[1]
    if not isinstance(normalized_shape, (list, tuple)):
        return False
    return all(d == 1 for d in normalized_shape[:-1]) and vulkan_check(node)


def _fp32_unary_features() -> OpFeatures:
    return OpFeatures(
        inputs_storage=vk_utils.ANY_STORAGE,
        inputs_dtypes={torch.float32},
        supports_resize=True,
    )


def _fp32_only(op: OpKey) -> OpFeatures:
    # The registry's features for the op, limited to fp32 inputs and outputs.
    features = copy.copy(get_op_features(op))
    features.inputs_dtypes = vk_utils.DtypeSetList({torch.float32})
    features.outputs_dtypes = vk_utils.DtypeSetList({torch.float32})
    return features


# Ops whose WebGPU kernels read and write array<f32>. The Vulkan registry also
# advertises fp16 (and, for clamp, int32) for them, which these kernels would
# misread or refuse.
_FP32_ONLY_OPS: List[OpKey] = [
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.cos.default,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.gelu.default,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.leaky_relu.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.pixel_shuffle.default,
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.round.default,
    exir_ops.edge.aten.rsqrt.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.sin.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.tanh.default,
    exir_ops.edge.aten.upsample_nearest2d.vec,
]

# Ops whose WebGPU kernels cover less (or more) than the Vulkan registry
# advertises, so unsupported forms stay on CPU instead of failing at delegate
# init, and WebGPU-only kernels are delegated.
_WEBGPU_OP_FEATURES: Dict[OpKey, OpFeatures] = {
    **{op: _fp32_only(op) for op in _FP32_ONLY_OPS},
    # fp32 and int32 kernels only; int64 is downcast to int32 on device.
    exir_ops.edge.aten.arange.start_step: OpFeatures(
        inputs_storage=vk_utils.ANY_STORAGE,
        inputs_dtypes={torch.float32, torch.int32, torch.int64},
        are_node_inputs_supported_fn=_arange_bounds_are_static,
    ),
    exir_ops.edge.aten.native_layer_norm.default: _webgpu_features(
        exir_ops.edge.aten.native_layer_norm.default, _layer_norm_check
    ),
    exir_ops.edge.aten.elu.default: _fp32_unary_features(),
    exir_ops.edge.aten.silu.default: _fp32_unary_features(),
}


def _edge_op(op: torch._ops.OpOverload) -> Optional[OpKey]:
    """The edge dialect form of the aten op `op`, or None for another op."""
    if op.namespace != "aten":
        return None
    try:
        return getattr(getattr(exir_ops.edge.aten, op._opname), op._overloadname)
    except AttributeError:
        return None


# Vulkan fusion patterns whose fused ops the WebGPU runtime does not implement.
_WEBGPU_SKIP_FUSION_PATTERNS = {"conv2d_with_epilogue"}

_MUL_TARGETS = (torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor)


@final
class WebGPUPartitioner(Partitioner):
    """WebGPU frontend for the Vulkan partitioning and serialization path."""

    def __init__(
        self,
        compile_options: Optional[Dict[str, Any]] = None,
        operator_blocklist: Optional[List[OpKey]] = None,
        operator_allowlist: Optional[List[OpKey]] = None,
        nn_module_blocklist: Optional[List[str]] = None,
        nn_module_allowlist: Optional[List[str]] = None,
    ) -> None:
        compile_options = dict(compile_options or {})
        compile_options["skip_fusion_patterns"] = {
            *compile_options.get("skip_fusion_patterns", ()),
            *_WEBGPU_SKIP_FUSION_PATTERNS,
        }
        self._vulkan_partitioner = VulkanPartitioner(
            compile_options=compile_options,
            operator_blocklist=operator_blocklist,
            operator_allowlist=operator_allowlist,
            nn_module_blocklist=nn_module_blocklist,
            nn_module_allowlist=nn_module_allowlist,
            op_features_overrides=_WEBGPU_OP_FEATURES,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        ops, vulkan_filter = self._vulkan_partitioner.ops_to_not_decompose(ep)
        ops = ops + [torch.ops.aten.silu.default]
        # Kept ops whose WebGPU support differs from Vulkan's, by their aten op.
        overridden = {
            op: edge_op
            for op in ops
            if (edge_op := _edge_op(op)) in _WEBGPU_OP_FEATURES
        }
        support = self._kept_op_support(overridden)
        # filter_fn sees the aten form, and the edge form when lowering checks
        # what was left undelegated.
        overridden_targets = set(overridden) | set(overridden.values())

        def filter_fn(node: torch.fx.Node) -> bool:
            if node.target in (
                torch.ops.aten.silu.default,
                exir_ops.edge.aten.silu.default,
            ):
                return self._keeps_silu(node, ep, support)
            if node.target in overridden_targets:
                return self._is_delegated_where_kept(node, ep, support)
            return vulkan_filter is None or vulkan_filter(node)

        return ops, filter_fn

    def _kept_op_support(
        self, overridden: Dict[torch._ops.OpOverload, OpKey]
    ) -> VulkanSupportedOperators:
        """The partitioner's own node support check, also accepting the aten
        forms of the kept ops in `overridden` that ops_to_not_decompose sees
        before the edge conversion."""
        vulkan = self._vulkan_partitioner
        allowlist = vulkan.operator_allowlist
        if allowlist is not None:
            allowlist = allowlist | {
                op for op, edge_op in overridden.items() if edge_op in allowlist
            }
        blocklist = vulkan.operator_blocklist
        blocked = {op for op, edge_op in overridden.items() if edge_op in blocklist}
        if blocked:
            blocklist = blocklist | blocked
        return vulkan.make_supported_operators(
            op_features_overrides={
                **_WEBGPU_OP_FEATURES,
                **{
                    op: _WEBGPU_OP_FEATURES[edge_op]
                    for op, edge_op in overridden.items()
                },
            },
            operator_allowlist=allowlist,
            operator_blocklist=blocklist,
        )

    def _is_delegated_where_kept(
        self,
        node: torch.fx.Node,
        ep: ExportedProgram,
        support: VulkanSupportedOperators,
    ) -> bool:
        """Whether the partitioner will delegate `node` if it is kept whole. A
        kept op that is not delegated fails lowering, so keep it only in the
        top-level graph (control-flow submodules are not partitioned) and where
        it passes the partitioner's support check, e.g. fp32 for a kernel that
        reads fp32. Elsewhere it decomposes as usual."""
        return node.graph is ep.graph_module.graph and support.is_node_supported(
            {}, node
        )

    def _keeps_silu(
        self,
        node: torch.fx.Node,
        ep: ExportedProgram,
        support: VulkanSupportedOperators,
    ) -> bool:
        """Whether `node` stays an aten.silu for the WebGPU silu kernel: only
        where it will be delegated (_is_delegated_where_kept). Where it feeds a
        mul it decomposes to x * sigmoid(x), as before, so the SwiGLU fusion
        still matches.
        """
        return not any(
            user.target in _MUL_TARGETS for user in node.users
        ) and self._is_delegated_where_kept(node, ep, support)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return self._vulkan_partitioner.partition(exported_program)
