# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Any, Callable, Dict, final, List, Optional, Tuple

import executorch.backends.vulkan.utils as vk_utils

import torch
from executorch.backends.vulkan.op_registry import get_op_features, OpFeatures, OpKey
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


def _arange_bounds_are_static(node: torch.fx.Node) -> bool:
    # A SymInt start/end/step makes the length dynamic, which the WebGPU kernel
    # does not resize for.
    return not any(isinstance(arg, torch.fx.Node) for arg in node.args[:3])


def _fp32_only(op: OpKey) -> OpFeatures:
    # The registry's features for the op, limited to fp32 inputs and outputs.
    features = copy.copy(get_op_features(op))
    features.inputs_dtypes = vk_utils.DtypeSetList({torch.float32})
    features.outputs_dtypes = vk_utils.DtypeSetList({torch.float32})
    return features


# Elementwise ops whose WebGPU kernels read and write array<f32>. The Vulkan
# registry also advertises fp16 for them, which these kernels would misread.
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
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.round.default,
    exir_ops.edge.aten.rsqrt.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.sin.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.tanh.default,
]

# Ops whose WebGPU kernels cover less than the Vulkan registry advertises, so the
# unsupported forms stay on CPU instead of failing at delegate init.
_WEBGPU_OP_FEATURES: Dict[OpKey, OpFeatures] = {
    **{op: _fp32_only(op) for op in _FP32_ONLY_OPS},
    # fp32 and int32 kernels only; int64 is downcast to int32 on device.
    exir_ops.edge.aten.arange.start_step: OpFeatures(
        inputs_storage=vk_utils.ANY_STORAGE,
        inputs_dtypes={torch.float32, torch.int32, torch.int64},
        are_node_inputs_supported_fn=_arange_bounds_are_static,
    ),
}


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
        return self._vulkan_partitioner.ops_to_not_decompose(ep)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return self._vulkan_partitioner.partition(exported_program)
