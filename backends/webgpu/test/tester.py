# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages

import torch
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.dialects._ops import ops as exir_ops

# Edge ops the WebGPU runtime implements; restricts the Vulkan partitioner.
WEBGPU_SUPPORTED_OPS = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.et_vk.rms_norm.default,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.select_copy.int,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.squeeze_copy.dims,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.slice_copy.Tensor,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.amax.default,
    exir_ops.edge.aten.amin.default,
    exir_ops.edge.aten.argmax.default,
    exir_ops.edge.aten.argmin.default,
    exir_ops.edge.aten.native_group_norm.default,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.full_like.default,
    exir_ops.edge.aten.zeros.default,
    exir_ops.edge.aten.zeros_like.default,
    exir_ops.edge.aten.ones.default,
    exir_ops.edge.aten.ones_like.default,
    exir_ops.edge.aten.scalar_tensor.default,
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.rsqrt.default,
    exir_ops.edge.aten.sin.default,
    exir_ops.edge.aten.cos.default,
    exir_ops.edge.aten.tanh.default,
    exir_ops.edge.aten.round.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.logical_and.default,
    exir_ops.edge.aten.bitwise_and.Tensor,
    exir_ops.edge.aten.bitwise_not.default,
    exir_ops.edge.aten.flip.default,
    exir_ops.edge.aten.repeat.default,
    exir_ops.edge.aten.index_select.default,
    exir_ops.edge.aten.avg_pool2d.default,
    exir_ops.edge.aten.pixel_shuffle.default,
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.et_vk.conv_with_clamp.default,
    exir_ops.edge.aten.grid_sampler_2d.default,
    exir_ops.edge.et_vk.grid_priors.default,
    exir_ops.edge.et_vk.linear_qcs4w.default,
    exir_ops.edge.et_vk.linear_q8ta_q8csw.default,
    exir_ops.edge.et_vk.q8ta_add.default,
    exir_ops.edge.et_vk.q8ta_relu.default,
    exir_ops.edge.et_vk.q8ta_pixel_shuffle.default,
    exir_ops.edge.et_vk.q8ta_linear.default,
    exir_ops.edge.et_vk.q8ta_linear_gemv.default,
    exir_ops.edge.et_vk.q8ta_conv2d.default,
    exir_ops.edge.et_vk.q8ta_conv2d_dw.default,
    exir_ops.edge.et_vk.q8ta_conv2d_pw.default,
    exir_ops.edge.et_vk.linear_dq8ca_q4gsw.default,
    exir_ops.edge.torchao.choose_qparams_affine.default,
]


# Lowers via VulkanPartitioner (WebGPU consumes the Vulkan VK00 serialization),
# restricted to the ops the WebGPU runtime implements.
class Partition(BaseStages.Partition):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        super().__init__(
            partitioner=partitioner
            or VulkanPartitioner(
                {"skip_bool_tensors": True},
                operator_allowlist=WEBGPU_SUPPORTED_OPS,
            ),
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        if partitioners is None:
            partitioners = [
                VulkanPartitioner(
                    {"skip_bool_tensors": True},
                    operator_allowlist=WEBGPU_SUPPORTED_OPS,
                )
            ]

        super().__init__(
            default_partitioner_cls=VulkanPartitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config
            or EdgeCompileConfig(_check_ir_validity=False),
        )


class WebGPUTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
    ):
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.PARTITION: Partition,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
