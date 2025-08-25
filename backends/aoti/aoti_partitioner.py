# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator
from typing import Callable, cast, Dict, final, List, Optional, Set, Tuple

import torch
from executorch.backends.aoti.aoti_backend import AotiBackend  # usort: skip
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase

# exist fallback operators in et namespace; should map to inductor_fallback_ops
supported_fallback_operators: Dict[str, Dict[str, List[str]]] = {}

inductor_fallback_ops: Set[str] = {
    "aten._adaptive_avg_pool2d_backward.default",
    "aten._adaptive_avg_pool2d.default",
    "aten._adaptive_avg_pool3d_backward.default",
    "aten._adaptive_avg_pool3d.default",
    "aten._addmm_activation.default",
    "aten._cdist_backward.default",
    "aten._cdist_forward.default",
    "aten._cudnn_rnn.default",
    "aten._dyn_quant_matmul_4bit.default",
    "aten._dyn_quant_pack_4bit_weight.default",
    "aten._efficient_attention_backward.default",
    "aten._efficient_attention_forward.default",
    "aten._efficientzerotensor.default",
    "aten._embedding_bag_dense_backward.default",
    "aten._embedding_bag_forward_only.default",
    "aten._embedding_bag_per_sample_weights_backward.default",
    "aten._embedding_bag.default",
    "aten._fft_c2c.default",
    "aten._fft_r2c.default",
    "aten._flash_attention_backward.default",
    "aten._flash_attention_forward.default",
    "aten._fused_moving_avg_obs_fq_helper_functional.default",
    "aten._fused_moving_avg_obs_fq_helper.default",
    "aten._fused_rms_norm.default",
    "aten._histogramdd_from_bin_cts.default",
    "aten._int_mm.out",
    "aten._pdist_backward.default",
    "aten._pdist_forward.default",
    "aten._scaled_dot_product_attention_math_for_mps.default",
    "aten._scaled_dot_product_cudnn_attention_backward.default",
    "aten._scaled_dot_product_cudnn_attention.default",
    "aten._scaled_dot_product_efficient_attention_backward.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten._scaled_dot_product_flash_attention_backward.default",
    "aten._scaled_dot_product_flash_attention_for_cpu_backward.default",
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_fused_attention_overrideable_backward.default",
    "aten._scaled_dot_product_fused_attention_overrideable.default",
    "aten._scaled_mm.default",
    "aten._scaled_mm.out",
    "aten._segment_reduce_backward.default",
    "aten._thnn_fused_lstm_cell.default",
    "aten._to_sparse.default",
    "aten._trilinear.default",
    "aten._weight_int4pack_mm.default",
    "aten._weight_int8pack_mm.default",
    "aten.abs.default",
    "aten.adaptive_max_pool2d_backward.default",
    "aten.adaptive_max_pool2d.default",
    "aten.adaptive_max_pool3d_backward.default",
    "aten.adaptive_max_pool3d.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.addbmm.default",
    "aten.addmm.out",
    "aten.addmv.default",
    "aten.angle.default",
    "aten.avg_pool2d_backward.default",
    "aten.avg_pool2d.default",
    "aten.avg_pool3d_backward.default",
    "aten.avg_pool3d.default",
    "aten.baddbmm.out",
    "aten.bernoulli_.float",
    "aten.bernoulli_.Tensor",
    "aten.bmm.out",
    "aten.bucketize.Tensor",
    "aten.cat.default",
    "aten.cholesky_inverse.default",
    "aten.cholesky_solve.default",
    "aten.convolution_backward.default",
    "aten.convolution.default",
    "aten.cummax.default",
    "aten.cummin.default",
    "aten.cumprod.default",
    "aten.cumsum.default",
    "aten.exponential.default",
    "aten.fill_.Scalar",
    "aten.fractional_max_pool2d_backward.default",
    "aten.fractional_max_pool2d.default",
    "aten.fractional_max_pool3d_backward.default",
    "aten.fractional_max_pool3d.default",
    "aten.gcd.default",
    "aten.geqrf.default",
    "aten.grid_sampler_2d_backward.default",
    "aten.hann_window.default",
    "aten.histc.default",
    "aten.histogram.bin_ct",
    "aten.index_put.default",
    "aten.index_reduce.default",
    "aten.index.Tensor",
    "aten.kthvalue.default",
    "aten.logcumsumexp.default",
    "aten.lu_unpack.default",
    "aten.masked_scatter_backward.default",
    "aten.masked_scatter.default",
    "aten.masked_select.default",
    "aten.max_pool2d_with_indices_backward.default",
    "aten.max_pool2d_with_indices.default",
    "aten.max_pool3d_with_indices_backward.default",
    "aten.max_pool3d_with_indices.default",
    "aten.max_unpool2d.default",
    "aten.max_unpool3d.default",
    "aten.median.default",
    "aten.mm.out",
    "aten.mode.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.nanmedian.default",
    "aten.narrow.default",
    "aten.native_dropout.default",
    "aten.nonzero.default",
    "aten.normal_functional.default",
    "aten.ormqr.default",
    "aten.pad.default",
    "aten.permute.default",
    "aten.polar.default",
    "aten.pow.Scalar",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Tensor",
    "aten.rand.default",
    "aten.rand.generator",
    "aten.randint.default",
    "aten.randint.generator",
    "aten.randint.low_out",
    "aten.randint.low",
    "aten.randn.default",
    "aten.randn.generator",
    "aten.randperm.default",
    "aten.repeat_interleave.Tensor",
    "aten.replication_pad1d_backward.default",
    "aten.replication_pad2d_backward.default",
    "aten.reshape.default",
    "aten.resize_.default",
    "aten.resize_as_.default",
    "aten.scatter_reduce.two_out",
    "aten.scatter.src_out",
    "aten.scatter.value_out",
    "aten.searchsorted.Scalar",
    "aten.searchsorted.Tensor",
    "aten.segment_reduce.default",
    "aten.set_.source_Tensor",
    "aten.slice.Tensor",
    "aten.soft_margin_loss_backward.default",
    "aten.sort.default",
    "aten.sort.stable",
    "aten.squeeze.dim",
    "aten.to_sparse.default",
    "aten.topk.default",
    "aten.triangular_solve.default",
    "aten.uniform.default",
    "aten.upsample_bicubic2d_backward.default",
    "aten.upsample_linear1d_backward.default",
    "aten.upsample_trilinear3d_backward.default",
    "aten.view_as_complex.default",
    "aten.view_as_real.default",
    "aten.view.dtype",
    "aten._weight_int4pack_mm_with_scales_and_zeros.default",
}


class AOTISupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # supported = node.op == "call_function" and (
        #     node.target == operator.getitem
        #     or str(node.target._op) not in inductor_fallback_ops
        #     or str(node.target._op) in supported_fallback_operators
        # )

        supported = node.op == "call_function"

        return supported

    def is_node_supported_custom(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.edge.aten.mean.dim:
            keep_dim = node.args[2] if len(node.args) > 2 else False
            return cast(bool, keep_dim)
        if node.target == exir_ops.edge.aten.var.correction:
            keep_dim = node.kwargs.get("keepdim", False)
            return cast(bool, keep_dim)
        return True


@final
class AotiPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(AotiBackend.__name__, compile_spec)
        print(self.delegation_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        # logger.info("AotiPartitioner::partition")
        print("entering partitioner...")

        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            AOTISupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        assert len(partition_list) == 1, "Graph break is not supported yet"

        print(f"graph breaks into {len(partition_list)} parts")

        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Return a list of operations that should not be decomposed and let the AOT compiler handle them.
        """
        do_not_decompose = set()
        op_support = AOTISupportedOperators()

        for node in ep.graph.nodes:
            if (
                node.op == "call_function"
                and isinstance(node.target, torch._ops.OpOverload)
                and op_support.is_node_supported(None, node)
            ):
                do_not_decompose.add(node.target)
        return list(do_not_decompose), None
