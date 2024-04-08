# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
import os
from typing import final, List

import torch
from executorch.backends.arm.arm_backend import ArmBackend
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


class TOSASupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten._softmax.default,
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.clone.default,
            exir_ops.edge.aten.mean.dim,
            operator.getitem,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        ]

        supported &= self.is_node_supported_custom(node)

        # Override partitioning based on pre partition passes
        if supported and "arm_partition" in node.meta:
            supported = supported & node.meta["arm_partition"]

        return supported

    def is_node_supported_custom(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.edge.aten.mean.dim:
            dim = node.args[1]
            keep_dim = node.args[2]
            if dim != [-1, -2] or keep_dim is False:
                return False
        return True


from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import PassManager


class TagIOQuant(ExportPass):
    """
    Pass run before partitioning to tag Q/DQ on any placeholder and output
    to ensure we don't greedily partition them for device. Float conversion
    has to happen outside a TOSA base inference profile.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(TagIOQuant, self).__init__()
        self.edge_program = edge_program

    def is_quant_node(self, node: torch.fx.node.Node):
        return node.target in {
            exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        }

    def is_dequant_node(self, node: torch.fx.node.Node):
        return node.target in {
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            # tag q of input
            if node.op == "placeholder":
                for user in node.users.keys():
                    # if we have an input going into a quantize
                    if self.is_quant_node(user):
                        user.meta["arm_partition"] = False

            # tag dq of outputs
            if node.op == "output":
                quant, *_ = node.args[0]
                if self.is_dequant_node(quant):
                    quant.meta["arm_partition"] = False

        graph_module.recompile()
        return PassResult(graph_module, True)


@final
class ArmPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(ArmBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("ArmPartitioner::partition")
        partition_tags = {}

        for spec in self.delegation_spec.compile_specs:
            if spec.key == "quantize_io" and spec.value.decode() == "True":
                # Exclude IO quantization from the partition
                passes = PassManager(
                    passes=[
                        TagIOQuant(exported_program),
                    ]
                )
                passes(exported_program.graph_module)

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            TOSASupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
