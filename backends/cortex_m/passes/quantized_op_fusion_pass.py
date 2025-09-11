# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Set

import executorch.backends.cortex_m.ops.operators  # noqa
import torch

from executorch.backends.cortex_m.passes.passes_utils import (
    extract_scalar_value,
    quantize_multiplier_aot,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger("quant_op_fusion_pass")
logger.setLevel(logging.INFO)


class QuantizedOpFusionPass(ExportPass):
    """
    Generic ExportPass that:
    1. Replaces certain ops with cortex_m variants based on qualifiers.
    2. Fuses patterns: dequantize_per_tensor -> [binary_op] -> quantize_per_tensor
       into cortex_m.quantized_[op].default with AoT computed multipliers/shifts.


    Supports multiple binary operations with backward compatibility for add.
    """

    # Generic operation mapping
    SUPPORTED_OPS_MAPPING = {
        exir_ops.edge.aten.add.Tensor: exir_ops.edge.cortex_m.quantized_add.default,
        # Future ops to be added here:
    }

    def __init__(self):
        super().__init__()

    def _get_dequant_targets(self) -> Set:
        """Support both decomposed and cortex_m dequant targets for flexible pass ordering."""
        return {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.cortex_m.dequantize_per_tensor.default,
        }

    def _get_quant_targets(self) -> Set:
        """Support both decomposed and cortex_m quant targets for flexible pass ordering."""
        return {
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.cortex_m.quantize_per_tensor.default,
        }

    def _is_supported_binary_op(self, node: torch.fx.Node) -> bool:
        """Check if node is a supported binary operation."""
        return node.op == "call_function" and node.target in self.SUPPORTED_OPS_MAPPING

    def _is_dequant_node(self, node: torch.fx.Node) -> bool:
        """Check if node is a dequantize operation."""
        return (
            hasattr(node, "op")
            and node.op == "call_function"
            and node.target in self._get_dequant_targets()
        )

    def _is_quant_node(self, node: torch.fx.Node) -> bool:
        """Check if node is a quantize operation."""
        return (
            hasattr(node, "op")
            and node.op == "call_function"
            and node.target in self._get_quant_targets()
        )

    def _transfer_metadata(
        self,
        new_node: torch.fx.Node,
        source_node: torch.fx.Node,
        pass_name: str = "QuantizedOpFusionPass",
    ) -> None:
        """Metadata transfer with proper provenance tracking."""
        if hasattr(source_node, "meta") and source_node.meta:
            new_node.meta = source_node.meta.copy()

            if "from_node" in new_node.meta:
                from_node_list = new_node.meta.get("from_node", []).copy()
                from_node_list.append(
                    {"source": source_node.name, "pass": pass_name, "op": "fuse"}
                )
                new_node.meta["from_node"] = from_node_list

            # Copy essential fields
            for field in ["tensor_meta", "stack_trace"]:
                if field in source_node.meta:
                    new_node.meta[field] = source_node.meta[field]

    def _normalize_to_cortex_m_targets(self, graph_module: torch.fx.GraphModule) -> int:
        """Convert decomposed targets to cortex_m equivalents for consistent handling."""
        target_mapping = {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: exir_ops.edge.cortex_m.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: exir_ops.edge.cortex_m.quantize_per_tensor.default,
        }

        normalization_count = 0
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and node.target in target_mapping:
                logger.info(f"Normalizing {node.target} to cortex_m equivalent")
                node.target = target_mapping[node.target]
                normalization_count += 1

        return normalization_count

    def _fuse_quantized_binary_patterns(
        self, graph_module: torch.fx.GraphModule
    ) -> int:
        """Generic fusion for quantized binary operation patterns."""
        fusion_count = 0
        nodes_to_erase = []

        for node in list(graph_module.graph.nodes):
            if not self._is_quant_node(node):
                continue

            quantize_node = node
            if not quantize_node.args:
                continue

            binary_op_node = quantize_node.args[0]
            if not self._is_supported_binary_op(binary_op_node):
                continue

            if len(binary_op_node.args) < 2:
                continue

            dequant_node1, dequant_node2 = binary_op_node.args[:2]
            if not (
                self._is_dequant_node(dequant_node1)
                and self._is_dequant_node(dequant_node2)
            ):
                continue

            # Get the target quantized operation
            quantized_target = self.SUPPORTED_OPS_MAPPING[binary_op_node.target]
            # Extract op name (e.g., 'Tensor' -> 'add')
            op_name = str(binary_op_node.target).split(".")[-1]
            logger.info(f"✅ Found complete cortex_m Q/DQ + {op_name} pattern!")

            try:
                # Extract values
                int8_tensor1, scale1, zero_point1 = dequant_node1.args[:3]
                int8_tensor2, scale2, zero_point2 = dequant_node2.args[:3]
                output_scale, output_zero_point = quantize_node.args[1:3]

                # Convert to Python floats
                scale1_val = extract_scalar_value(scale1)
                scale2_val = extract_scalar_value(scale2)
                output_scale_val = extract_scalar_value(output_scale)
                zp1_val = int(extract_scalar_value(zero_point1))
                zp2_val = int(extract_scalar_value(zero_point2))
                output_zp_val = int(extract_scalar_value(output_zero_point))

                # AoT COMPUTATION: Calculate multipliers and shifts
                input1_mult, input1_shift = quantize_multiplier_aot(
                    scale1_val / output_scale_val
                )
                input2_mult, input2_shift = quantize_multiplier_aot(
                    scale2_val / output_scale_val
                )
                output_mult, output_shift = quantize_multiplier_aot(
                    1.0
                )  # Output multiplier is 1

                logger.info("AoT computed parameters:")
                logger.info(f"   Input1: mult={input1_mult}, shift={input1_shift}")
                logger.info(f"   Input2: mult={input2_mult}, shift={input2_shift}")
                logger.info(f"   Output: mult={output_mult}, shift={output_shift}")

                with graph_module.graph.inserting_after(quantize_node):
                    fused = graph_module.graph.create_node(
                        "call_function",
                        target=quantized_target,
                        args=(
                            int8_tensor1,
                            zp1_val,
                            input1_mult,
                            input1_shift,
                            int8_tensor2,
                            zp2_val,
                            input2_mult,
                            input2_shift,
                            output_zp_val,
                            output_mult,
                            output_shift,
                        ),
                        kwargs={},
                    )

                    # metadata transfer
                    self._transfer_metadata(fused, quantize_node)

                logger.info(f"✅ Created fused quantized_{op_name} node: {fused}")

                # Replace all uses
                quantize_node.replace_all_uses_with(fused)
                binary_op_node.replace_all_uses_with(fused)
                dequant_node1.replace_all_uses_with(fused)
                dequant_node2.replace_all_uses_with(fused)

                nodes_to_erase.extend(
                    [quantize_node, binary_op_node, dequant_node1, dequant_node2]
                )
                fusion_count += 1
                logger.info(f"Pattern fused, total so far: {fusion_count}")

            except Exception as e:
                logger.info(f"❌ Error during AoT computation: {e}")
                logger.info("   Skipping fusion for this pattern")
                continue

        for old_node in reversed(nodes_to_erase):
            if old_node in graph_module.graph.nodes and len(old_node.users) == 0:
                logger.info(f"🗑️ Erasing node: {old_node}")
                graph_module.graph.erase_node(old_node)

        return fusion_count

    def call(self, graph_module: torch.fx.GraphModule):
        logger.info("QuantizedOpFusionPass.call() started")

        # Normalize targets for flexible pass ordering
        normalization_count = self._normalize_to_cortex_m_targets(graph_module)

        # Generic fusion for supported binary operations
        fusion_count = self._fuse_quantized_binary_patterns(graph_module)

        total_changes = normalization_count + fusion_count
        logger.info(f"Total changes: {total_changes}")

        if total_changes > 0:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()

        logger.debug("=== AFTER FUSION: All nodes in the graph ===")
        for i, node in enumerate(graph_module.graph.nodes):
            logger.debug(f"Node {i}: op={node.op}, target={node.target}")
            if "quantized_" in str(node.target) and "add" in str(node.target):
                logger.debug(" ⭐ FOUND QUANTIZED BINARY OP NODE! ⭐")
        logger.debug("=== END DEBUG ===")

        return PassResult(graph_module, total_changes > 0)
