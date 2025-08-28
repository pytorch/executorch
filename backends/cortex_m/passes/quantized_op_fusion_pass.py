# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Set
import numpy as np

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
    Generic ExportPass that supports:
    1. Binary ops: dequantize_per_tensor -> [add] -> quantize_per_tensor
    2. Linear ops: dequantize_per_tensor -> [linear] -> quantize_per_tensor
    """

    # Mapping of supported ops to their corresponding quantized op
    SUPPORTED_OPS_MAPPING = {
        # Binary operations
        exir_ops.edge.aten.add.Tensor: exir_ops.edge.cortex_m.quantized_add.default,

        # Linear operations (now properly defined)
        exir_ops.edge.aten.linear.default: torch.ops.cortex_m.quantized_linear.default,
        exir_ops.edge.aten.addmm.default: torch.ops.cortex_m.quantized_linear.default,
    }

    def __init__(self):
        super().__init__()

    def _get_dequant_targets(self) -> Set:
        """Support both decomposed and cortex_m dequant targets."""
        return {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.cortex_m.dequantize_per_tensor.default,
        }

    def _get_quant_targets(self) -> Set:
        """Support both decomposed and cortex_m quant targets."""
        return {
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.cortex_m.quantize_per_tensor.default,
        }

    def _is_supported_op(self, node: torch.fx.Node) -> bool:
        """Check if node is a supported operation (binary or linear)."""
        return node.op == "call_function" and node.target in self.SUPPORTED_OPS_MAPPING

    def _is_binary_op(self, node: torch.fx.Node) -> bool:
        """Check if node is a binary operation (add, sub, mul, etc.)."""
        binary_ops = {exir_ops.edge.aten.add.Tensor}
        return node.op == "call_function" and node.target in binary_ops

    def _is_linear_op(self, node: torch.fx.Node) -> bool:
        """Check if node is a linear operation."""
        linear_ops = {
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.addmm.default,
        }
        return node.op == "call_function" and node.target in linear_ops

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

    def _extract_weight_bias_from_linear(self, linear_node: torch.fx.Node):
        """Extract weight tensor and bias from linear operation."""
        # linear(input, weight, bias) format
        if len(linear_node.args) >= 2:
            weight_arg = linear_node.args[1]  # weight tensor
            bias_arg = linear_node.args[2] if len(linear_node.args) > 2 else None

            # Check if weight comes from dequantize node
            weight_dq_node = None
            bias_dq_node = None

            if self._is_dequant_node(weight_arg):
                weight_dq_node = weight_arg

            if bias_arg and self._is_dequant_node(bias_arg):
                bias_dq_node = bias_arg

            return weight_dq_node, bias_dq_node
        return None, None

    def _precompute_linear_params_aot(self, weight_dq_node, input_scale_val, output_scale_val):
        """AOT precomputation for linear layer parameters."""
        if not weight_dq_node or len(weight_dq_node.args) < 3:
            return None

        # Extract weight quantization parameters
        weight_tensor = weight_dq_node.args[0]  # This should be the constant weight tensor
        weight_scale = extract_scalar_value(weight_dq_node.args[1])
        weight_zero_point = int(extract_scalar_value(weight_dq_node.args[2]))

        # Check if weight_tensor has actual tensor data
        if hasattr(weight_tensor, 'tensor') and weight_tensor.tensor is not None:
            weight_data = weight_tensor.tensor.detach().numpy()

            # AOT Precomputation: Transpose weights for CMSIS-NN
            # CMSIS expects [in_features, out_features] but PyTorch stores [out_features, in_features]
            transposed_weights = weight_data.T.copy()

            # AOT Precomputation: Calculate kernel sums for CMSIS optimization
            kernel_sums = weight_data.sum(axis=1, dtype=np.int32)  # Sum along input dimension

            # AOT Precomputation: Calculate effective scale and quantization parameters
            effective_scale = (input_scale_val * weight_scale) / output_scale_val
            weight_mult, weight_shift = quantize_multiplier_aot(effective_scale)

            logger.info(f"AOT Linear Precomputation:")
            logger.info(f"   Weight shape: {weight_data.shape} -> {transposed_weights.shape}")
            logger.info(f"   Kernel sums shape: {kernel_sums.shape}")
            logger.info(f"   Effective scale: {effective_scale}")
            logger.info(f"   Weight mult/shift: {weight_mult}/{weight_shift}")

            # Calculate scratch buffer size needed
            in_features, out_features = transposed_weights.shape
            scratch_size = self._calculate_scratch_buffer_size_aot(in_features, out_features)

            return {
                'transposed_weights': torch.from_numpy(transposed_weights.astype(np.int8)),
                'kernel_sums': torch.from_numpy(kernel_sums),
                'weight_multiplier': weight_mult,
                'weight_shift': weight_shift,
                'weight_zero_point': weight_zero_point,
                'scratch_size': scratch_size,
                'in_features': in_features,
                'out_features': out_features,
            }

        logger.warning("Could not access weight tensor data for AOT precomputation")
        return None

    def _calculate_scratch_buffer_size_aot(self, in_features: int, out_features: int) -> int:
        """AOT calculation of scratch buffer size for arm_fully_connected_s8."""
        # Based on CMSIS-NN arm_fully_connected_s8 requirements
        accumulator_size = out_features * 4  # int32 accumulators
        input_buffer_size = ((in_features + 3) // 4) * 4  # SIMD-aligned
        temp_calc_size = 32  # Requantization workspace
        alignment_padding = 16  # ARM SIMD alignment

        total_size = accumulator_size + input_buffer_size + temp_calc_size + alignment_padding
        aligned_size = ((total_size + 15) // 16) * 16  # 16-byte aligned

        logger.info(f"AOT Scratch buffer size calculation: {aligned_size} bytes")
        return aligned_size

    def _store_precomputed_tensors_as_attributes(self, graph_module, precomputed_data, node_id):
        """Store precomputed tensors as module attributes and return get_attr nodes."""
        weights_attr = f"_precomputed_weights_{node_id}"
        kernel_sums_attr = f"_precomputed_kernel_sums_{node_id}"
        scratch_attr = f"_precomputed_scratch_{node_id}"

        # Store as module attributes
        setattr(graph_module, weights_attr, precomputed_data['transposed_weights'])
        setattr(graph_module, kernel_sums_attr, precomputed_data['kernel_sums'])

        # Create scratch buffer tensor (zeros)
        scratch_buffer = torch.zeros(precomputed_data['scratch_size'], dtype=torch.int8)
        setattr(graph_module, scratch_attr, scratch_buffer)

        # Create get_attr nodes
        with graph_module.graph.inserting_before(graph_module.graph.nodes.__iter__().__next__()):
            weights_node = graph_module.graph.get_attr(weights_attr)
            kernel_sums_node = graph_module.graph.get_attr(kernel_sums_attr)
            scratch_node = graph_module.graph.get_attr(scratch_attr)

        return weights_node, kernel_sums_node, scratch_node

    def _fuse_quantized_linear_patterns(self, graph_module: torch.fx.GraphModule) -> int:
        """Fuse dequantize -> linear -> quantize patterns."""
        fusion_count = 0
        nodes_to_erase = []

        for node in list(graph_module.graph.nodes):
            if not self._is_quant_node(node):
                continue

            quantize_node = node
            if not quantize_node.args:
                continue

            linear_op_node = quantize_node.args[0]
            if not self._is_linear_op(linear_op_node):
                continue

            # Check for input dequantization
            if len(linear_op_node.args) < 2:
                continue

            input_dq_node = linear_op_node.args[0]
            if not self._is_dequant_node(input_dq_node):
                continue

            # Extract weight and bias dequantization
            weight_dq_node, bias_dq_node = self._extract_weight_bias_from_linear(linear_op_node)
            if not weight_dq_node:
                continue

            logger.info("✅ Found complete cortex_m Q/DQ + linear pattern!")

            try:
                # Extract quantization parameters
                int8_input, input_scale, input_zero_point = input_dq_node.args[:3]
                output_scale, output_zero_point = quantize_node.args[1:3]

                input_scale_val = extract_scalar_value(input_scale)
                output_scale_val = extract_scalar_value(output_scale)
                input_zp_val = int(extract_scalar_value(input_zero_point))
                output_zp_val = int(extract_scalar_value(output_zero_point))

                # AOT PRECOMPUTATION: Calculate all static parameters
                precomputed = self._precompute_linear_params_aot(
                    weight_dq_node, input_scale_val, output_scale_val
                )

                if not precomputed:
                    logger.warning("Failed to precompute linear parameters, skipping fusion")
                    continue

                # Input quantization parameters (for runtime)
                input_mult, input_shift = quantize_multiplier_aot(1.0)  # Input multiplier is 1

                # Handle bias if present
                bias_tensor = None
                bias_mult, bias_shift = 0, 0
                if bias_dq_node:
                    bias_tensor = bias_dq_node.args[0]
                    bias_scale = extract_scalar_value(bias_dq_node.args[1])
                    # Bias effective scale = input_scale * weight_scale
                    bias_effective_scale = input_scale_val * extract_scalar_value(weight_dq_node.args[1])
                    bias_mult, bias_shift = quantize_multiplier_aot(bias_effective_scale / output_scale_val)

                # Store precomputed tensors as module attributes
                node_id = str(id(linear_op_node))
                weights_node, kernel_sums_node, scratch_node = self._store_precomputed_tensors_as_attributes(
                    graph_module, precomputed, node_id
                )

                logger.info("AOT computed linear parameters:")
                logger.info(f"   Input: mult={input_mult}, shift={input_shift}")
                logger.info(f"   Weight: mult={precomputed['weight_multiplier']}, shift={precomputed['weight_shift']}")
                logger.info(f"   Bias: mult={bias_mult}, shift={bias_shift}")

                with graph_module.graph.inserting_after(quantize_node):
                    fused = graph_module.graph.create_node(
                        "call_function",
                        target=torch.ops.cortex_m.quantized_linear.default,
                        args=(
                            int8_input,                           # input tensor
                            input_zp_val,                         # input zero point
                            input_mult,                           # input multiplier
                            input_shift,                          # input shift
                            weights_node,                         # precomputed transposed weights
                            kernel_sums_node,                     # precomputed kernel sums
                            precomputed['weight_zero_point'],     # weight zero point
                            precomputed['weight_multiplier'],     # weight multiplier
                            precomputed['weight_shift'],          # weight shift
                            bias_tensor,                          # bias tensor (optional)
                            bias_mult,                            # bias multiplier
                            bias_shift,                           # bias shift
                            scratch_node,                         # scratch buffer tensor
                            output_zp_val,                        # output zero point
                            precomputed['in_features'],           # input features
                            precomputed['out_features'],          # output features
                        ),
                        kwargs={},
                    )

                    self._transfer_metadata(fused, quantize_node)

                logger.info(f"✅ Created fused quantized_linear node: {fused}")

                # Replace all uses and mark for deletion
                quantize_node.replace_all_uses_with(fused)
                nodes_to_erase.extend([quantize_node, linear_op_node, input_dq_node, weight_dq_node])
                if bias_dq_node:
                    nodes_to_erase.append(bias_dq_node)

                fusion_count += 1

            except Exception as e:
                logger.error(f"❌ Error during linear AOT computation: {e}")
                logger.error("   Skipping fusion for this pattern")
                continue

        # Clean up old nodes
        for old_node in reversed(nodes_to_erase):
            if old_node in graph_module.graph.nodes and len(old_node.users) == 0:
                logger.info(f"🗑️ Erasing node: {old_node}")
                graph_module.graph.erase_node(old_node)

        return fusion_count

    def _fuse_quantized_binary_patterns(self, graph_module: torch.fx.GraphModule) -> int:
        """ Fuse dequantize -> binary_op -> quantize patterns."""
        # This handles the quantized_add operations that work
        fusion_count = 0
        nodes_to_erase = []

        for node in list(graph_module.graph.nodes):
            if not self._is_quant_node(node):
                continue

            quantize_node = node
            if not quantize_node.args:
                continue

            binary_op_node = quantize_node.args[0]
            if not self._is_binary_op(binary_op_node):
                continue

            if len(binary_op_node.args) < 2:
                continue

            dequant_node1, dequant_node2 = binary_op_node.args[:2]
            if not (self._is_dequant_node(dequant_node1) and self._is_dequant_node(dequant_node2)):
                continue

            # Extract op name
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
                input1_mult, input1_shift = quantize_multiplier_aot(scale1_val / output_scale_val)
                input2_mult, input2_shift = quantize_multiplier_aot(scale2_val / output_scale_val)
                output_mult, output_shift = quantize_multiplier_aot(1.0)  # Output multiplier is 1

                with graph_module.graph.inserting_after(quantize_node):
                    fused = graph_module.graph.create_node(
                        "call_function",
                        target=exir_ops.edge.cortex_m.quantized_add.default,
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

                    self._transfer_metadata(fused, quantize_node)

                # Replace all uses
                quantize_node.replace_all_uses_with(fused)
                binary_op_node.replace_all_uses_with(fused)
                dequant_node1.replace_all_uses_with(fused)
                dequant_node2.replace_all_uses_with(fused)

                nodes_to_erase.extend([quantize_node, binary_op_node, dequant_node1, dequant_node2])
                fusion_count += 1

            except Exception as e:
                logger.info(f"❌ Error during AoT computation: {e}")
                continue

        for old_node in reversed(nodes_to_erase):
            if old_node in graph_module.graph.nodes and len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)

        return fusion_count

    def _normalize_to_cortex_m_targets(self, graph_module: torch.fx.GraphModule) -> int:
        """Convert decomposed targets to cortex_m equivalents."""
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

    def _transfer_metadata(self, new_node, source_node, pass_name="QuantizedOpFusionPass"):
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

    def call(self, graph_module: torch.fx.GraphModule):
        logger.info("QuantizedOpFusionPass.call() started")

        # Normalize targets
        normalization_count = self._normalize_to_cortex_m_targets(graph_module)

        # Fuse binary operations
        binary_fusion_count = self._fuse_quantized_binary_patterns(graph_module)

        # Fuse linear operations
        linear_fusion_count = self._fuse_quantized_linear_patterns(graph_module)

        total_changes = normalization_count + binary_fusion_count + linear_fusion_count
        logger.info(f"Total changes: normalization={normalization_count}, "
                   f"binary_fusion={binary_fusion_count}, linear_fusion={linear_fusion_count}")

        if total_changes > 0:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, total_changes > 0)
