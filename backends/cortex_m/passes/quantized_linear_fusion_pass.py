# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import executorch.backends.cortex_m.ops.operators  # noqa
import torch
import torch.fx

from executorch.backends.cortex_m.passes.passes_utils import (
    cleanup_nodes,
    is_dequant_node,
    quantize_multiplier_aot,
    transfer_metadata,
)

from executorch.backends.transforms.utils import create_mutable_buffer, get_param_tensor

from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger("quantized_linear_fusion_pass")
logger.setLevel(logging.INFO)


class QuantizedLinearFusionPass(XNNPACKPass):
    """
    Cortex-M backend pass that fuses quantized linear-like patterns.
    Fuses: dequantize -> [linear/addmm/fc_ops] -> quantize
    Into: cortex_m.quantized_linear.default with direct parameters.
    """

    SUPPORTED_OPS_MAPPING = {
        exir_ops.edge.aten.addmm.default: exir_ops.edge.cortex_m.quantized_linear.default,
        exir_ops.edge.aten.mm.default: exir_ops.edge.cortex_m.quantized_linear.default,
    }

    requires_exported_program = True

    def __init__(self, exported_program: ExportedProgram):
        super().__init__(exported_program)
        self.nodes_to_erase = []

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        logger.info("Starting QuantizedLinearFusionPass")
        assert id(self._exported_program.graph_module.graph) == id(
            graph_module.graph
        ), "QuantizedLinearFusionPass requires same graph instance"

        try:
            fusion_count = self._fuse_quantized_linear_patterns(graph_module)
            if fusion_count > 0:
                graph_module.graph.eliminate_dead_code()
                graph_module.graph.lint()
                graph_module.recompile()
            logger.info(f"Linear fusion completed: {fusion_count} patterns fused")
            return PassResult(graph_module, fusion_count > 0)
        except Exception as e:
            logger.error(f"Error in QuantizedLinearFusionPass: {e}")
            raise e

    def _extract_linear_pattern(self, quantize_node: Node):
        if not quantize_node.args:
            return None
        fc_node = quantize_node.args[0]
        if not (
            fc_node.op == "call_function"
            and fc_node.target in self.SUPPORTED_OPS_MAPPING
        ):
            return None

        op_name = str(fc_node.target).split(".")[-1]

        if "addmm" in str(fc_node.target):
            input_dq_node = fc_node.args[1]
        else:
            input_dq_node = fc_node.args[0]
        if not is_dequant_node(input_dq_node):
            logger.info("input_dq_node is not a dequant node")
            return None
        weight_dq_node, bias_dq_node = self._extract_weight_bias_from_fc_op(fc_node)
        if not weight_dq_node:
            logger.info("No weight, bias dequantize node found")
            return None
        return (
            quantize_node,
            fc_node,
            input_dq_node,
            weight_dq_node,
            bias_dq_node,
            op_name,
        )

    def _extract_weight_bias_from_fc_op(self, fc_node: Node):
        """Generic extraction for FC-like operations."""

        if "addmm" in str(fc_node.target):
            if len(fc_node.args) >= 3:
                bias_arg = fc_node.args[0]
                weight_arg = fc_node.args[2]
                weight_dq_node = self._trace_to_dequantize(weight_arg)
                logger.info(
                    f"weight_arg: {weight_arg}, traced weight_dq_node: {weight_dq_node}"
                )

                if weight_dq_node is None:
                    logger.info("No weight dequantize node found ")

                # For bias, try to trace to dequantize but allow None (no-bias case)
                bias_dq_node = self._trace_to_dequantize(bias_arg)
                if bias_dq_node is None:
                    logger.info("No bias dequantize node found - likely no-bias linear")
                return weight_dq_node, bias_dq_node
        elif any(op in str(fc_node.target) for op in ["linear", "mm"]):
            if len(fc_node.args) >= 2:
                weight_arg = fc_node.args[1]
                bias_arg = fc_node.args[2] if len(fc_node.args) > 2 else None
                weight_dq_node = self._trace_to_dequantize(weight_arg)
                bias_dq_node = self._trace_to_dequantize(bias_arg) if bias_arg else None
                return weight_dq_node, bias_dq_node
        return None, None

    def _extract_input_quantization_parameters(
        self, input_dq_node: Node
    ) -> Optional[dict]:
        """Extract input quantization parameters from dequantize node."""
        try:
            # Find the quantize operation that produces the int8 tensor
            input_quantize_node = None
            if hasattr(input_dq_node, "args") and input_dq_node.args:
                quantize_candidate = input_dq_node.args[0]
                if getattr(
                    quantize_candidate, "op", None
                ) == "call_function" and "quantize" in str(
                    getattr(quantize_candidate, "target", "")
                ):
                    input_quantize_node = quantize_candidate

            if not input_quantize_node:
                logger.error("Could not find quantize node for input!")
                return None

            # Extract input quantization parameters
            input_scale = self._extract_param_value(input_dq_node.args[1])
            input_zero_point = int(self._extract_param_value(input_dq_node.args[2]))
            input_multiplier, input_shift = quantize_multiplier_aot(input_scale)

            return {
                "input_scale": input_scale,
                "input_zero_point": input_zero_point,
                "input_multiplier": input_multiplier,
                "input_shift": input_shift,
                "input_tensor": input_quantize_node,
            }
        except Exception as e:
            logger.error(f"Failed to extract input quantization parameters: {e}")
            return None

    def _extract_output_quantization_parameters(
        self, quantize_node: Node
    ) -> Optional[dict]:
        """Extract output quantization parameters from quantize node."""
        try:
            output_scale = self._extract_param_value(quantize_node.args[1])
            output_zero_point = int(self._extract_param_value(quantize_node.args[2]))

            return {
                "output_scale": output_scale,
                "output_zero_point": output_zero_point,
            }
        except Exception as e:
            logger.error(f"Failed to extract output quantization parameters: {e}")
            return None

    def _create_constant_parameter_buffer(
        self, graph, quantize_node: Node, data: torch.Tensor, name: str
    ):
        """Create a parameter buffer"""
        buffer_name = f"{name}_{id(quantize_node)}"

        setattr(graph.owning_module, buffer_name, data)

        # Create a get_attr node
        with graph.inserting_before(quantize_node):
            buffer_node = graph.create_node(
                op="get_attr", target=buffer_name, name=buffer_name
            )

            # Set metadata
            buffer_node.meta["val"] = data

        return buffer_node

    def _extract_weight_parameters(self, weight_dq_node: Node) -> Optional[dict]:
        try:
            weight_tensor = weight_dq_node.args[0]
            weight_scale = weight_dq_node.args[1]
            weight_zero_point = (
                weight_dq_node.args[2] if len(weight_dq_node.args) > 2 else None
            )

            weight_scale_data = self._extract_param_value(weight_scale)
            weight_zp_data = (
                self._extract_param_value(weight_zero_point)
                if weight_zero_point
                else None
            )

            # Get actual tensor data to determine output features
            weight_tensor_data = get_param_tensor(self._exported_program, weight_tensor)
            out_features = weight_tensor_data.shape[0]

            # Handle both per-tensor and per-channel
            if (
                isinstance(weight_scale_data, torch.Tensor)
                and weight_scale_data.numel() > 1
            ):
                # Per-channel: ensure we have the right number of elements
                assert (
                    weight_scale_data.numel() == out_features
                ), f"Scale size {weight_scale_data.numel()} != out_features {out_features}"

                multipliers = []
                shifts = []
                for scale in weight_scale_data:
                    mult, shift = quantize_multiplier_aot(scale.item())
                    multipliers.append(mult)
                    shifts.append(shift)

                weight_multiplier = torch.tensor(multipliers, dtype=torch.int32)
                weight_shift = torch.tensor(shifts, dtype=torch.int32)
                weight_zp_tensor = (
                    weight_zp_data.int()
                    if weight_zp_data is not None
                    else torch.zeros(out_features, dtype=torch.int32)
                )
            else:
                # Per-tensor: create tensors with correct size for output features
                scale_val = (
                    weight_scale_data.item()
                    if isinstance(weight_scale_data, torch.Tensor)
                    else weight_scale_data
                )
                mult, shift = quantize_multiplier_aot(scale_val)

                # Create tensors sized for out_features (not single element)
                weight_multiplier = torch.full((out_features,), mult, dtype=torch.int32)
                weight_shift = torch.full((out_features,), shift, dtype=torch.int32)
                weight_zp_tensor = torch.full(
                    (out_features,),
                    weight_zp_data if weight_zp_data else 0,
                    dtype=torch.int32,
                )

            # Validate multipliers
            for i, mult in enumerate(weight_multiplier):
                if mult < (1 << 30) or mult > ((1 << 31) - 1):
                    logger.error(
                        f"Invalid multiplier[{i}]: {mult}, scale was: {weight_scale_data}"
                    )
                    return None

            return {
                "weight_tensor": weight_tensor,
                "weight_zero_point_data": weight_zp_tensor,
                "weight_multiplier_data": weight_multiplier,
                "weight_shift_data": weight_shift,
            }
        except Exception as e:
            logger.error(f"Failed to extract weight parameters: {e}")
            return None

    def _extract_bias_parameters(self, bias_dq_node: Optional[Node]) -> Optional[dict]:
        """
        Extract bias parameters for quantized linear fusion.
        Handles both dequantized bias nodes and constant bias tensors.
        Returns a dict with bias_tensor, bias_multiplier, and bias_shift.
        """
        if not bias_dq_node:
            # No bias present
            return None
        try:
            # Case 1: Bias is a dequantize node
            if hasattr(bias_dq_node, "op") and is_dequant_node(bias_dq_node):
                bias_tensor = bias_dq_node.args[0]
                bias_scale = bias_dq_node.args[1]

                bias_scale_data = self._extract_param_value(bias_scale)

                if (
                    isinstance(bias_scale_data, torch.Tensor)
                    and bias_scale_data.numel() > 1
                ):
                    # Per-channel bias
                    bias_multipliers = []
                    bias_shifts = []
                    for scale_val in bias_scale_data.tolist():
                        mult, shift = quantize_multiplier_aot(scale_val)
                        bias_multipliers.append(mult)
                        bias_shifts.append(shift)
                    return {
                        "bias_tensor": bias_tensor,
                        "bias_multiplier": bias_multipliers,
                        "bias_shift": bias_shifts,
                    }
                else:
                    # Per-tensor bias
                    bias_scale_val = (
                        bias_scale_data.item()
                        if isinstance(bias_scale_data, torch.Tensor)
                        else bias_scale_data
                    )
                    bias_multiplier, bias_shift = quantize_multiplier_aot(
                        bias_scale_val
                    )
                    return {
                        "bias_tensor": bias_tensor,
                        "bias_multiplier": bias_multiplier,
                        "bias_shift": bias_shift,
                    }
            else:
                # Case 2: Bias is a constant tensor (not dequantized)
                # This can happen if bias is not quantized in the model
                bias_tensor = bias_dq_node
                # Use default multiplier/shift for unquantized bias
                bias_multiplier = 1
                bias_shift = 0
                return {
                    "bias_tensor": bias_tensor,
                    "bias_multiplier": bias_multiplier,
                    "bias_shift": bias_shift,
                }
        except Exception as e:
            logger.error(f"Failed to extract bias parameters: {e}")
            return None

    def _prepare_bias_tensors(
        self, bias_params: Optional[dict], out_features: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare bias multiplier and shift tensors for kernel call.
        Returns (bias_multiplier_tensor, bias_shift_tensor) both sized [out_features].
        """
        if bias_params:
            bias_multiplier = bias_params["bias_multiplier"]
            bias_shift = bias_params["bias_shift"]

            # Convert to tensors of the right size
            if isinstance(bias_multiplier, int):
                bias_multiplier_tensor = torch.full(
                    [out_features], bias_multiplier, dtype=torch.int32
                )
            elif isinstance(bias_multiplier, list):
                assert (
                    len(bias_multiplier) == out_features
                ), f"Bias multiplier size {len(bias_multiplier)} != out_features {out_features}"
                bias_multiplier_tensor = torch.tensor(
                    bias_multiplier, dtype=torch.int32
                )
            elif isinstance(bias_multiplier, torch.Tensor):
                assert (
                    bias_multiplier.numel() == out_features
                ), f"Bias multiplier size {bias_multiplier.numel()} != out_features {out_features}"
                bias_multiplier_tensor = bias_multiplier
            else:
                raise TypeError(
                    f"Unsupported bias_multiplier type: {type(bias_multiplier)}"
                )

            if isinstance(bias_shift, int):
                bias_shift_tensor = torch.full(
                    [out_features], bias_shift, dtype=torch.int32
                )
            elif isinstance(bias_shift, list):
                assert (
                    len(bias_shift) == out_features
                ), f"Bias shift size {len(bias_shift)} != out_features {out_features}"
                bias_shift_tensor = torch.tensor(bias_shift, dtype=torch.int32)
            elif isinstance(bias_shift, torch.Tensor):
                assert (
                    bias_shift.numel() == out_features
                ), f"Bias shift size {bias_shift.numel()} != out_features {out_features}"
                bias_shift_tensor = bias_shift
            else:
                raise TypeError(f"Unsupported bias_shift type: {type(bias_shift)}")

            return bias_multiplier_tensor, bias_shift_tensor
        else:
            # No bias: return zero tensors of correct shape
            return (
                torch.zeros([out_features], dtype=torch.int32),
                torch.zeros([out_features], dtype=torch.int32),
            )

    def _extract_param_value(self, node_or_value):
        """
        Extract a scalar value from a Node or a direct float/int.
        """
        if isinstance(node_or_value, (float, int)):
            return node_or_value
        # If it's a tensor, get its scalar value if possible
        if isinstance(node_or_value, torch.Tensor):
            return node_or_value.item() if node_or_value.numel() == 1 else node_or_value
        # If it's a Node, use get_param_tensor
        if hasattr(node_or_value, "op"):
            tensor = get_param_tensor(self._exported_program, node_or_value)
            return tensor.item() if tensor.numel() == 1 else tensor
        raise TypeError(f"Unsupported parameter type: {type(node_or_value)}")

    def _calculate_cmsis_scratch_size(self, weight_tensor) -> int:
        """Calculate CMSIS-NN scratch buffer size for quantized linear operations.

        Source: CMSIS-NN arm_fully_connected_s8_get_buffer_size() returns filter_dims->w * sizeof(int32_t).
        This buffer stores pre-computed kernel sums (weight row sums) - one int32_t per output feature.
        Same buffer size applies to both per-tensor and per-channel quantization paths since both use
        identical kernel sum optimization in the underlying matrix multiplication.
        """
        try:
            print(f"weight_tensor type: {type(weight_tensor)}, value: {weight_tensor}")
            weight_shape = get_param_tensor(self._exported_program, weight_tensor).shape
            out_features = weight_shape[0]  # filter_dims->w in CMSIS terms

            # CMSIS-NN implementation expects the following size
            cmsis_buffer_size = out_features * 4  # sizeof(int32_t)
            return cmsis_buffer_size
        except Exception as e:
            logger.error(f"Failed to calculate CMSIS scratch size: {e}")
            return 2048  # Fallback

    def _create_scratch_buffer(self, graph, quantize_node: Node, weight_tensor):
        cmsis_scratch = self._calculate_cmsis_scratch_size(weight_tensor)

        kernel_sum_header = 8  # sizeof(KernelSumHeader)
        total_size = kernel_sum_header + cmsis_scratch

        logger.info(
            f"Kernel sum header: {kernel_sum_header}, CMSIS buffer: {cmsis_scratch}, total: {total_size}"
        )

        return create_mutable_buffer(
            self._exported_program,
            name=f"b_cmsis_linear_scratch_{id(quantize_node)}",
            data=torch.zeros((total_size,), dtype=torch.int8),
        )

    def _create_fused_node(
        self,
        graph,
        quantize_node: Node,
        quant_params: dict,
        weight_params: dict,
        bias_params: Optional[dict],
        quantized_target,
    ) -> Node:
        """Generic fused node creation for any FC-like operation."""
        # Extract all parameters
        input_tensor = quant_params["input_tensor"]
        input_zp = quant_params["input_zero_point"]
        input_multiplier = quant_params["input_multiplier"]
        input_shift = quant_params["input_shift"]
        weight_tensor = weight_params["weight_tensor"]

        weight_zp_node = self._create_constant_parameter_buffer(
            graph, quantize_node, weight_params["weight_zero_point_data"], "weight_zp"
        )
        weight_mult_node = self._create_constant_parameter_buffer(
            graph, quantize_node, weight_params["weight_multiplier_data"], "weight_mult"
        )
        weight_shift_node = self._create_constant_parameter_buffer(
            graph, quantize_node, weight_params["weight_shift_data"], "weight_shift"
        )
        # Get dimensions
        weight_shape = get_param_tensor(self._exported_program, weight_tensor).shape
        assert (
            len(weight_shape) == 2
        ), f"Weight tensor must be 2D, got shape {weight_shape}"
        in_features = weight_shape[1]
        out_features = weight_shape[0]

        # Handle bias
        bias_tensor = bias_params["bias_tensor"] if bias_params else None
        bias_multiplier, bias_shift = self._prepare_bias_tensors(
            bias_params, out_features
        )
        output_zp = quant_params["output_zero_point"]

        scratch_buffer = self._create_scratch_buffer(
            graph, quantize_node, weight_tensor
        )

        with graph.inserting_after(quantize_node):
            fused = graph.create_node(
                "call_function",
                target=quantized_target,
                args=(
                    input_tensor,
                    input_zp,
                    input_multiplier,
                    input_shift,
                    weight_tensor,
                    weight_zp_node,
                    weight_mult_node,
                    weight_shift_node,
                    bias_tensor,
                    bias_multiplier,
                    bias_shift,
                    scratch_buffer,
                    output_zp,
                    in_features,
                    out_features,
                ),
                kwargs={},
            )

            transfer_metadata(fused, quantize_node, "QuantizedLinearFusionPass")
        return fused

    def _mark_for_cleanup(self, nodes):
        for node in nodes:
            if node is not None:
                self.nodes_to_erase.append(node)

    def _cleanup_nodes(self, graph):
        cleanup_nodes(self.nodes_to_erase, graph)
        self.nodes_to_erase.clear()

    def _extract_linear_pattern_with_validation(self, quantize_node: Node):
        pattern_info = self._extract_linear_pattern(quantize_node)
        if not pattern_info:
            return None
        # Optionally add more validation here if needed
        return pattern_info

    def _trace_to_dequantize(self, node: Optional[Node], max_depth=3) -> Optional[Node]:
        """Trace through transformations to find dequantize node."""
        current_node = node
        depth = 0
        while current_node and depth < max_depth:
            if is_dequant_node(current_node):
                return current_node
            if current_node.op == "call_function" and current_node.target in {
                exir_ops.edge.aten.permute_copy.default,
                exir_ops.edge.aten.view_copy.default,
            }:
                if current_node.args:
                    current_node = current_node.args[0]
                    depth += 1
                    continue
            break
        return None

    def _fuse_quantized_linear_patterns(
        self, graph_module: torch.fx.GraphModule
    ) -> int:
        fusion_count = 0
        graph = graph_module.graph
        for node in list(graph.nodes):
            if not (
                node.op == "call_function" and "quantize_per_tensor" in str(node.target)
            ):
                continue
            pattern_info = self._extract_linear_pattern_with_validation(node)
            if not pattern_info:
                continue

            (
                quantize_node,
                fc_node,
                input_dq_node,
                weight_dq_node,
                bias_dq_node,
                op_name,
            ) = pattern_info

            # Get quantized target for this FC operation
            quantized_target = self.SUPPORTED_OPS_MAPPING.get(fc_node.target)
            if not quantized_target:
                logger.warning(f"No quantized target found for {fc_node.target}")
                continue

            logger.info(f"✅ Found complete cortex_m Q/DQ + {op_name} pattern!")

            try:
                input_params = self._extract_input_quantization_parameters(
                    input_dq_node
                )
                if not input_params:
                    logger.error(
                        "Quantization parameter extraction failed for node: %s", node
                    )
                    return None
                output_params = self._extract_output_quantization_parameters(
                    quantize_node
                )
                if not output_params:
                    logger.error(
                        "Output quantization parameter extraction failed for node: %s",
                        node,
                    )
                    return None
                quant_params = {**input_params, **output_params}
                logger.info(f"Quantization parameters: {quant_params}")

                weight_params = self._extract_weight_parameters(weight_dq_node)
                if not weight_params:
                    continue
                bias_params = self._extract_bias_parameters(bias_dq_node)
                if bias_dq_node and not bias_params:
                    continue
                fused_node = self._create_fused_node(
                    graph,
                    quantize_node,
                    quant_params,
                    weight_params,
                    bias_params,
                    quantized_target,
                )
                logger.info(f"Created fused {op_name} node: {fused_node}")

                quantize_node.replace_all_uses_with(fused_node)
                self._mark_for_cleanup(
                    [
                        quantize_node,
                        fc_node,
                        input_dq_node,
                        weight_dq_node,
                        bias_dq_node,
                    ]
                )
                fusion_count += 1
                logger.info(f"✅ Successfully fused {op_name} operation {fusion_count}")
            except Exception as e:
                logger.error(
                    f"Failed to fuse {op_name} pattern for {fc_node.name}: {e}"
                )
                continue
        self._cleanup_nodes(graph)
        return fusion_count
