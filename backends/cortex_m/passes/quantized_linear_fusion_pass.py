# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
    extract_scalar_value,
    is_dequant_node,
    is_quant_node,
    quantize_multiplier_aot,
    trace_to_dequantize,
    transfer_metadata,
)

from executorch.backends.transforms.utils import create_mutable_buffer, get_param_tensor
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import Node
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger("quantized_linear_fusion_pass")
logger.setLevel(logging.INFO)


class QuantizedLinearFusionPass(ExportPass):
    """
    Cortex-M backend pass that fuses quantized linear-like patterns.
    Fuses: dequantize -> [linear/addmm/fc_ops] -> quantize
    Into: cortex_m.quantized_linear.default with direct parameters.
    """

    # Extensible operation mapping for FC-like operations
    SUPPORTED_OPS_MAPPING = {
        exir_ops.edge.aten.linear.default: exir_ops.edge.cortex_m.quantized_linear.default,
        exir_ops.edge.aten.addmm.default: exir_ops.edge.cortex_m.quantized_linear.default,
        # Future FC-like ops can be added here
    }

    requires_exported_program = True

    def __init__(self, exported_program: ExportedProgram):
        logger.info("C'tor QuantizedLinearFusionPass")
        super().__init__()
        self._exported_program = exported_program
        self.nodes_to_erase = []

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        logger.info("Starting QuantizedLinearFusionPass")
        try:
            fusion_count = self._fuse_quantized_linear_patterns(graph_module)
            for node in graph_module.graph.nodes:
                logger.debug(
                    "Post-fusion Node: %s, op: %s, target: %s",
                    node.name,
                    node.op,
                    getattr(node, "target", None),
                )
            if fusion_count > 0:
                graph_module.graph.eliminate_dead_code()
                graph_module.graph.lint()
                graph_module.recompile()
            logger.info(f"Linear fusion completed: {fusion_count} patterns fused")
            return PassResult(graph_module, fusion_count > 0)
        except Exception as e:
            logger.error(f"Error in QuantizedLinearFusionPass: {e}")
            raise

    def _is_supported_fc_op(self, node: Node) -> bool:
        """Check if node is a supported FC-like operation."""
        return node.op == "call_function" and node.target in self.SUPPORTED_OPS_MAPPING

    def _get_quantized_target(self, fc_op_target):
        """Get the corresponding quantized target for a FC operation."""
        return self.SUPPORTED_OPS_MAPPING.get(fc_op_target)

    def _extract_linear_pattern(self, quantize_node: Node):
        if not quantize_node.args:
            return None
        fc_node = quantize_node.args[0]
        if not self._is_supported_fc_op(fc_node):
            return None

        # Extract op name for logging
        op_name = str(fc_node.target).split(".")[-1]

        if "addmm" in str(fc_node.target):
            input_dq_node = fc_node.args[1]
        else:
            input_dq_node = fc_node.args[0]
        if not is_dequant_node(input_dq_node):
            return None
        weight_dq_node, bias_dq_node = self._extract_weight_bias_from_fc_op(fc_node)
        if not weight_dq_node:
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
                weight_dq_node = trace_to_dequantize(weight_arg)
                bias_dq_node = trace_to_dequantize(bias_arg)
                return weight_dq_node, bias_dq_node
        elif any(op in str(fc_node.target) for op in ["linear", "mm", "bmm", "addmv"]):
            if len(fc_node.args) >= 2:
                weight_arg = fc_node.args[1]
                bias_arg = fc_node.args[2] if len(fc_node.args) > 2 else None
                weight_dq_node = trace_to_dequantize(weight_arg)
                bias_dq_node = trace_to_dequantize(bias_arg) if bias_arg else None
                return weight_dq_node, bias_dq_node
        return None, None

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

    def _extract_quantization_parameters(
        self, input_dq_node: Node, quantize_node: Node
    ) -> Optional[dict]:
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

            if input_quantize_node and input_quantize_node.args:
                # Get the actual input node (float tensor)
                actual_input = input_quantize_node.args[0]
                logger.info(
                    f"Quantize operation input: {actual_input}, op: {getattr(actual_input, 'op', None)}"
                )

            # Extract quantization parameters
            input_scale = self._extract_param_value(input_dq_node.args[1])
            input_zero_point = int(self._extract_param_value(input_dq_node.args[2]))
            output_scale = self._extract_param_value(quantize_node.args[1])
            output_zero_point = int(self._extract_param_value(quantize_node.args[2]))

            input_multiplier, input_shift = quantize_multiplier_aot(input_scale)

            return {
                "input_scale": input_scale,
                "input_zero_point": input_zero_point,
                "input_multiplier": input_multiplier,
                "input_shift": input_shift,
                "output_scale": output_scale,
                "output_zero_point": output_zero_point,
                "input_tensor": input_quantize_node,
            }
        except Exception as e:
            logger.error(f"Failed to extract quantization parameters: {e}")
            return None

    def _create_parameter_buffer(
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

            # Get actual tensor data to check dimensions
            weight_scale_data = get_param_tensor(self._exported_program, weight_scale)
            weight_zp_data = (
                get_param_tensor(self._exported_program, weight_zero_point)
                if weight_zero_point
                else None
            )

            # Handle both per-tensor and per-channel
            if weight_scale_data.numel() == 1:
                # Per-tensor: create single-element tensors
                scale_val = weight_scale_data.item()
                mult, shift = quantize_multiplier_aot(scale_val)
                weight_multiplier = torch.tensor([mult], dtype=torch.int32)
                weight_shift = torch.tensor([shift], dtype=torch.int32)
                weight_zp_tensor = torch.tensor(
                    [weight_zp_data.item() if weight_zp_data else 0], dtype=torch.int32
                )
            else:
                # Per-channel: create tensors with multiple elements
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
                    else torch.zeros_like(weight_scale_data, dtype=torch.int32)
                )

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
        if not bias_dq_node:
            return None
        try:
            bias_tensor = bias_dq_node.args[0]
            bias_scale = bias_dq_node.args[1]  # Could be scalar or tensor

            # Get bias scale data (could be per-channel)
            bias_scale_data = get_param_tensor(self._exported_program, bias_scale)
            if bias_scale_data.numel() > 1:
                # Per-channel bias: quantize each channel separately
                bias_multipliers = []
                bias_shifts = []
                for scale_val in bias_scale_data.tolist():
                    mult, shift = quantize_multiplier_aot(scale_val)
                    bias_multipliers.append(mult)
                    bias_shifts.append(shift)
                return {
                    "bias_tensor": bias_tensor,
                    "bias_multiplier": bias_multipliers,  # List of int
                    "bias_shift": bias_shifts,  # List of int
                }
            else:
                # Per-tensor bias (original logic)
                bias_scale_val = bias_scale_data.item()
                bias_multiplier, bias_shift = quantize_multiplier_aot(bias_scale_val)
                return {
                    "bias_tensor": bias_tensor,
                    "bias_multiplier": bias_multiplier,
                    "bias_shift": bias_shift,
                }
        except Exception as e:
            logger.error(f"Failed to extract bias parameters: {e}")
            return None

    def _create_scratch_buffer(
        self, graph, quantize_node: Node, batch_size: int, out_features: int
    ):
        struct_size = 2060  # 2048 + sizeof(kernel_sum_state defined in kernel impl)
        cmsis_scratch = 1024  # CMSIS scratch buffer size
        total_size = struct_size + cmsis_scratch  # Should be ~3084

        print(f"Allocating scratch buffer size: {total_size}")
        # Creates mutable runtime buffer
        scratch_buffer = create_mutable_buffer(
            self._exported_program,
            name=f"b_scratch_{id(quantize_node)}",
            data=torch.zeros((total_size,), dtype=torch.int8),
        )
        return scratch_buffer

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
        try:
            if hasattr(input_tensor, "op") and input_tensor.op == "call_function":
                # If it's a quantize operation, try to get its output
                if hasattr(input_tensor, "meta") and "val" in input_tensor.meta:
                    tensor_val = input_tensor.meta["val"]
            else:
                # Try to get actual tensor data
                actual_tensor = get_param_tensor(self._exported_program, input_tensor)
        except Exception as e:
            logger.info(f"Failed to extract tensor: {e}")

        input_zp = quant_params["input_zero_point"]
        input_multiplier = quant_params["input_multiplier"]
        input_shift = quant_params["input_shift"]
        weight_tensor = weight_params["weight_tensor"]

        weight_zp_node = self._create_parameter_buffer(
            graph, quantize_node, weight_params["weight_zero_point_data"], "weight_zp"
        )
        weight_mult_node = self._create_parameter_buffer(
            graph, quantize_node, weight_params["weight_multiplier_data"], "weight_mult"
        )
        weight_shift_node = self._create_parameter_buffer(
            graph, quantize_node, weight_params["weight_shift_data"], "weight_shift"
        )
        # Get dimensions
        try:
            weight_shape = get_param_tensor(self._exported_program, weight_tensor).shape
            in_features = weight_shape[1]
            out_features = weight_shape[0]
        except Exception:
            in_features = 1
            out_features = 1

        # Handle bias
        bias_tensor = None
        # Always create Tensors for bias_multiplier and bias_shift
        if bias_params:
            bias_tensor = bias_params["bias_tensor"]
            bias_multiplier = bias_params["bias_multiplier"]
            bias_shift = bias_params["bias_shift"]
            # Convert to tensor if needed (per-tensor or per-channel)
            if isinstance(bias_multiplier, int):
                bias_multiplier = torch.tensor([bias_multiplier], dtype=torch.int32)
            elif isinstance(bias_multiplier, list):
                bias_multiplier = torch.tensor(bias_multiplier, dtype=torch.int32)
            # If already a tensor, leave as is
            if isinstance(bias_shift, int):
                bias_shift = torch.tensor([bias_shift], dtype=torch.int32)
            elif isinstance(bias_shift, list):
                bias_shift = torch.tensor(bias_shift, dtype=torch.int32)
            # If already a tensor, leave as is
        else:
            # No bias: pass zero tensors of correct shape
            bias_multiplier = torch.zeros([out_features], dtype=torch.int32)
            bias_shift = torch.zeros([out_features], dtype=torch.int32)

        output_zp = quant_params["output_zero_point"]

        scratch_buffer = self._create_scratch_buffer(
            graph, quantize_node, 1, out_features
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
            logger.debug(f"Found quantize_per_tensor node: {node}")
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
            quantized_target = self._get_quantized_target(fc_node.target)
            if not quantized_target:
                logger.warning(f"No quantized target found for {fc_node.target}")
                continue

            logger.info(f"✅ Found complete cortex_m Q/DQ + {op_name} pattern!")

            try:
                quant_params = self._extract_quantization_parameters(
                    input_dq_node, quantize_node
                )
                if not quant_params:
                    continue
                logger.info(f"Quantization parameters: {quant_params}")

                weight_params = self._extract_weight_parameters(weight_dq_node)
                if not weight_params:
                    continue
                bias_params = self._extract_bias_parameters(bias_dq_node)

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

    def _find_original_input_placeholder(self, dq_node: Node) -> Node:
        """Traverse back to find the original input placeholder"""
        current = dq_node
        while current and current.op != "placeholder":
            if current.op == "call_function" and "dequantize" in str(current.target):
                if current.args and len(current.args) > 0:
                    quantize_node = current.args[0]
                    if quantize_node.op == "call_function" and "quantize" in str(
                        quantize_node.target
                    ):
                        if quantize_node.args and len(quantize_node.args) > 0:
                            input_node = quantize_node.args[0]
                            if input_node.op == "placeholder":
                                return input_node
            current = current.args[0] if current.args else None
        return current
