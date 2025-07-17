# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.cortex_m.passes.passes_utils import (
    cleanup_erased_nodes
    extract_scalar_value,
    is_qualified_int8_node,
    quantize_multiplier_aot,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_manager import PassResult


class QuantizedAddFusionPass(ExportPass):
    """
    ExportPass that:
    1. Replaces certain ops with cortex_m variants based on qualifiers.
    2. Fuses the pattern dequantize_per_tensor -> add -> quantize_per_tensor
       into cortex_m.quantized_add.default with AoT computed multipliers/shifts.
    """

    def __init__(self):
        super().__init__()

    def _replace_operators_with_cortex_m(
        self, graph_module: torch.fx.GraphModule
    ) -> int:
        replacement_count = 0
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function":
                print(f"🔍 Checking node: {node.target}")
                if (
                    node.target
                    == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                ):
                    print(
                        f"🔍 Found quantized_decomposed.quantize node with args: {node.args}"
                    )
                    if is_qualified_int8_node(node.args):
                        print("✅ Replacing quantize node")
                        node.target = exir_ops.edge.cortex_m.quantize_per_tensor.default
                        replacement_count += 1
                elif (
                    node.target
                    == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                ):
                    print("🔍 Found quantized_decomposed.dequantize node")
                    if is_qualified_int8_node(node.args):
                        print("✅ Replacing dequantize node")
                        node.target = (
                            exir_ops.edge.cortex_m.dequantize_per_tensor.default
                        )
                        replacement_count += 1
        print(f"🔧 Operator replacement: Replaced {replacement_count} nodes")
        return replacement_count

    def _fuse_quantized_add_patterns(self, graph_module: torch.fx.GraphModule) -> int:
        fusion_count = 0
        nodes_to_erase = []
        for node in list(graph_module.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.cortex_m.quantize_per_tensor.default
            ):
                continue
            quantize_node = node
            if not quantize_node.args:
                continue
            add_node = quantize_node.args[0]
            if not (
                hasattr(add_node, "op")
                and add_node.op == "call_function"
                and add_node.target == exir_ops.edge.aten.add.Tensor
                and len(add_node.args) >= 2
            ):
                continue
            dequant_node1, dequant_node2 = add_node.args[:2]
            if not (
                hasattr(dequant_node1, "op")
                and dequant_node1.op == "call_function"
                and dequant_node1.target
                == exir_ops.edge.cortex_m.dequantize_per_tensor.default
                and hasattr(dequant_node2, "op")
                and dequant_node2.op == "call_function"
                and dequant_node2.target
                == exir_ops.edge.cortex_m.dequantize_per_tensor.default
            ):
                continue
            print("✅ Found complete cortex_m Q/DQ + add pattern!")
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
                print("🧮 AoT computed parameters:")
                print(f"   Input1: mult={input1_mult}, shift={input1_shift}")
                print(f"   Input2: mult={input2_mult}, shift={input2_shift}")
                print(f"   Output: mult={output_mult}, shift={output_shift}")
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
                    # Copy metadata from the quantize_node to the new fused node
                    if hasattr(quantize_node, "meta") and quantize_node.meta:
                        fused.meta = quantize_node.meta.copy()
                        if "val" in quantize_node.meta:
                            fused.meta["val"] = quantize_node.meta["val"]
                print(f"✅ Created fused quantized_add node: {fused}")
                quantize_node.replace_all_uses_with(fused)
                add_node.replace_all_uses_with(fused)
                dequant_node1.replace_all_uses_with(fused)
                dequant_node2.replace_all_uses_with(fused)
                nodes_to_erase.extend(
                    [quantize_node, add_node, dequant_node1, dequant_node2]
                )
                fusion_count += 1
                print(f"🎯 Pattern fused, total so far: {fusion_count}")
            except Exception as e:
                print(f"❌ Error during AoT computation: {e}")
                print("   Skipping fusion for this pattern")
                continue
        # Clean up erased nodes
        for old_node in reversed(nodes_to_erase):
            if old_node in graph_module.graph.nodes and len(old_node.users) == 0:
                print(f"🗑️ Erasing node: {old_node}")
                graph_module.graph.erase_node(old_node)
        return fusion_count

    def call(self, graph_module: torch.fx.GraphModule):
        print("🔧 QuantizedAddFusionPass.call() started")
        replacement_count = self._replace_operators_with_cortex_m(graph_module)
        fusion_count = self._fuse_quantized_add_patterns(graph_module)
        cleanup_erased_nodes(graph_module)
        total_changes = replacement_count + fusion_count
        print(
            f"🎯 Total changes: {replacement_count} replacements + {fusion_count} fusions = {total_changes}"
        )
        if total_changes > 0:
            graph_module.graph.lint()
            graph_module.recompile()
        print("=== AFTER FUSION: All nodes in the graph ===")
        for i, node in enumerate(graph_module.graph.nodes):
            print(f"Node {i}: op={node.op}, target={node.target}")
            if "quantized_add" in str(node.target):
                print(" ⭐ FOUND QUANTIZED_ADD NODE! ⭐")
        print("=== END DEBUG ===")
        return PassResult(graph_module, total_changes > 0)
