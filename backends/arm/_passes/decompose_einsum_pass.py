# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.get_decomposition_pass import GetDecompositionPass


class DecomposeEinsumPass(GetDecompositionPass):
    """Decomposes aten.einsum.default into more primitive ops.

    This pass is intended to be called in transform_for_annotation to prepare
    the graph for quantization. Einsum is not annotated directly by the Arm
    quantizer, but the decomposed ops are.

    """

    targeted_ops = [torch.ops.aten.einsum.default]

    def _get_input_tensors(self, node: torch.fx.Node) -> list:
        """Override the base hook because aten.einsum.default takes (equation,
        [operands]), which cannot be handled by the generic one-arg-per-input
        logic.
        """
        equation, operands = node.args  # type: ignore[union-attr]
        fake_operands = [operand.meta["val"] for operand in operands]  # type: ignore[union-attr]
        return [equation, fake_operands]

    def _get_placeholder_map(
        self,
        node: torch.fx.Node,
        decomposed_module: torch.fx.GraphModule,
    ) -> dict[str, torch.fx.Node]:
        """Override the base hook because einsum does not trace placeholders
        one-to-one with node.args.

        The traced graph includes arg0_1 for the equation string and arg1_i for
        each tensor inside the operand list, so we must skip the equation
        placeholder, which is not an original FX tensor node, and map each
        operand placeholder back to the corresponding original FX node.

        """
        _, operands = node.args
        name_to_input_tensor_map = {}

        for decomposed_node in decomposed_module.graph.nodes:
            if decomposed_node.op != "placeholder":
                continue
            if decomposed_node.name == "arg0_1":
                continue
            if not decomposed_node.name.startswith("arg1_"):
                raise RuntimeError(
                    f"Unexpected einsum placeholder name {decomposed_node.name!r}."
                )

            operand_idx = int(decomposed_node.name.split("_")[1]) - 1
            name_to_input_tensor_map[decomposed_node.name] = operands[operand_idx]  # type: ignore[index]

        return name_to_input_tensor_map  # type: ignore[return-value]

    def _get_output_node(self, output_node: torch.fx.Node) -> torch.fx.Node:
        """Return the traced value node for einsum graphs that emit
        output([node]).
        """
        return output_node.args[0][0]  # type: ignore[index, return-value]
