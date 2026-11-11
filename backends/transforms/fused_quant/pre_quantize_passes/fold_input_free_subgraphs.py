# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.transforms.utils import add_constant
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch import fx
from torch._export.passes.constant_folding import ConstantFolder
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.utils._python_dispatch import _disable_current_modes


class _InputFreeFolder(ConstantFolder):
    """Folds what depends only on lifted tensor constants and constructors.

    User inputs, parameters and buffers are unknown, as are all placeholders in
    torch's ``ConstantFolder``; lifted tensor constants are known.
    """

    def __init__(
        self, graph_module: fx.GraphModule, constants: dict[fx.Node, torch.Tensor]
    ) -> None:
        super().__init__(graph_module, skip_constructors=False)
        self.graph_module = graph_module
        self.constants = constants

    def is_impure(self, node: fx.Node) -> bool:
        # torch's version keeps quantize/dequantize chains, which do not exist
        # before quantization, and needs torchao's ops registered to check.
        return False

    def run(self) -> object:
        env = {
            node: self.constants.get(node, self.unknown_value)
            for node in self.graph_module.graph.find_nodes(op="placeholder")
        }
        return fx.Interpreter.run(self, initial_env=env)


class FoldInputFreeSubgraphs(ExportedProgramPassBase):
    """Replace each subgraph that depends on no graph input with a lifted constant.

    User inputs, parameters and buffers count as inputs; lifted tensor constants
    do not. So the values that fold are the ones models build in-graph from
    tensor constructors such as ``full`` and ``arange`` and literal tensors, e.g.
    attention masks and position tables. Parameter- and buffer-derived values
    are left intact, so they stay trainable and visible to the quantizer.

    A bare constructor whose value no other op refines is left as is: lifting it
    would only grow the program. Values returned from the graph are left too.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        nodes_by_name = {node.name: node for node in graph.find_nodes(op="placeholder")}
        constants = {
            nodes_by_name[name]: exported_program.constants[target]
            for name, target in exported_program.graph_signature.inputs_to_lifted_tensor_constants.items()
        }
        with _disable_current_modes():
            folder = _InputFreeFolder(graph_module, constants)
            folder.run()

        output = graph.output_node()
        modified = False
        for node, value in folder.node_replacements.items():
            if not node.all_input_nodes or output in node.users:
                continue
            constant = add_constant(
                exported_program,
                f"folded_{node.name}",
                value.contiguous(),
                node,
                kind=InputKind.CONSTANT_TENSOR,
            )
            node.replace_all_uses_with(constant)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
