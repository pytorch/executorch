# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature, InputKind, TensorArgument


class RemoveUnusedConstantsPass(ExportedProgramPassBase):
    """Retire unused tensor constants after backend graph rewrites."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        signature = exported_program.graph_signature
        placeholders = {
            node.name: node
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        }
        protected_targets = {
            spec.target for spec in signature.output_specs if spec.target is not None
        }
        preserved_names = {
            argument.name
            for entry in exported_program.module_call_graph
            if entry.signature is not None
            for argument in (*entry.signature.inputs, *entry.signature.outputs)
            if isinstance(argument, TensorArgument)
        }
        unused = [
            spec
            for spec in signature.input_specs
            if spec.kind
            in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR)
            and spec.target not in protected_targets
            and spec.arg.name not in preserved_names
            and not placeholders[spec.arg.name].users
        ]
        if not unused:
            return ExportedProgramPassResult(exported_program, False)

        unused_names = {spec.arg.name for spec in unused}
        signature = ExportGraphSignature(
            input_specs=[
                spec
                for spec in signature.input_specs
                if spec.arg.name not in unused_names
            ],
            output_specs=list(signature.output_specs),
        )
        for spec in unused:
            exported_program.graph.erase_node(placeholders[spec.arg.name])

        remaining_targets = {spec.target for spec in signature.input_specs}
        # Entry points can share state dictionaries; retain their tensor identities.
        state_dict = exported_program.state_dict.copy()
        constants = exported_program.constants.copy()
        for spec in unused:
            if spec.target not in remaining_targets:
                state_dict.pop(spec.target, None)
                constants.pop(spec.target, None)

        exported_program._state_dict = state_dict
        exported_program._constants = constants
        exported_program._graph_signature = signature
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
