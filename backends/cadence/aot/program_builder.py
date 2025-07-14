# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from typing import Optional

from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.pass_base import ProxyValue
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier

from torch import Tensor
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)


class ProgramBuilder(GraphBuilder):
    """Utility class to build a program from a graph module."""

    def __init__(self) -> None:
        self.input_specs: list[InputSpec] = []
        self.output_specs: list[OutputSpec] = []
        self.constants: dict[str, Tensor] = {}
        self.state_dict: dict[str, Tensor] = {}
        super().__init__()

    def insert_input_spec(
        self, target: str, input_kind: InputKind, value: Tensor
    ) -> None:
        if input_kind == InputKind.USER_INPUT:
            self.input_specs.append(
                InputSpec(input_kind, TensorArgument(target), target=target)
            )

    def placeholder(
        self,
        target: str,
        fake_tensor: Tensor,
        input_kind: InputKind = InputKind.USER_INPUT,
    ) -> ProxyValue:
        placeholder = super().placeholder(target, fake_tensor)
        self.insert_input_spec(target, input_kind, fake_tensor)
        return placeholder

    def output(
        self, results: list[ProxyValue], output_kinds: Optional[list[OutputKind]] = None
    ) -> ProxyValue:
        if output_kinds is None:
            output_kinds = [OutputKind.USER_OUTPUT] * len(results)
        for result, out_kind in zip(results, output_kinds):
            self.output_specs.append(
                OutputSpec(out_kind, TensorArgument(result.node.name), target=None)
            )
        return super().output(results)

    def get_program(self) -> ExportedProgram:
        gm = self.get_graph_module()
        return ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=ExportGraphSignature(
                input_specs=self.input_specs, output_specs=self.output_specs
            ),
            # pyre-ignore[6]: Incompatible parameter type.
            constants=self.constants,
            state_dict=self.state_dict,
            range_constraints={},
            module_call_graph=[],
            verifiers=[
                EXIREdgeDialectVerifier(
                    edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
                    class_only=True,
                )
            ],
        )

    def get_edge_program(self) -> EdgeProgramManager:
        return EdgeProgramManager(self.get_program())
