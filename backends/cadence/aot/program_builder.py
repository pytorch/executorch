# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from enum import auto, Enum
from typing import Optional

from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.pass_base import ProxyValue
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
from torch import Tensor
from torch._export.verifier import Verifier
from torch.export import ExportedProgram
from torch.export.exported_program import ModuleCallEntry, ModuleCallSignature
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.utils import _pytree as pytree


class IrMode(Enum):
    EXIR = auto()
    ATEN = auto()


class ProgramBuilder(GraphBuilder):
    """Utility class to build a program from a graph module."""

    def __init__(self, mode: Optional[IrMode] = None) -> None:
        self.input_specs: list[InputSpec] = []
        self.output_specs: list[OutputSpec] = []
        self.constants: dict[str, Tensor] = {}
        self.state_dict: dict[str, Tensor] = {}
        self.mode: IrMode = mode or IrMode.EXIR
        super().__init__()

    def insert_input_spec(
        self, target: str, input_kind: InputKind, value: Tensor
    ) -> None:
        persistent: Optional[bool] = None
        if input_kind == InputKind.BUFFER:
            persistent = True
        self.input_specs.append(
            InputSpec(
                input_kind, TensorArgument(target), target=target, persistent=persistent
            )
        )
        if input_kind == InputKind.PARAMETER or input_kind == InputKind.BUFFER:
            self.state_dict[target] = value
        elif input_kind == InputKind.CONSTANT_TENSOR:
            self.constants[target] = value

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

    def get_verifiers(self) -> Optional[list[Verifier]]:
        if self.mode == IrMode.ATEN:
            return None
        return [
            EXIREdgeDialectVerifier(
                edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
                class_only=True,
            )
        ]

    def get_program(self) -> ExportedProgram:
        gm = self.get_graph_module()
        graph_signature = ExportGraphSignature(self.input_specs, self.output_specs)
        in_spec = pytree.tree_flatten((tuple(graph_signature.user_inputs), {}))[1]
        out_spec = pytree.tree_flatten(graph_signature.user_outputs)[1]
        return ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=graph_signature,
            # pyre-ignore[6]: Incompatible parameter type.
            constants=self.constants,
            state_dict=self.state_dict,
            range_constraints={},
            module_call_graph=[
                ModuleCallEntry(
                    "",
                    ModuleCallSignature(
                        inputs=[], outputs=[], in_spec=in_spec, out_spec=out_spec
                    ),
                )
            ],
            # pyre-ignore[6]: Incompatible parameter type.
            verifiers=self.get_verifiers(),
        )

    def get_edge_program(self) -> EdgeProgramManager:
        return EdgeProgramManager(self.get_program())
