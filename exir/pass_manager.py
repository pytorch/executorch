# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, overload, Union

import torch
import torch.fx.passes.infra.pass_manager as fx
import torch.utils._pytree as pytree
from executorch.exir import ExportedProgram
from executorch.exir.error import ExportError, ExportErrorType
from torch.fx.passes.infra.pass_base import PassResult
from typing_extensions import TypeAlias


@dataclass
class ExportedProgramPassResult:
    """Result type for passes operating on ExportedProgram."""

    program: ExportedProgram
    modified: bool


# Type aliases for documentation/type checking only
PassType: TypeAlias = Callable[[torch.fx.GraphModule], PassResult]
ExportPassType: TypeAlias = Callable[[ExportedProgram], ExportedProgramPassResult]


class ExportedProgramPass(ABC):
    """
    Abstract base class for passes that operate on ExportedProgram.

    Subclass this when your pass needs access to ExportedProgram metadata
    (graph_signature, state_dict, etc.) rather than just the GraphModule.

    Example:
        class MyExportPass(ExportPass):
            def __call__(self, program: ExportedProgram) -> ExportPassResult:
                # Access program.graph_signature, program.state_dict, etc.
                gm = program.graph_module
                # ... transform ...
                new_program = program._update(graph_module=gm, graph_signature=new_sig)
                return ExportPassResult(new_program, modified=True)
    """

    @abstractmethod
    def __call__(self, program: ExportedProgram) -> ExportedProgramPassResult: ...


class PassManager(fx.PassManager):
    """
    Extended PassManager that supports both GraphModule and ExportedProgram inputs.

    When called with a GraphModule, behaves like the parent class and returns PassResult.
    When called with an ExportedProgram, runs passes and returns ExportPassResult.

    Passes can be either:
    - PassType: Callable[[GraphModule], PassResult] - operates on graph only
    - ExportPassType: Callable[[ExportedProgram], ExportPassResult] - operates on full program

    To mark a pass as requiring ExportedProgram, either:
    1. Set `requires_exported_program = True` as a class attribute
    2. Annotate the first parameter as `ExportedProgram`
    """

    def __init__(
        self,
        passes: Optional[Union[List[Callable], List[List[Callable]]]] = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ) -> None:
        """
        Args:
            passes: A list of passes (can be nested lists)
            run_checks_after_each_pass: whether to run checks and linting after each pass
            suppress_check_failures: whether to suppress check failures
        """
        # Flatten the passes to a list of callables
        passes = passes if passes else []
        flattened_passes: List[Callable] = [
            fx.pass_result_wrapper(fn) for fn in pytree.tree_flatten(passes)[0]
        ]

        super().__init__(
            flattened_passes,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_check_failures,
        )

    @overload
    def __call__(self, module: torch.fx.GraphModule) -> PassResult: ...

    @overload
    def __call__(self, program: ExportedProgram) -> ExportedProgramPassResult: ...

    def __call__(
        self, module_or_program: Union[torch.fx.GraphModule, ExportedProgram]
    ) -> Union[PassResult, ExportedProgramPassResult]:
        """
        Runs passes on either a GraphModule or ExportedProgram.
        """
        if isinstance(module_or_program, ExportedProgram):
            return self._run_on_exported_program(module_or_program)
        else:
            return super().__call__(module_or_program)

    def _run_on_exported_program(
        self, program: ExportedProgram
    ) -> ExportedProgramPassResult:
        """Internal implementation for ExportedProgram inputs."""
        if not self._validated:
            self.solve_constraints()

        # Check graph invariants
        self._check_exported_program(program)

        overall_modified = False
        for _ in range(self.steps):
            modified = False

            for i, fn in enumerate(self.passes):
                fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__
                fx.logger.debug("Running pass '%s'", fn_name)

                try:
                    if isinstance(fn, ExportedProgramPass):
                        # Pass operates on ExportedProgram
                        export_pass_result = fn(program)
                        modified = modified or export_pass_result.modified
                        program = export_pass_result.program
                    else:
                        # Pass operates on GraphModule
                        pass_result = fn(program.graph_module)
                        modified = modified or pass_result.modified
                        # Update the graph_module (using internal attribute)
                        # pyre-ignore[16]: Accessing private attribute
                        program._graph_module = pass_result.graph_module

                    if modified:
                        program.graph_module.recompile()

                    if self.run_checks_after_each_pass:
                        self._check_exported_program(program)

                except Exception as e:
                    prev_pass_names = [
                        p.__name__ if inspect.isfunction(p) else type(p).__name__
                        for p in self.passes[:i]
                    ]
                    msg = f"An error occurred when running the '{fn_name}' pass after the following passes: {prev_pass_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            overall_modified = overall_modified or modified
            if not modified:
                break

        return ExportedProgramPassResult(program, overall_modified)

    @overload
    def check(self, module: torch.fx.GraphModule) -> None: ...

    @overload
    def check(self, program: ExportedProgram) -> None: ...

    def check(
        self, module_or_program: Union[torch.fx.GraphModule, ExportedProgram]
    ) -> None:
        """Runs various checks on the given module or program."""
        if isinstance(module_or_program, ExportedProgram):
            self._check_exported_program(module_or_program)
        else:
            self._check_graph_module(module_or_program)

    def _check_graph_module(self, module: torch.fx.GraphModule) -> None:
        """Check invariants for GraphModule."""
        module.recompile()
        module.graph.lint()

        for node in module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )

        for p in self.passes:
            if isinstance(p, ExportedProgramPass):
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"Pass `{p}` is an ExportedProgram pass but was called with a GraphModule.",
                )

    def _check_exported_program(self, program: ExportedProgram) -> None:
        """Check invariants for ExportedProgram."""
        self._check_graph_module(program.graph_module)
        # Add any ExportedProgram-specific checks here (e.g., signature validation)
