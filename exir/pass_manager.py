# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import inspect
import logging
from typing import Callable, List, Optional, TypeAlias, Union

import torch
import torch.fx.passes.infra.pass_manager as fx
import torch.utils._pytree as pytree
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.pass_base import (
    ExportedProgramPassResult,
    ExportedProgramPassBase,
)
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import pass_result_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

PassType: TypeAlias = Union[
    ExportedProgramPassBase, Callable[[torch.fx.GraphModule], Optional[PassResult]]
]


def _get_pass_name(fn: PassType) -> str:
    """Returns a human-readable name for a pass."""
    return fn.__name__ if inspect.isfunction(fn) else type(fn).__name__


def _is_graph_module_pass(fn: PassType) -> bool:
    """Returns True if the pass operates on GraphModule (not ExportedProgram)."""
    return not isinstance(fn, ExportedProgramPassBase)

class PassManager(fx.PassManager):
    """
    Runs multiple passes on a GraphModule.

    This is the legacy PassManager that extends torch.fx.passes.infra.pass_manager.PassManager.
    Use this when you need to run passes on a GraphModule directly.

    For running passes on ExportedProgram, use ExportedProgramPassManager instead.
    """

    def __init__(
        self,
        passes: Optional[Union[List[PassType], List[List[PassType]]]] = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
        steps: int = 1,
    ) -> None:
        logger.warning(
            "PassManager is deprecated. Please use ExportedProgramPassManager instead."
        )
        # Flatten the passes to a list of callables
        passes = passes if passes else []
        flattened_passes = [
            fx.pass_result_wrapper(fn) for fn in pytree.tree_flatten(passes)[0]
        ]

        super().__init__(
            flattened_passes,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_check_failures,
            steps=steps,
        )

    def check(self, module: torch.nn.Module) -> None:
        """
        Runs various checks on the given graph module to make sure it contains
        the needed data for passes.

        Some checks that need to be run:
            - Ensure that types of operator node match the types specified in
              the node's spec field (ex. if the op returns a tuple then the
              node's spec field is a tuple)
            - Ensure that the graph module has type torch.fx.GraphModule
        """
        assert isinstance(module, torch.fx.GraphModule)
        module.recompile()
        module.graph.lint()

        for node in module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )

class ExportedProgramPassManager(fx.PassManager):
    """
    Runs multiple passes on an ExportedProgram.

    This PassManager is specifically designed for ExportedProgram and supports
    both GraphModule-only passes and ExportedProgram-aware passes.

    For running passes on GraphModule directly, use PassManager instead.
    """

    def __init__(
        self,
        passes: Optional[Union[List[PassType], List[List[PassType]]]] = None,
        constraints: Optional[List[Callable[[Callable, Callable], bool]]] = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
        steps: int = 1,
    ) -> None:
        wrapped_passes = (
            [
                fn if isinstance(fn, ExportedProgramPassBase) else pass_result_wrapper(fn)
                for fn in pytree.tree_flatten(passes)[0]
            ]
            if passes
            else []
        )

        super().__init__(
            wrapped_passes,
            constraints=constraints,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_check_failures,
            steps=steps,
        )

    def check(self, module: torch.fx.GraphModule) -> None:
        """Validates graph module invariants."""
        module.recompile()
        module.graph.lint()

        for node in module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )

    def _run_graph_module_pass(
        self,
        fn: PassType,
        graph_module: torch.fx.GraphModule,
    ) -> PassResult:
        """Runs a pass that operates on GraphModule."""
        res = fn(graph_module)

        if res is None:
            raise TypeError(
                f"The result of pass {_get_pass_name(fn)} should be type PassResult. "
                "Please wrap it with pass_result_wrapper()"
            )

        if res.modified:
            logger.debug(
                "Graph after pass '%s': %s", _get_pass_name(fn), res.graph_module.graph
            )
            res.graph_module.recompile()

        return res

    def _run_exported_program_pass(
        self,
        fn: ExportedProgramPassBase,
        exported_program: ExportedProgram,
    ) -> ExportedProgramPassResult:
        """Runs a pass that operates on ExportedProgram."""
        res = fn(exported_program)

        if res.modified:
            logger.debug(
                "Graph after pass '%s': %s",
                _get_pass_name(fn),
                res.exported_program.graph_module.graph,
            )
            res.exported_program.graph_module.recompile()

        return res

    # pyre-ignore[14]: Intentionally overriding with different signature for ExportedProgram
    def __call__(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """
        Runs passes on an ExportedProgram.

        Handles both GraphModule-only passes and ExportedProgram-aware passes.

        Args:
            exported_program: The exported program to transform.

        Returns:
            ExportedProgramPassResult containing the transformed program.
        """
        if not self._validated:
            self.solve_constraints()

        exported_program = copy.copy(exported_program)

        # Check graph invariants before running passes
        self.check(exported_program.graph_module)

        overall_modified = False

        for _ in range(self.steps):
            step_modified = False

            for i, fn in enumerate(self.passes):
                try:
                    if _is_graph_module_pass(fn):
                        result = self._run_graph_module_pass(
                            fn, exported_program.graph_module
                        )
                        exported_program._graph_module = result.graph_module
                        step_modified = step_modified or result.modified

                        if self.run_checks_after_each_pass:
                            self.check(result.graph_module)
                    else:
                        assert isinstance(fn, ExportedProgramPassBase)
                        result = self._run_exported_program_pass(fn, exported_program)
                        exported_program = result.exported_program
                        step_modified = step_modified or result.modified

                        if self.run_checks_after_each_pass:
                            exported_program.validate()
                            self.check(exported_program.graph_module)

                except Exception as e:
                    prev_names = [_get_pass_name(p) for p in self.passes[:i]]
                    msg = f"An error occurred when running the '{_get_pass_name(fn)}' pass after the following passes: {prev_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            overall_modified = overall_modified or step_modified
            if not step_modified:
                break

        return ExportedProgramPassResult(exported_program, overall_modified)
