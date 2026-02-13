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
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import pass_result_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

PassType: TypeAlias = Union[
    ExportedProgramPassBase, Callable[[torch.fx.GraphModule], Optional[PassResult]]
]


def _get_pass_name(fn: PassType) -> str:
    return fn.__name__ if inspect.isfunction(fn) else type(fn).__name__


class PassManager(fx.PassManager):
    """
    Class to run multiple passes on a given graph module. The PassManager is
    callable so to run it, we can just call the PassManager instance.

    Private Attributes:
        * **passes**: A list of callable passes
        * **params**: An instance of PassManagerParams containing the result of the
            flags set in the constructor.

    Note: This class is deprecated. Please use ExportedProgramPassManager instead.
    """

    def __init__(
        self,
        passes: Optional[Union[List[PassType], List[List[PassType]]]] = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
        steps: int = 1,
    ) -> None:
        r"""
        Args:
            passes: A list of passes
            run_checks_after_each_pass: Whether to run checks and linting after each pass
            suppress_check_failures: Whether to raise errors when running checks
            steps: Number of times we wish to run passes iteratively.
        """
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
        # TODO(qihan): use verifier.check_is_exir

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
        # Setting default to True since many tests are failing pre-verification.
        suppress_exported_program_pre_verification: bool = True,
        steps: int = 1,
        override_verifiers: bool = False,
    ) -> None:
        wrapped_passes = (
            [
                (
                    fn
                    if isinstance(fn, ExportedProgramPassBase)
                    else pass_result_wrapper(fn)
                )
                for fn in pytree.tree_flatten(passes)[0]
            ]
            if passes
            else []
        )

        if suppress_exported_program_pre_verification:
            logger.warning(
                "Pre-verification of exported program is suppressed. This means that the exported program may pass validation prior to running the pass manager."
            )

        super().__init__(
            wrapped_passes,
            constraints=constraints,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_exported_program_pre_verification,
            steps=steps,
        )
        self._override_verifiers = override_verifiers

    def check(self, exported_program: ExportedProgram) -> None:
        """
        Runs exported program validation.
        """
        if not self.suppress_check_failures:
            exported_program.validate()

        module = exported_program.graph_module
        module.recompile()
        module.graph.lint()

        for node in module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )

    # pyre-ignore[14]: Intentionally overriding with different signature for ExportedProgram
    def __call__(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """
        Runs passes on an ExportedProgram.

        Handles both GraphModule-only passes and ExportedProgram-aware passes.

        Args:
            exported_program: The exported program to transform.

        Returns:
            ExportedProgramPassResult containing the transformed program and whether or
            not the program was modified.
        """
        # Lazy import to avoid circular dependency
        from executorch.exir.program._program import (
            _update_exported_program_graph_module,
        )

        if not self._validated:
            self.solve_constraints()

        exported_program = copy.copy(exported_program)

        # Check graph invariants before running passes
        self.check(exported_program)

        overall_modified = False

        for _ in range(self.steps):
            step_modified = False

            for i, fn in enumerate(self.passes):
                try:
                    if not isinstance(fn, ExportedProgramPassBase):
                        result = fn(exported_program.graph_module)
                        if result.modified:
                            logger.debug(
                                "Graph after pass '%s': %s",
                                _get_pass_name(fn),
                                result.graph_module.graph,
                            )
                            result.graph_module.recompile()

                        exported_program = _update_exported_program_graph_module(
                            exported_program,
                            result.graph_module,
                            self._override_verifiers,
                        )
                        step_modified = step_modified or result.modified

                        if self.run_checks_after_each_pass:
                            self.check(exported_program)
                    else:
                        assert isinstance(fn, ExportedProgramPassBase)
                        result = fn(exported_program)
                        if result.modified:
                            logger.debug(
                                "Graph after pass '%s': %s",
                                _get_pass_name(fn),
                                result.exported_program.graph_module.graph,
                            )
                            result.exported_program.graph_module.recompile()

                        exported_program = result.exported_program
                        step_modified = step_modified or result.modified

                        if self.run_checks_after_each_pass:
                            self.check(exported_program)

                except Exception as e:
                    prev_names = [_get_pass_name(p) for p in self.passes[:i]]
                    msg = f"An error occurred when running the '{_get_pass_name(fn)}' pass after the following passes: {prev_names}"
                    e.add_note(msg)
                    raise

            overall_modified = overall_modified or step_modified
            if not step_modified:
                break

        return ExportedProgramPassResult(exported_program, overall_modified)
