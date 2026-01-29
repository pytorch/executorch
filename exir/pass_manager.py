# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
from typing import Callable, cast, List, Optional, Union
import torch
import torch.fx.passes.infra.pass_manager as fx
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    LegacyPassWrapper,
)
from torch.fx.passes.infra.pass_base import PassResult

# Legacy type for backwards compatibility
LegacyPassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]

# New union that accepts both
PassType = Union[ExportedProgramPassBase, LegacyPassType]


def _normalize_pass(
    p: Union[ExportedProgramPassBase, Callable]
) -> ExportedProgramPassBase:
    """Normalize a pass to ExportedProgramPassBase, wrapping legacy callables."""
    if isinstance(p, ExportedProgramPassBase):
        return p
    elif callable(p):
        return LegacyPassWrapper(p)
    else:
        raise TypeError(f"Expected ExportedProgramPassBase or callable, got {type(p)}")


class PassManager(fx.PassManager):
    """
    Class to run multiple passes on a given graph module. The PassManager is
    callable so to run it, we can just call the PassManager instance.

    Private Attributes:
        * **passes**: A list of callable passes
        * **params**: An instance of PassManagerParams containing the result of the
            flags set in the constructor.
    """

    def __init__(
        self,
        passes: Optional[Union[List[PassType], List[List[PassType]]]] = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ) -> None:
        r"""
        Args:
            passes: A list of passes
            enable_debug_pass: set to true to enable the debug passes
            run_checks_after_each_pass: whether to run checks and linting after each pass
        """

        # Flatten the passes to a list of callables
        passes = passes if passes else []
        flattened_passes = [
            fx.pass_result_wrapper(fn) for fn in pytree.tree_flatten(passes)[0]
        ]

        normalized: list[ExportedProgramPassBase] = []
        for p in flattened_passes:
            normalized.append(_normalize_pass(p))

        flattened_passes = normalized

        super().__init__(
            flattened_passes,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_check_failures,
        )

        # Improves type-checking in call_exported_program
        self.passes = cast(list[ExportedProgramPassBase], self.passes)



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
        assert isinstance(module, fx.GraphModule)
        module.recompile()
        module.graph.lint()
        # TODO(qihan): use verifier.check_is_exir

        for node in module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )

    def call(self, module: torch.nn.Module) -> PassResult:
        """
        Runs the passes on the given graph module.

        Args:
            module: The graph module to run the passes on

        Returns:
            A PassResult object containing the modified graph module and a
            boolean indicating whether the module was modified
        """
        # Check for passes that might have ExportedProgram-specific logic
        problematic_passes = []
        for p in self.passes:
            if type(p).call_exported_program is not ExportedProgramPassBase.call_exported_program:
                problematic_passes.append(type(p).__name__)

        if problematic_passes:
            fx.logger.warning(
                f"The following passes have overridden 'call_exported_program': {problematic_passes}. "
                "Calling PassManager.call() will only run the graph module logic via 'call()' "
                "and may miss ExportedProgram-specific transformations. "
                "Consider using PassManager.call_exported_program() instead.",
                UserWarning,
                stacklevel=2,
            )

        return super()(module)

    def call_exported_program(
        self, exported_program: ExportedProgram
    ) -> ExportedProgramPassResult:
        """
        Runs the passes on the given ExportedProgram.

        Args:
            exported_program: The ExportedProgram to run the passes on

        Returns:
            A PassResult object containing the modified ExportedProgram and a
            boolean indicating whether the ExportedProgram was modified
        """
        # Order the passes based on the constraints
        if not self._validated:
            self.solve_constraints()

        # Check graph invariants
        self.check(exported_program.graph_module)

        # Run the set of passes `steps` number of times or until the graph stops
        # changing
        overall_modified = False

        # Done once at the beginning to make sure we don't modify the original
        # in place
        exported_program = copy.deepcopy(exported_program)
        for _ in range(self.steps):
            modified = False

            # Run the set of passes on the graph module
            for i, p in enumerate(self.passes):
                pass_name = type(p).__name__
                fx.logger.debug("Running pass '%s'", pass_name)

                try:
                    res = p.call_exported_program(exported_program)
                    exported_program = res.exported_program
                    modified = modified or res.modified

                    fx.logger.debug(
                        "Graph after pass '%s': %s",
                        pass_name,
                        exported_program.graph_module.graph,
                    )
                    exported_program.graph_module.recompile()

                    # Check graph invariants
                    if self.run_checks_after_each_pass:
                        self.check(exported_program.graph_module)

                except Exception as e:
                    prev_pass_names = [type(p).__name__ for p in self.passes[:i]]
                    msg = f"An error occurred when running the '{pass_name}' pass after the following passes: {prev_pass_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            # If the graph no longer changes, then we can stop running these passes
            overall_modified = overall_modified or modified
            if not modified:
                break

        return ExportedProgramPassResult(exported_program, overall_modified)
