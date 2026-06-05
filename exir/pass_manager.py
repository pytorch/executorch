# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import inspect
import logging
from typing import Callable, List, Optional, Type, TypeAlias, Union

import torch
import torch.fx.passes.infra.pass_manager as fx
import torch.utils._pytree as pytree
from executorch.exir._program_utils import (
    _get_updated_graph_signature,
    _get_updated_range_constraints,
)
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch._export.verifier import Verifier
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

        super().__init__(
            wrapped_passes,
            constraints=constraints,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_check_failures,
            steps=steps,
        )

    def check(self, exported_program: ExportedProgram) -> None:
        """Validates graph module invariants."""
        graph_module = exported_program.graph_module
        graph_module.recompile()
        graph_module.graph.lint()

        for node in graph_module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )

        exported_program.validate()

    # pyre-ignore[14]: Intentionally overriding with different signature for ExportedProgram
    def __call__(  # noqa: C901
        self,
        exported_program: ExportedProgram,
        override_verifiers: Optional[list[Type[Verifier]]] = None,
    ) -> ExportedProgramPassResult:
        """
        Runs passes on an ExportedProgram.

        Handles both GraphModule-only passes and ExportedProgram-aware passes. Will create a shallow copy of the exported program before running passes.

        Args:
            exported_program: The exported program to transform.

        Returns:
            ExportedProgramPassResult containing the transformed program.
        """
        if not self._validated:
            self.solve_constraints()

        exported_program = copy.copy(exported_program)

        if override_verifiers:
            exported_program._verifiers = override_verifiers

        self.check(exported_program)

        overall_modified = False

        for _ in range(self.steps):
            step_modified = False

            for i, fn in enumerate(self.passes):
                pass_modified = False
                try:
                    if not isinstance(fn, ExportedProgramPassBase):
                        res = fn(exported_program.graph_module)
                        if res is None:
                            raise TypeError(
                                f"The result of pass {_get_pass_name(fn)} should be type PassResult. "
                                "Please wrap it with pass_result_wrapper()"
                            )

                        if res.modified:
                            # Not running _update_exported_program_graph_module here because it is
                            # possible that the verifier will fail upon new ExportedProgram construction,
                            # and we should only run verification after each pass if
                            # run_checks_after_each_pass is True.
                            res.graph_module.recompile()
                            exported_program._graph_module = res.graph_module
                            exported_program._graph_signature = (
                                _get_updated_graph_signature(
                                    exported_program.graph_signature,
                                    res.graph_module,
                                )
                            )
                            exported_program._range_constraints = (
                                _get_updated_range_constraints(res.graph_module)
                            )
                            pass_modified = True

                    else:
                        assert isinstance(fn, ExportedProgramPassBase)
                        ep_res = fn(exported_program)
                        exported_program = ep_res.exported_program

                        if ep_res.modified:
                            pass_modified = True
                            exported_program.graph_module.recompile()

                    if self.run_checks_after_each_pass:
                        self.check(exported_program)

                    if pass_modified:
                        step_modified = True
                        logger.debug(
                            "Graph after pass '%s': %s",
                            _get_pass_name(fn),
                            exported_program.graph_module.graph,
                        )

                except Exception as e:
                    prev_names = [_get_pass_name(p) for p in self.passes[:i]]
                    msg = f"An error occurred when running the '{_get_pass_name(fn)}' pass after the following passes: {prev_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            overall_modified = overall_modified or step_modified
            if not step_modified:
                break

        self.check(exported_program)
        return ExportedProgramPassResult(exported_program, overall_modified)
