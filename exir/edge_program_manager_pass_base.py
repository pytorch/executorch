# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, TYPE_CHECKING, Union

import torch
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassResult

if TYPE_CHECKING:
    from executorch.exir.program._program import EdgeProgramManager


@dataclass(frozen=True)
class ExportedProgramPassResult:
    """Result of running a pass on an ExportedProgram."""

    exported_program: ExportedProgram
    modified: bool


class ExportedProgramPassBase(ABC):
    """
    Base interface for implementing passes that operate on ExportedProgram.
    """

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """

        self.requires(exported_program)
        res = self.call(exported_program)
        self.ensures(exported_program)
        return res

    @abstractmethod
    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """
        The pass that is run through the given exported program. To implement a
        pass, it is required to implement this function.

        Args:
            exported_program: The exported program we will run a pass on
        """

    def requires(self, exported_program: ExportedProgram) -> None:  # noqa: B027
        """
        This function will be called before the pass is run and will check that
        the given exported program contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            exported_program: The exported program we will run checks on
        """

    def ensures(self, exported_program: ExportedProgram) -> None:  # noqa: B027
        """
        This function will be called after the pass is run and will check that
        the given exported program contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            exported_program: The exported program we will run checks on
        """


@dataclass(frozen=True)
class EdgeProgramManagerPassResult:
    """Result of running a pass on an EdgeProgramManager."""

    edge_program_manager: "EdgeProgramManager"
    modified: bool


class EdgeProgramManagerPassBase(ABC):
    """
    Base interface for implementing passes that operate on EdgeProgramManager.

    This is the highest-level pass abstraction. Passes at this level can:
    - Transform individual ExportedPrograms within the manager
    - Modify constant methods
    - Split one program into multiple programs
    - Add or remove programs from the manager

    Lower-level passes (ExportedProgramPassBase, GraphModule callables) can be
    lifted to this level using the provided wrapper classes.
    """

    def __call__(
        self, epm: "EdgeProgramManager"
    ) -> EdgeProgramManagerPassResult:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """
        self.requires(epm)
        res = self.call(epm)
        self.ensures(res.edge_program_manager)
        return res

    @abstractmethod
    def call(
        self, epm: "EdgeProgramManager"
    ) -> EdgeProgramManagerPassResult:
        """
        The pass that is run on the given EdgeProgramManager. To implement a
        pass, it is required to implement this function.

        Args:
            epm: The EdgeProgramManager to transform
        """

    def requires(self, epm: "EdgeProgramManager") -> None:  # noqa: B027
        """
        This function will be called before the pass is run and will check that
        the given EdgeProgramManager contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            epm: The EdgeProgramManager we will run checks on
        """

    def ensures(self, epm: "EdgeProgramManager") -> None:  # noqa: B027
        """
        This function will be called after the pass is run and will check that
        the given EdgeProgramManager contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            epm: The EdgeProgramManager we will run checks on
        """


class GraphModuleBackedExportedProgramPassWrapper(ExportedProgramPassBase):
    """
    Wrapper that adapts a GraphModule pass to work as an ExportedProgramPassBase.

    This wrapper takes a pass that operates on GraphModule and makes it compatible
    with ExportedProgramPassBase by extracting the graph module, running the pass,
    and updating the ExportedProgram in-place.
    """

    def __init__(
        self,
        graph_module_pass: Callable[[torch.fx.GraphModule], PassResult],
    ) -> None:
        super().__init__()
        self._pass = graph_module_pass

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        from executorch.exir.program._program import (
            _get_updated_graph_signature,
            _get_updated_range_constraints,
        )

        result = self._pass(exported_program.graph_module)

        if result.modified:
            # Cannot use _update_exported_program_graph_module because it
            # runs verification, and it is not the responsibility of the
            # pass to run verification. EdgeProgram manager can
            # optionally run verification after a pass.
            result.graph_module.recompile()
            exported_program = copy.copy(exported_program)  # bypasses __init__ and _validate()

            exported_program._graph_module = result.graph_module
            exported_program._graph_signature = _get_updated_graph_signature(
                exported_program.graph_signature, result.graph_module
            )
            exported_program._range_constraints = _get_updated_range_constraints(
                result.graph_module
            )
            exported_program._module_call_graph = copy.deepcopy(
                exported_program._module_call_graph
            )
            exported_program._graph_module.meta.update(exported_program.graph_module.meta)


        return ExportedProgramPassResult(exported_program, result.modified)


class ExportedProgramToEdgeProgramManagerPassWrapper(EdgeProgramManagerPassBase):
    """
    Adapts an ExportedProgramPassBase to run on every method in an EdgeProgramManager.

    This wrapper takes a pass that operates on a single ExportedProgram and applies it
    to every method in the EdgeProgramManager, collecting results into a new EPM.
    This is where the iteration over methods lives -- not in the pass manager, and not
    in EdgeProgramManager.transform().
    """

    def __init__(self, ep_pass: ExportedProgramPassBase) -> None:
        super().__init__()
        self._pass = ep_pass

    def call(
        self, epm: "EdgeProgramManager"
    ) -> EdgeProgramManagerPassResult:
        new_epm = copy.copy(epm)
        new_epm._edge_programs = dict(epm._edge_programs)

        overall_modified = False
        for name, program in epm._edge_programs.items():
            result = self._pass(program)
            new_epm._edge_programs[name] = result.exported_program
            overall_modified = overall_modified or result.modified

        new_epm._config_methods = epm._config_methods
        return EdgeProgramManagerPassResult(new_epm, overall_modified)


PassType = Union[
    EdgeProgramManagerPassBase,
    ExportedProgramPassBase,
    Callable[[torch.fx.GraphModule], Optional[PassResult]],
]

# Passes that operate on a single method (ExportedProgram or GraphModule level).
# Excludes EdgeProgramManagerPassBase, which operates on the whole EdgeProgramManager.
# Use this for per-method pass specifications (e.g. Dict[str, Sequence[MethodPassType]]).
MethodPassType = Union[
    ExportedProgramPassBase,
    Callable[[torch.fx.GraphModule], Optional[PassResult]],
]


def _get_pass_name(fn: PassType) -> str:
    """Unwraps wrapper chain to get the underlying pass name."""
    import inspect

    if isinstance(fn, ExportedProgramToEdgeProgramManagerPassWrapper):
        return _get_pass_name(fn._pass)
    if isinstance(fn, GraphModuleBackedExportedProgramPassWrapper):
        return _get_pass_name(fn._pass)
    return fn.__name__ if inspect.isfunction(fn) else type(fn).__name__


def wrap_passes(
    passes: Sequence[PassType],
) -> list[EdgeProgramManagerPassBase]:
    """
    Wraps a list of mixed-level passes up to the EdgeProgramManager level.

    Accepts passes at three levels:
    - EdgeProgramManagerPassBase: used as-is
    - ExportedProgramPassBase: wrapped with ExportedProgramToEdgeProgramManagerPassWrapper
    - GraphModule callables: wrapped with GraphModuleBackedExportedProgramPassWrapper
      then ExportedProgramToEdgeProgramManagerPassWrapper

    Args:
        passes: A sequence of passes at any level.

    Returns:
        A list of EdgeProgramManagerPassBase passes.
    """
    from torch.fx.passes.infra.pass_manager import pass_result_wrapper

    wrapped: list[EdgeProgramManagerPassBase] = []
    for fn in passes:
        if isinstance(fn, EdgeProgramManagerPassBase):
            wrapped.append(fn)
        elif isinstance(fn, ExportedProgramPassBase):
            wrapped.append(
                ExportedProgramToEdgeProgramManagerPassWrapper(fn)
            )
        else:
            assert callable(fn)
            ep_pass = GraphModuleBackedExportedProgramPassWrapper(
                pass_result_wrapper(fn)
            )
            wrapped.append(
                ExportedProgramToEdgeProgramManagerPassWrapper(ep_pass)
            )
    return wrapped


class MethodFilteredEdgeProgramManagerPass(EdgeProgramManagerPassBase):
    """
    Applies different passes to different methods in an EdgeProgramManager.

    Converts the Dict[str, Sequence[MethodPassType]] pattern (previously handled inline
    in EdgeProgramManager.transform) into a proper pass. Used by
    to_edge_transform_and_lower to handle the dict case.
    """

    def __init__(self, passes_dict: Dict[str, Sequence[MethodPassType]]) -> None:
        super().__init__()
        self._passes_dict = passes_dict

    def call(
        self, epm: "EdgeProgramManager"
    ) -> EdgeProgramManagerPassResult:
        from executorch.exir.program._program import _transform

        new_epm = copy.copy(epm)
        new_epm._edge_programs = dict(epm._edge_programs)

        overall_modified = False
        for name, program in epm._edge_programs.items():
            if name in self._passes_dict:
                new_program = _transform(program, *self._passes_dict[name])
                new_epm._edge_programs[name] = new_program
                overall_modified = True

        return EdgeProgramManagerPassResult(new_epm, overall_modified)
