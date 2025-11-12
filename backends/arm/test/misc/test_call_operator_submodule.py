# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class _DepthRecordingPass(ArmPass):
    _passes_required_after = set()

    def __init__(self, initial_graph_module):
        super().__init__()
        self.depths: list[int] = []
        self.initial_submodule = initial_graph_module
        self.submodule = None
        self.num_submodules_called = 0

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        """Should only be called from the top-level graph module."""
        self.depths.append(self.submodule_depth)
        assert self.submodule == self.initial_submodule
        return super().call_operator(op, args, kwargs, meta, updated)

    def call_submodule(
        self, graph_module: GraphModule, inputs: tuple[Any, ...]
    ) -> PassResult:
        """Should be called for all three graph_modules: top-level, if, and else."""
        self.submodule = graph_module
        self.num_submodules_called += 1
        return super().call_submodule(graph_module, inputs)


class _CondModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _true_branch(arg: torch.Tensor) -> torch.Tensor:
            return arg + 1

        def _false_branch(arg: torch.Tensor) -> torch.Tensor:
            return arg - 1

        predicate = x.sum() > 0
        return torch.cond(predicate, _true_branch, _false_branch, [x])


def test_call_operator_runs_once_for_cond_submodules() -> None:
    module = _CondModule()
    example_inputs = (torch.randn(2, 3),)
    exported = torch.export.export(module, example_inputs)
    graph_module = exported.graph_module

    recording_pass = _DepthRecordingPass(graph_module)
    pass_manager = ArmPassManager(TosaSpecification.create_from_string("TOSA-1.00+FP"))
    pass_manager.add_pass(recording_pass)
    pass_manager._transform(graph_module)

    assert recording_pass.num_submodules_called == 3
    assert recording_pass.depths, "call_operator was never invoked"
    assert (
        max(recording_pass.depths) == 1
    ), "call_operator was invoked with larger than one submodule depth."
    assert (
        min(recording_pass.depths) == 1
    ), "call_operator was invoked with zero submodule depth."
