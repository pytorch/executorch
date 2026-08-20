# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from executorch.backends.arm._passes.arm_pass_manager import (
    _ExportedProgramGraphPassAdapter,
)
from executorch.exir.pass_manager import ExportedProgramPassManager, PassType
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


def test_exported_program_adapter_preserves_pass_names_in_errors() -> None:
    """Report wrapped pass names instead of the adapter class name."""

    def successful_pass(graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)

    def failing_pass(graph_module: GraphModule) -> PassResult:
        raise RuntimeError("test failure")

    exported_program = torch.export.export(torch.nn.ReLU(), (torch.randn(2, 3),))
    passes: list[PassType] = [
        _ExportedProgramGraphPassAdapter(successful_pass),
        _ExportedProgramGraphPassAdapter(failing_pass),
    ]

    with pytest.raises(Exception) as error:
        ExportedProgramPassManager(passes)(exported_program)

    assert "running the 'failing_pass' pass" in str(error.value)
    assert "following passes: ['successful_pass']" in str(error.value)
