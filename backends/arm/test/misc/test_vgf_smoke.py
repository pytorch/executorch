# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch

from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

# Smoke tests for VGF backends


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def test_vgf_aot_smoke_lowers_add_model_to_executorch_program():
    example_inputs = (
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4),
    )
    exported_program = torch.export.export(AddModule().eval(), example_inputs)

    compile_spec = VgfCompileSpec()
    partitioner = VgfPartitioner(compile_spec)

    fake_vgf_bytes = b"fake-vgf-smoke-test-binary"

    with mock.patch(
        "executorch.backends.arm.vgf.backend.vgf_compile",
        return_value=fake_vgf_bytes,
    ) as mock_vgf_compile:
        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )

        executorch_program_manager = edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )

    assert executorch_program_manager is not None
    mock_vgf_compile.assert_called_once()

    tosa_flatbuffer = mock_vgf_compile.call_args.args[0]
    compiler_flags = mock_vgf_compile.call_args.args[1]

    assert isinstance(tosa_flatbuffer, bytes)
    assert len(tosa_flatbuffer) > 0
    assert compiler_flags == []
