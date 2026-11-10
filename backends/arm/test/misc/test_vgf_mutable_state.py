# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess  # nosec B404

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import LoweredBackendModule


class MutableFloatState(torch.nn.Module):
    def __init__(self, persistent):
        super().__init__()
        self.register_buffer("state", torch.zeros((1, 4)), persistent=persistent)

    def forward(self, value, keep):
        self.state.mul_(keep).add_(value)
        return self.state * 2.0


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("keep", [0.0, 1.0])
def test_vgf_float_state_persistence_reset_and_reload(tmp_path, persistent, keep):
    runner = os.environ.get("EXECUTORCH_VGF_TEST_RUNNER")
    if not runner:
        pytest.skip("Set EXECUTORCH_VGF_TEST_RUNNER to a VGF executor_runner")
    args = (torch.ones((1, 4)), torch.tensor([keep]))
    exported = torch.export.export(
        MutableFloatState(persistent), args
    ).run_decompositions()
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[
            VgfPartitioner(
                VgfCompileSpec("TOSA-1.0+FP+INT", alias_buffer_mutations=True)
            )
        ],
    )
    program = edge.exported_program()
    assert not program.graph_signature.buffers_to_mutate
    assert not program.graph_signature.inputs_to_buffers
    delegate = next(
        module
        for module in program.graph_module.modules()
        if isinstance(module, LoweredBackendModule)
    )
    assert (
        next(
            spec.value
            for spec in delegate.compile_specs
            if spec.key == "mutable_buffer_pairs"
        )
        == b"0:0:0"
    )
    assert not any(
        spec.key.startswith("vgf_shared_") for spec in delegate.compile_specs
    )
    model_path = tmp_path / "state.pte"
    model_path.write_bytes(edge.to_executorch().buffer)
    input_paths = [tmp_path / "value.bin", tmp_path / "keep.bin"]
    for arg, path in zip(args, input_paths):
        path.write_bytes(arg.numpy().tobytes())
    for _ in range(2):
        result = subprocess.run(  # nosec B603
            [
                runner,
                f"--model_path={model_path}",
                "--server_mode=true",
                f"--inputs={','.join(str(path) for path in input_paths)}",
                f"--output_file={tmp_path / 'output'}",
            ],
            input="\n\n\n\n",
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (result.stdout + result.stderr).count("SERVER MODE DONE") == 4
        eager = MutableFloatState(persistent)
        for _ in range(4):
            expected = eager(*args)
        actual = torch.frombuffer(
            bytearray((tmp_path / "output-0.bin").read_bytes()), dtype=torch.float32
        ).reshape_as(expected)
        torch.testing.assert_close(actual, expected)
