# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A delegate runs on the device it was lowered for, whichever device the
caller has current. Needs two CUDA devices."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY


class Dense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ELU(), torch.nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.net(x)


class Int4(torch.nn.Module):
    """A linear the int4 tinygemm shim runs, _weight_int4pack_mm."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)


# Each run is its own process: a failed one leaves its CUDA context unusable.
_RUN = """
import sys

import torch
from executorch.runtime import Runtime

directory, current = sys.argv[1], int(sys.argv[2])
torch.cuda.set_device(current)
method = Runtime.get().load_program(
    directory + "/model.pte", data_path=directory + "/aoti_cuda_blob.ptd"
).load_method("forward")
inputs = torch.load(directory + "/inputs.pt")
expected = torch.load(directory + "/expected.pt")
for _ in range(2):
    (output,) = method.execute(inputs)
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)
assert torch.cuda.current_device() == current, torch.cuda.current_device()
"""


def _export(kind: str, directory: str, target: int) -> None:
    if kind == "int4":
        from executorch.extension.llm.export.quantize import quantize_model_

        model = Int4().to("cuda")
        quantize_model_(
            model,
            qlinear_config="4w",
            qlinear_group_size=32,
            qlinear_packing_format="tile_packed_to_4d",
        )
        inputs = (torch.randn(2, 256, dtype=torch.bfloat16),)
        with torch.no_grad():
            expected = model(inputs[0].cuda()).cpu()
        example = (inputs[0].cuda(),)
    else:
        model = Dense()
        inputs = (torch.randn(1, 8),)
        with torch.no_grad():
            expected = model(inputs[0])
        example = inputs
    program = to_edge_transform_and_lower(
        torch.export.export(model.eval(), example),
        partitioner=[
            CudaPartitioner(
                [
                    CudaBackend.generate_method_name_compile_spec("forward"),
                    CompileSpec(
                        TARGET_DEVICE_COMPILE_SPEC_KEY, f"cuda:{target}".encode()
                    ),
                ]
            )
        ],
        # As the CUDA model exports set it: the int4 linear lowers to
        # _weight_int4pack_mm, which is not a core ATen op.
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).to_executorch()
    with open(Path(directory) / "model.pte", "wb") as f:
        program.write_to_file(f)
    program.write_tensor_data_to_file(directory)
    torch.save(list(inputs), Path(directory) / "inputs.pt")
    torch.save(expected, Path(directory) / "expected.pt")


# Programs on both devices in one process, with cross-method sharing enabled
# as a runner would: each runs on its own device, and freeing both trims each
# device's pool.
_RUN_BOTH = """
import gc
import sys

import torch
from executorch.runtime import Runtime

runtime = Runtime.get()
runtime.backend_registry.set_option(
    "CudaBackend", {"weight_sharing_across_methods": True}
)
methods = []
for directory in sys.argv[1:]:
    methods.append(
        (
            directory,
            runtime.load_program(
                directory + "/model.pte", data_path=directory + "/aoti_cuda_blob.ptd"
            ).load_method("forward"),
        )
    )
for _ in range(2):
    for directory, method in methods:
        inputs = torch.load(directory + "/inputs.pt")
        (output,) = method.execute(inputs)
        torch.testing.assert_close(
            output.cpu(), torch.load(directory + "/expected.pt")
        )
del methods
gc.collect()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""


@unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two CUDA devices")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class CudaTargetDeviceTest(unittest.TestCase):
    def _check(self, kind: str) -> None:
        for target in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                _export(kind, directory, target)
                for current in (0, 1):
                    with self.subTest(target=target, current=current):
                        result = subprocess.run(
                            [sys.executable, "-c", _RUN, directory, str(current)],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        self.assertEqual(
                            result.returncode,
                            0,
                            f"lowered for cuda:{target}, cuda:{current} current:\n"
                            f"{result.stdout}\n{result.stderr}",
                        )

    def test_runs_on_its_device_whichever_device_is_current(self):
        self._check("dense")

    def test_programs_on_both_devices_in_one_process(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _export("dense", first, 0)
            _export("dense", second, 1)
            result = subprocess.run(
                [sys.executable, "-c", _RUN_BOTH, first, second],
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")

    def test_int4_matmul_runs_on_the_target_device(self):
        self._check("int4")
