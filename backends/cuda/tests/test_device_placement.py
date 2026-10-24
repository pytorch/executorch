# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY


class EachAddOnCuda(Partitioner):
    """Lowers each add to its own CUDA delegate. Parameters stay outside, so
    they cross the delegate boundary as in a partial lowering."""

    def partition(self, exported_program) -> PartitionResult:
        spec = DelegationSpec(
            CudaBackend.__name__,
            [
                CudaBackend.generate_method_name_compile_spec("forward"),
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0"),
            ],
        )
        tags = {}
        adds = [
            node
            for node in exported_program.graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == exir_ops.edge.aten.add.Tensor
        ]
        for i, node in enumerate(adds):
            node.meta["delegation_tag"] = f"add{i}"
            tags[f"add{i}"] = spec
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=tags
        )


def run(model, inputs) -> list[torch.Tensor]:
    """Lower model with EachAddOnCuda, run it in the runtime, return its outputs."""
    from executorch.runtime import Runtime

    edge = to_edge_transform_and_lower(
        torch.export.export(model, inputs, strict=True),
        partitioner=[EachAddOnCuda()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pte"
        path.write_bytes(edge.to_executorch(ExecutorchBackendConfig()).buffer)
        method = Runtime.get().load_program(path).load_method("forward")
        return [output.cpu() for output in method.execute(list(inputs))]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class CudaDevicePlacementTest(unittest.TestCase):
    def test_transposed_parameter_read_by_a_delegate(self):
        """The program stores a transposed parameter contiguous and copies
        those bytes to the device. to_edge makes the parameter contiguous
        before the delegate is compiled, so the delegate expects that layout."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = torch.arange(6.0).reshape(3, 2).t()
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def forward(self, x):
                return torch.add(x, self.weight)

        model = Model()
        x = torch.arange(6.0).reshape(2, 3) * 10
        (output,) = run(model, (x,))
        torch.testing.assert_close(output, model(x))

    def test_constant_read_by_two_delegates(self):
        """Two delegates read a constant stored in the program from outside
        them, so the program copies it to the device for them."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.arange(6.0).reshape(2, 3))

            def forward(self, x):
                return torch.add(x, self.table), torch.add(x * 2, self.table)

        model = Model()
        x = torch.arange(6.0).reshape(2, 3) * 10
        for output, expected in zip(run(model, (x,)), model(x), strict=True):
            torch.testing.assert_close(output, expected)
