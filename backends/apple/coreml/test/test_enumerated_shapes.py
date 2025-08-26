# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import unittest

import coremltools as ct

import executorch.exir

import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)

if IS_VALID_TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestEnumeratedShapes(unittest.TestCase):
    def _compare_outputs(self, executorch_program, eager_program, example_inputs):
        if not IS_VALID_TEST_RUNTIME:
            return
        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        et_outputs = method.execute(example_inputs)[0]
        eager_outputs = eager_program(*example_inputs)
        self.assertTrue(
            torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02)
        )

    def test_e2e(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(11, 5)

            def forward(self, x, y):
                return self.linear1(x).sum() + self.linear2(y)

        model = Model()
        example_inputs = (
            torch.randn((4, 6, 10)),
            torch.randn((5, 11)),
        )
        enumerated_shapes = {"x": [[1, 5, 10], [4, 6, 10]], "y": [[3, 11], [5, 11]]}
        dynamic_shapes = [
            {
                0: torch.export.Dim.AUTO(min=1, max=4),
                1: torch.export.Dim.AUTO(min=5, max=6),
            },
            {0: torch.export.Dim.AUTO(min=3, max=5)},
        ]
        ep = torch.export.export(
            model.eval(), example_inputs, dynamic_shapes=dynamic_shapes
        )

        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        compile_specs.append(
            CoreMLBackend.generate_enumerated_shapes_compile_spec(
                ep,
                enumerated_shapes,
            )
        )
        partitioner = CoreMLPartitioner(
            compile_specs=compile_specs, lower_full_graph=True
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[partitioner],
        )
        et_prog = delegated_program.to_executorch()

        good_input1 = (
            torch.randn((1, 5, 10)),
            torch.randn((3, 11)),
        )
        good_input2 = (
            torch.randn((4, 6, 10)),
            torch.randn((5, 11)),
        )
        bad_input = (
            torch.randn((1, 5, 10)),
            torch.randn((5, 11)),
        )
        bad_input2 = (
            torch.randn((2, 7, 12)),
            torch.randn((3, 11)),
        )

        self._compare_outputs(et_prog, model, good_input1)
        self._compare_outputs(et_prog, model, good_input2)
        if IS_VALID_TEST_RUNTIME:
            self.assertRaises(
                RuntimeError, lambda: self._compare_outputs(et_prog, model, bad_input)
            )
            self.assertRaises(
                RuntimeError, lambda: self._compare_outputs(et_prog, model, bad_input2)
            )


if __name__ == "__main__":
    test_runner = TestEnumeratedShapes()
    test_runner.test_e2e()
