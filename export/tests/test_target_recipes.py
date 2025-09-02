# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest

import torch
from executorch.export import export
from executorch.export.target_recipes import get_ios_recipe
from executorch.runtime import Runtime


class TestMultiBackendTargetRecipes(unittest.TestCase):
    """Test multi-backend target recipes that combine CoreML and XNNPACK."""

    def test_ios_fp32_recipe_with_xnnpack_fallback(self) -> None:
        # Linear ops skipped by coreml but handled by xnnpack
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 2)

            def forward(self, x, y):
                a = self.linear1(x)
                b = a + y
                c = b - x
                result = self.linear2(c)
                return result

        model = Model()
        model.eval()

        example_inputs = [(torch.randn(2, 4), torch.randn(2, 4))]

        # Export using multi-backend target recipe with CoreML configured to skip linear operations
        recipe = get_ios_recipe(
            "ios-arm64-coreml-fp32",
            skip_ops_for_coreml_delegation=["aten.linear.default"],
        )

        # Export the model
        session = export(
            model=model, example_inputs=example_inputs, export_recipe=recipe
        )

        # Verify we can create executable
        executorch_program = session.get_executorch_program()
        # session.print_delegation_info()

        self.assertIsNotNone(
            executorch_program, "ExecutorTorch program should not be None"
        )

        # Assert there is an execution plan
        self.assertTrue(len(executorch_program.execution_plan) == 1)

        # Check number of partitions created
        self.assertTrue(len(executorch_program.execution_plan[0].delegates) == 3)

        # First delegate backend is Xnnpack
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[0].id,
            "XnnpackBackend",
        )

        # Second delegate backend is CoreML
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[1].id,
            "CoreMLBackend",
        )

        # Third delegate backend is Xnnpack
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[2].id,
            "XnnpackBackend",
        )

        et_runtime: Runtime = Runtime.get()
        backend_registry = et_runtime.backend_registry
        logging.info(
            f"backends registered: {et_runtime.backend_registry.registered_backend_names}"
        )
        if backend_registry.is_available(
            "CoreMLBackend"
        ) and backend_registry.is_available("XnnpackBackend"):
            logging.info("Running with CoreML and XNNPACK backends")
            et_output = session.run_method("forward", *example_inputs[0])
            logging.info(f"et output {et_output}")

    def test_ios_int8_recipe(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 2)

            def forward(self, x, y):
                a = self.linear1(x)
                b = a + y
                c = b - x
                result = self.linear2(c)
                return result

        model = Model()
        model.eval()

        example_inputs = [(torch.randn(2, 4), torch.randn(2, 4))]

        recipe = get_ios_recipe("ios-arm64-coreml-int8")

        # Export the model
        session = export(
            model=model, example_inputs=example_inputs, export_recipe=recipe
        )

        # Verify we can create executable
        executorch_program = session.get_executorch_program()
        # session.print_delegation_info()

        self.assertIsNotNone(
            executorch_program, "ExecutorTorch program should not be None"
        )

        # Assert there is an execution plan
        self.assertTrue(len(executorch_program.execution_plan) == 1)

        # Check number of partitions created
        self.assertTrue(len(executorch_program.execution_plan[0].delegates) == 1)

        # Delegate backend is CoreML
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[0].id,
            "CoreMLBackend",
        )

        et_runtime: Runtime = Runtime.get()
        backend_registry = et_runtime.backend_registry
        logging.info(
            f"backends registered: {et_runtime.backend_registry.registered_backend_names}"
        )
        if backend_registry.is_available("CoreMLBackend"):
            et_output = session.run_method("forward", *example_inputs[0])
            logging.info(f"et output {et_output}")
