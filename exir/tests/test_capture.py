# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple

import executorch.exir as exir
import executorch.exir.tests.models as models
import torch

from parameterized import parameterized


class TestCapture(unittest.TestCase):
    # pyre-ignore
    @parameterized.expand(models.MODELS)
    def test_module_call(self, model_name: str, model: torch.nn.Module) -> None:
        inputs = model.get_random_inputs()
        expected = model(*inputs)
        # TODO(ycao): Replace it with capture_multiple
        exported_program = exir.capture(model, inputs, exir.CaptureConfig())

        self.assertTrue(torch.allclose(expected, exported_program(*inputs)))

    def test_capture_multiple(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

            def method2(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return x + y - z

        module = MultipleMethodModule()
        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
            "method2": (torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)),
        }

        mmep = exir.capture_multiple(module, method_name_to_args)

        for method_name, args in method_name_to_args.items():
            eager_method = getattr(module, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_merge(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        class AnotherMultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method3(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module1 = MultipleMethodModule()
        method_name_to_args1 = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
        }

        module2 = AnotherMultipleMethodModule()
        method_name_to_args2 = {
            "method2": (torch.rand(2, 2), torch.rand(2, 2)),
            "method3": (torch.rand(2, 2),),
        }

        mmep1 = exir.capture_multiple(module1, method_name_to_args1)
        mmep2 = exir.capture_multiple(module2, method_name_to_args2)

        mmep1.merge(mmep2)
        self.assertEqual(
            len(mmep1.methods()), len(method_name_to_args1) + len(method_name_to_args2)
        )

        for method_name, args in method_name_to_args1.items():
            eager_method = getattr(module1, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep1.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

        for method_name, args in method_name_to_args2.items():
            eager_method = getattr(module2, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep1.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_merge_failure(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        class AnotherMultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method1(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method2(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module1 = MultipleMethodModule()
        method_name_to_args1 = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
        }

        module2 = AnotherMultipleMethodModule()
        method_name_to_args2 = {
            "method1": (torch.rand(2, 2), torch.rand(2, 2)),
            "method2": (torch.rand(2, 2),),
        }

        mmep1 = exir.capture_multiple(module1, method_name_to_args1)
        mmep2 = exir.capture_multiple(module2, method_name_to_args2)

        with self.assertRaisesRegex(
            AssertionError, "There already is a method named method1"
        ):
            mmep1.merge(mmep2)

    def test_capture_multiple_part_of_method(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

            def method2(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        module = MultipleMethodModule()
        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
            # Intentionally do not capture method2
        }

        mmep = exir.capture_multiple(module, method_name_to_args)

        # Check that only `forward` and `method1` are captured.
        self.assertEqual(len(mmep.methods()), 2)

        for method_name, args in method_name_to_args.items():
            eager_method = getattr(module, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_no_method_specified(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        method_name_to_args = {}

        with self.assertRaisesRegex(AssertionError, "Expected at least 1 graph module"):
            _ = exir.capture_multiple(module, method_name_to_args)

    def test_capture_multiple_program_property_access_success_forward(self) -> None:
        """
        A MultiMethodExirExportedProgram should allow property access even if
        it contains multiple methods as long as one of the method is named
        `forward`
        """

        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
        }

        mmep = exir.capture_multiple(module, method_name_to_args)
        self.assertEqual(len(mmep.methods()), 2)

        forward_method_prog = mmep.find_method("forward")
        forward_method_gm = forward_method_prog.exported_program.graph_module
        self.assertEqual(mmep.module, forward_method_gm)
        self.assertEqual(mmep.graph, forward_method_gm.graph)
        self.assertEqual(mmep.code, forward_method_gm.code)

    def test_capture_multiple_program_property_access_success_non_forward(self) -> None:
        """
        A MultiMethodExirExportedProgram should allow property access if it only
        contains a single method even if the method isn't named `forward`
        """

        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        method_name_to_args = {
            "method1": (torch.rand(2, 2),),
        }

        mmep = exir.capture_multiple(module, method_name_to_args)
        self.assertEqual(len(mmep.methods()), 1)

        method1_prog = mmep.find_method("method1")
        method1_gm = method1_prog.exported_program.graph_module
        self.assertEqual(mmep.module, method1_gm)
        self.assertEqual(mmep.graph, method1_gm.graph)
        self.assertEqual(mmep.code, method1_gm.code)

    def test_capture_multiple_program_property_access_failure(self) -> None:
        """
        A MultiMethodExirExportedProgram should NOT allow property access when
        there are multiple methods captured and none of them is named `forward`
        """

        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

            def method2(self, x: torch.Tensor) -> torch.Tensor:
                return x + x + 1

        module = MultipleMethodModule()
        method_name_to_args = {
            "method1": (torch.rand(2, 2),),
            "method2": (torch.rand(2, 2),),
        }

        mmep = exir.capture_multiple(module, method_name_to_args)
        self.assertEqual(len(mmep.methods()), 2)

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.module

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.graph

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.code

    def test_capture_multiple_non_module_callable(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        args = (torch.rand(2, 2), torch.rand(2, 2))
        mmep = exir.capture_multiple(fn, args)
        self.assertEqual(len(mmep.methods()), 1)

        eager_results = fn(*args)

        exported_method = mmep.find_method("forward")
        self.assertIsNotNone(exported_method)
        exported_results = exported_method(*args)

        self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_non_module_callable_dict_args(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
        }

        with self.assertRaisesRegex(
            AssertionError, "must be a tuple of tracing inputs"
        ):
            _ = exir.capture_multiple(fn, method_name_to_args)

    def test_capture_multiple_capture_default_forward(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        args = (torch.rand(2, 2), torch.rand(2, 2))

        mmep = exir.capture_multiple(module, args)

        self.assertEqual(len(mmep.methods()), 1)

        eager_results = module(*args)

        exported_method = mmep.find_method("forward")
        self.assertIsNotNone(exported_method)
        exported_results = exported_method(*args)

        self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_prim(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(
                self,
                prim_int: int,
                prim_str: str,
                prim_float: float,
                prim_bool: Tuple[bool, bool],
            ) -> None:
                super().__init__()
                self.prim_int = prim_int
                self.prim_str = prim_str
                self.prim_float = prim_float
                self.prim_bool = prim_bool

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def getter_int(self):
                return self.prim_int

            def getter_str(self):
                return self.prim_str

            def getter_bool(self):
                return self.prim_bool

            def getter_float(self):
                return self.prim_float

        module = MultipleMethodModule(2, "foo", 3.14, (True, False))
        args = (torch.rand(2, 2), torch.rand(2, 2))

        captured = exir.capture_multiple(
            module,
            args,
            prim_getters=["getter_int", "getter_str", "getter_bool", "getter_float"],
        )

        self.assertEqual(len(captured.methods()), 1)
        self.assertEqual(len(captured.prim_getters()), 4)
        getters = captured.prim_getters()
        self.assertEqual(getters["getter_int"], 2)
        self.assertEqual(getters["getter_float"], 3.14)
        self.assertEqual(getters["getter_str"], "foo")
        self.assertEqual(getters["getter_bool"], (True, False))
