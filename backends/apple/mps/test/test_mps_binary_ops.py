#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import inspect

import torch
from executorch.backends.apple.mps.test.test_mps_utils import TestMPS


class TestMPSAdd(TestMPS):
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x + y
            z = z + x
            z = z + x
            z = z + z
            return z

    class Add2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x + x
            return z

    class AddConstant(torch.nn.Module):
        def __init__(self, constant):
            super().__init__()
            self._constant1 = constant
            self.register_buffer("_constant2", constant, persistent=False)
            self.register_parameter("_constant3", torch.nn.Parameter(constant))

        def forward(self, x):
            out1 = x + self._constant1 + torch.ones(1, 1, 1)
            out2 = x + self._constant2 + self._constant3
            return out1, out2

    def test_fp16_add(self):
        inputs = (torch.ones(1).to(torch.float16), torch.ones(1).to(torch.float16))
        self.lower_and_test_with_partitioner(
            self.Add(), inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_fp32_add(self):
        inputs = (torch.ones(1), torch.ones(1))
        self.lower_and_test_with_partitioner(
            self.Add(), inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_fp32_add_constant(self):
        inputs = (torch.randn(4, 4, 4),)
        self.lower_and_test_with_partitioner(
            self.AddConstant(torch.ones(4, 4, 4)),
            inputs,
            func_name=inspect.stack()[0].function[5:],
        )

    def test_add_w_alpha(self):
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.add(x, y, alpha=0.1)
                return z

        add_module = AddModule()
        model_inputs = (torch.randn(1), torch.randn(1))

        self.lower_and_test_with_partitioner(
            add_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_add_scalar(self):
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                z = torch.ops.aten.add.Scalar(x, 2.0)
                return z

        add_module = AddModule()
        model_inputs = (torch.randn(2, 5),)

        self.lower_and_test_with_partitioner(
            add_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_add_scalar_int(self):
        class AddScalarModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._scalar_int = 3

            def forward(self, x):
                out1 = torch.ops.aten.add.Scalar(x, self._scalar_int)
                return out1

        add_scalar_module = AddScalarModule()
        model_inputs = (torch.randint(11, (4, 4, 4), dtype=torch.int32),)

        self.lower_and_test_with_partitioner(
            add_scalar_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_add_without_alpha(self):
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.add(x, y)
                return z

        add_module = AddModule()
        model_inputs = (torch.randn(1), torch.randn(1))

        self.lower_and_test_with_partitioner(
            add_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_add_scalar_float(self):
        class AddScalarModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._scalar_float = 3.14

            def forward(self, x):
                out = torch.ops.aten.add.Scalar(x, self._scalar_float)
                return out

        add_scalar_module = AddScalarModule()
        model_inputs = (torch.randn(4, 4, 4),)

        self.lower_and_test_with_partitioner(
            add_scalar_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_constant_add(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._constant = torch.ones(4, 4, 4)

            def forward(self, x):
                out1 = x + self._constant
                out2 = x + self._constant + self._constant
                return out1, out2

        const_module = Module()
        model_inputs = (torch.randn(4, 4, 4),)

        self.lower_and_test_with_partitioner(
            const_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )


class TestMPSSub(TestMPS):
    def test_mps_backend_sub_1(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.sub(x, y, alpha=0.1)
                return z

        sub_module = SubModule()
        model_inputs = (torch.randn(1), torch.randn(1))

        self.lower_and_test_with_partitioner(
            sub_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_sub_2(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                z = torch.ops.aten.sub.Scalar(x, 2.0)
                return z

        sub_module = SubModule()
        model_inputs = (torch.randn(2, 5),)

        self.lower_and_test_with_partitioner(
            sub_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_sub_3(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.sub(x, y)
                return z

        sub_module = SubModule()
        model_inputs = (torch.randn(1), torch.randn(1))

        self.lower_and_test_with_partitioner(
            sub_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )


class TestMPSMul(TestMPS):
    def test_mps_mul_scalar_float(self):
        class MulScalarModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._scalar = 3.14

            def forward(self, x):
                out1 = torch.ops.aten.mul.Scalar(x, self._scalar)
                return out1

        mul_scalar_module = MulScalarModule()
        model_inputs = (torch.randn(4, 4, 4),)

        self.lower_and_test_with_partitioner(
            mul_scalar_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_mul_scalar_int(self):
        class MulScalarModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._scalar = 3

            def forward(self, x):
                out1 = torch.ops.aten.mul.Scalar(x, self._scalar)
                return out1

        mul_scalar_module = MulScalarModule()
        model_inputs = (torch.randint(11, (4, 4, 4)),)

        self.lower_and_test_with_partitioner(
            mul_scalar_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )


class TestMPSDiv(TestMPS):
    def test_mps_backend_div(self):
        class DivModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x / y
                return z

        div_module = DivModule()
        model_inputs = (torch.ones(1), torch.ones(1))

        self.lower_and_test_with_partitioner(
            div_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_fmod(self):
        class FModModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.fmod(x, y)

        module = FModModule()
        model_inputs = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_floor_divide(self):
        class FloorDivideModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.floor_divide(x, y)

        module = FloorDivideModule()
        model_inputs = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )
