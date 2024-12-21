import unittest
import os

import torch
from torch.testing import FileCheck

from torch.export import export_for_training

from executorch.extension.llm.export.export_passes import RemoveRedundantTransposes

class RemoveRedundantTransposesPassTest(unittest.TestCase):
    def _export(self, model, example_inputs):
        exported_module = export_for_training(
            model,
            example_inputs,
        )
        return exported_module.module()

    def _check(self, model, example_inputs, key, before_count, after_count):
        gm = self._export(model, example_inputs)
        FileCheck().check_count(key, before_count, exactly=True).run(
            gm.code
        )
        pass_res = RemoveRedundantTransposes()(gm)
        FileCheck().check_count(key, after_count, exactly=True).run(
            pass_res.graph_module.code
        )

    def test_transpose_removal(self):
        class TestModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                x = torch.transpose(x, 1, 2)
                return x + 1

        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                x = torch.transpose(x, 1, 2)
                x =  x + 1

                x = torch.transpose(x, 2, 3)
                x = torch.transpose(x, 2, 3)

                return x + 2

        x = torch.rand((1, 2, 3, 4))
        key = "torch.ops.aten.transpose.int"
        m = TestModule1()
        self._check(m, (x,), key, 2, 0)

        m = TestModule2()
        self._check(m, (x,), key, 4, 0)

    def test_transpose_no_removal(self):
        class TestModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                x = torch.transpose(x, 1, 2)
                x =  x + 1

                x = torch.transpose(x, 2, 3)
                x = torch.transpose(x, 1, 2)

                return x + 2

        x = torch.rand((1, 2, 3, 4))
        key = "torch.ops.aten.transpose.int"

        m = TestModule1()
        self._check(m, (x,), key, 4, 2)

        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x_1 = torch.transpose(x, 1, 2)
                x_2 = torch.transpose(x_1, 1, 2)
                x_2 =  x_2 + 1

                x = x_1 + 2
                x = torch.transpose(x, 1, 2)

                return x + x_2

        m = TestModule2()
        self._check(m, (x,), key, 3, 2)

    def test_permute_removal(self):
        class TestModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.permute(x, [0, 2, 1, 3])
                x = torch.permute(x, [0, 2, 1, 3])
                return x + 1

        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.permute(x, [0, 2, 1, 3])
                x = torch.permute(x, [0, 2, 1, 3])
                x =  x + 1

                x = torch.permute(x, [0, 1, 3, 2])
                x = torch.permute(x, [0, 1, 3, 2])

                return x + 2

        x = torch.rand((1, 2, 3, 4))
        key = "torch.ops.aten.permute.default"
        m = TestModule1()
        self._check(m, (x,), key, 2, 0)

        m = TestModule2()
        self._check(m, (x,), key, 4, 0)

    def test_permute_no_removal(self):
        class TestModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.permute(x, [0, 2, 1, 3])
                x = torch.permute(x, [0, 2, 1, 3])
                x =  x + 1

                x = torch.permute(x, [0, 1, 3, 2])
                x = torch.permute(x, [0, 2, 1, 3])

                return x + 2

        x = torch.rand((1, 2, 3, 4))
        key = "torch.ops.aten.permute.default"

        m = TestModule1()
        self._check(m, (x,), key, 4, 2)

        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x_1 = torch.permute(x, [0, 2, 1, 3])
                x_2 = torch.permute(x_1, [0, 2, 1, 3])
                x_2 =  x_2 + 1

                x = x_1 + 2
                x = torch.permute(x, [0, 2, 1, 3])

                return x + x_2

        m = TestModule2()
        self._check(m, (x,), key, 3, 2)
