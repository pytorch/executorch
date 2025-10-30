import unittest

import torch

from executorch.extension.llm.export.export_passes import (
    RemoveRedundantTransposes,
    ReplaceSDPAWithCustomSDPAPass,
)

from torch.export import export
from torch.testing import FileCheck


class RemoveRedundantTransposesPassTest(unittest.TestCase):
    def _export(self, model, example_inputs):
        exported_module = export(model, example_inputs, strict=True)
        return exported_module.module()

    def _check(self, model, example_inputs, key, before_count, after_count):
        gm = self._export(model, example_inputs)
        FileCheck().check_count(key, before_count, exactly=True).run(gm.code)
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
                x = x + 1

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
                x = x + 1

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
                x_2 = x_2 + 1

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
                x = x + 1

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
                x = x + 1

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
                x_2 = x_2 + 1

                x = x_1 + 2
                x = torch.permute(x, [0, 2, 1, 3])

                return x + x_2

        m = TestModule2()
        self._check(m, (x,), key, 3, 2)


class ReplaceSDPAWithCustomSDPAPassTest(unittest.TestCase):
    class TestModule(torch.nn.Module):
        def forward(self, x, mask, is_causal):
            return torch.nn.functional.scaled_dot_product_attention(
                x, x, x, attn_mask=mask, is_causal=is_causal
            )

    def setUp(self):
        torch.manual_seed(0)

    def _test(self, args, assume_causal_mask=False):
        m = self.TestModule()
        gm = export(m, args, strict=True).module()

        sdpa_key = "torch.ops.aten.scaled_dot_product_attention.default"
        custom_sdpa_key = "torch.ops.llama.custom_sdpa.default"
        FileCheck().check_count(sdpa_key, 1, exactly=True).run(gm.code)
        gm = ReplaceSDPAWithCustomSDPAPass(assume_causal_mask)(gm).graph_module
        FileCheck().check_count(sdpa_key, 0, exactly=True).run(gm.code)
        FileCheck().check_count(custom_sdpa_key, 1, exactly=True).run(gm.code)

        y1 = m(*args)
        y2 = gm(*args)
        self.assertTrue(torch.allclose(y1, y2))

    def test_causal_mask(self):
        self._test((torch.rand(1, 4, 32, 64), None, True))

    def test_explicit_causal_mask(self):
        mask = torch.tril(torch.ones(32, 32, dtype=torch.bool))
        self._test((torch.rand(1, 4, 32, 64), mask, False), assume_causal_mask=True)

    def test_custom_mask(self):
        m1 = torch.tril(torch.ones(32, 32, dtype=torch.bool))
        m2 = torch.tril(torch.ones(32, 32, dtype=torch.bool), diagonal=-16)
        self._test((torch.rand(1, 4, 32, 64), torch.logical_xor(m1, m2), False))

    def test_squeezable_mask(self):
        m1 = torch.tril(torch.ones(32, 32, dtype=torch.bool))
        m2 = torch.tril(torch.ones(32, 32, dtype=torch.bool), diagonal=-16)
        m = torch.logical_xor(m1, m2).view(1, 1, 32, 32)
        self._test((torch.rand(1, 4, 32, 64), m, False))
