# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.extension.llm.custom_ops.custom_sdpa_schema import (
    custom_sdpa_fake,
    CustomSDPANativeOpUnavailable,
    ensure_custom_sdpa_registered,
)
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

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            ensure_custom_sdpa_registered()
        except CustomSDPANativeOpUnavailable as exc:
            raise unittest.SkipTest(
                f"llama.custom_sdpa native op is unavailable: {exc}"
            ) from exc

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

    def _assert_not_rewritten(self, model, args):
        gm = export(model, args, strict=True).module()

        custom_sdpa_key = "torch.ops.llama.custom_sdpa.default"
        gm = ReplaceSDPAWithCustomSDPAPass()(gm).graph_module
        FileCheck().check_count(custom_sdpa_key, 0, exactly=True).run(gm.code)

    def _assert_rewritten(self, model, args, assume_causal_mask=False):
        gm = export(model, args, strict=True).module()

        sdpa_key = "torch.ops.aten.scaled_dot_product_attention.default"
        custom_sdpa_key = "torch.ops.llama.custom_sdpa.default"
        FileCheck().check_count(sdpa_key, 1, exactly=True).run(gm.code)
        gm = ReplaceSDPAWithCustomSDPAPass(assume_causal_mask)(gm).graph_module
        FileCheck().check_count(sdpa_key, 0, exactly=True).run(gm.code)
        FileCheck().check_count(custom_sdpa_key, 1, exactly=True).run(gm.code)
        return gm

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

    def test_float16_mask_is_not_rewritten(self):
        mask = torch.zeros(32, 32, dtype=torch.float16)
        self._assert_not_rewritten(
            self.TestModule(),
            (torch.rand(1, 4, 32, 64, dtype=torch.float16), mask, False),
        )

    def test_broadcastable_2d_mask_is_not_rewritten(self):
        mask = torch.zeros(1, 32, dtype=torch.float32)
        self._assert_not_rewritten(
            self.TestModule(),
            (torch.rand(1, 4, 32, 64), mask, False),
        )

    def test_dropout_is_not_rewritten(self):
        class DropoutModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.5,
                    is_causal=False,
                )

        q = torch.rand(1, 4, 32, 64)
        k = torch.rand(1, 4, 32, 64)
        v = torch.rand(1, 4, 32, 64)
        self._assert_not_rewritten(DropoutModule(), (q, k, v))

    def test_unsupported_dtype_is_not_rewritten(self):
        class Float64Module(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

        q = torch.rand(1, 4, 32, 64, dtype=torch.float64)
        k = torch.rand(1, 4, 32, 64, dtype=torch.float64)
        v = torch.rand(1, 4, 32, 64, dtype=torch.float64)
        self._assert_not_rewritten(Float64Module(), (q, k, v))

    def test_gqa_decode_shape(self):
        class GQAModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    enable_gqa=True,
                )

        q = torch.rand(1, 32, 1, 128)
        k = torch.rand(1, 8, 512, 128)
        v = torch.rand(1, 8, 512, 128)

        gm = export(GQAModule(), (q, k, v), strict=True).module()
        sdpa_key = "torch.ops.aten.scaled_dot_product_attention.default"
        custom_sdpa_key = "torch.ops.llama.custom_sdpa.default"
        FileCheck().check_count(sdpa_key, 1, exactly=True).run(gm.code)

        gm = ReplaceSDPAWithCustomSDPAPass()(gm).graph_module
        FileCheck().check_count(sdpa_key, 0, exactly=True).run(gm.code)
        FileCheck().check_count(custom_sdpa_key, 1, exactly=True).run(gm.code)

        expected = GQAModule()(q, k, v)
        actual = gm(q, k, v)
        self.assertTrue(torch.allclose(expected, actual))

    def test_broadcast_mqa_without_enable_gqa_is_rewritten(self):
        class BroadcastMQAModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )

        q = torch.rand(1, 4, 3, 8)
        k = torch.rand(1, 1, 3, 8)
        v = torch.rand(1, 1, 3, 8)

        gm = self._assert_rewritten(BroadcastMQAModule(), (q, k, v))
        expected = BroadcastMQAModule()(q, k, v)
        actual = gm(q, k, v)
        self.assertTrue(torch.allclose(expected, actual))

    def test_value_embedding_dim_mismatch_is_not_supported(self):
        class ValueDimMismatchModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

        q = torch.rand(2, 4, 3, 8)
        k = torch.rand(2, 4, 5, 8)
        v = torch.rand(2, 4, 5, 6)

        self._assert_not_rewritten(ValueDimMismatchModule(), (q, k, v))

    def test_batch_broadcast_is_not_supported(self):
        class BatchBroadcastModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

        q = torch.rand(2, 4, 3, 8)
        k = torch.rand(1, 4, 5, 8)
        v = torch.rand(1, 4, 5, 8)

        self._assert_not_rewritten(BatchBroadcastModule(), (q, k, v))

    def test_head_mismatch_without_enable_gqa_is_not_supported(self):
        q = torch.rand(1, 4, 3, 4)
        k = torch.rand(1, 2, 3, 4)
        v = torch.rand(1, 2, 3, 4)
        self.assertFalse(
            ReplaceSDPAWithCustomSDPAPass._supports_head_layout(
                q,
                k,
                v,
                enable_gqa=False,
            )
        )

    def test_unmasked_non_causal_unequal_seq_len_is_not_supported(self):
        class NonCausalUnequalSeqLenModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

        q = torch.rand(1, 4, 3, 8)
        k = torch.rand(1, 4, 5, 8)
        v = torch.rand(1, 4, 5, 8)

        self._assert_not_rewritten(NonCausalUnequalSeqLenModule(), (q, k, v))

    def test_unmasked_causal_query_longer_than_kv_is_not_supported(self):
        class CausalQueryLongerThanKVModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )

        q = torch.rand(1, 4, 5, 8)
        k = torch.rand(1, 4, 3, 8)
        v = torch.rand(1, 4, 3, 8)

        self._assert_not_rewritten(CausalQueryLongerThanKVModule(), (q, k, v))

    def test_assume_causal_mask_rewrites_unmasked_prefix_attention(self):
        class PrefixAttentionModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

        q = torch.rand(1, 4, 3, 8)
        k = torch.rand(1, 4, 5, 8)
        v = torch.rand(1, 4, 5, 8)

        gm = self._assert_rewritten(
            PrefixAttentionModule(),
            (q, k, v),
            assume_causal_mask=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        actual = gm(q, k, v)
        self.assertTrue(torch.allclose(expected, actual))


class CustomSDPASchemaTest(unittest.TestCase):
    def test_custom_sdpa_fake_rejects_mismatched_kv_seq_len(self):
        q = torch.rand(1, 3, 4, 8)
        k = torch.rand(1, 5, 4, 8)
        v = torch.rand(1, 4, 4, 8)

        with self.assertRaisesRegex(AssertionError, "sequence lengths"):
            custom_sdpa_fake(q, k, v, 0, None, 0.0, False, None)

    def test_native_unavailable_is_an_import_error(self):
        self.assertTrue(issubclass(CustomSDPANativeOpUnavailable, ImportError))
