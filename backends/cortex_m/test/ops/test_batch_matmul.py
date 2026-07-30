# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase


class CortexMBmm(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


class CortexMBmmConstantRhs(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, rhs_shape):
        super().__init__()
        self.register_buffer("rhs", torch.randn(rhs_shape))

    def forward(self, lhs):
        return torch.bmm(lhs, self.rhs)


class CortexMMatmul(torch.nn.Module):
    """``torch.matmul`` is captured as ``aten.matmul.default`` at annotation time
    and only decomposes to ``bmm`` at ``to_edge`` -- after quantization -- so it
    would never receive qparams. The pre-annotation matmul->bmm rewrite makes it
    lower to ``cortex_m.quantized_batch_matmul`` for both rank-3 @ rank-3 and
    rank>3 (leading batch dims folded to 3D and reshaped back)."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 1,
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 0,
        "executorch_exir_dialects_edge__ops_aten_matmul_default": 0,
    }

    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


class CortexMMatmulBroadcast(torch.nn.Module):
    """Broadcasting rank-3 matmul whose batch dims differ ([1,4,8] @ [2,8,4] ->
    [2,4,4]). ``aten.bmm`` requires equal batch dims, so this must NOT be
    rewritten to bmm; it falls through, stays ``aten.matmul`` -> fp32, and is not
    lowered (no cortex_m_quantized_batch_matmul)."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 0,
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
    }

    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


test_cases = {
    "bmm_small": McuTestCase(
        CortexMBmm(),
        (torch.randn(1, 2, 3), torch.randn(1, 3, 4)),
    ),
    "bmm_square": McuTestCase(
        CortexMBmm(),
        (torch.randn(2, 4, 4), torch.randn(2, 4, 4)),
    ),
    "bmm_large_batch": McuTestCase(
        CortexMBmm(),
        (torch.randn(8, 3, 5), torch.randn(8, 5, 2)),
    ),
    "bmm_single_batch": McuTestCase(
        CortexMBmm(),
        (torch.randn(1, 8, 16), torch.randn(1, 16, 8)),
    ),
    "bmm_tall_skinny": McuTestCase(
        CortexMBmm(),
        (torch.randn(2, 16, 1), torch.randn(2, 1, 16)),
    ),
    "bmm_wide_short": McuTestCase(
        CortexMBmm(),
        (torch.randn(2, 1, 16), torch.randn(2, 16, 1)),
    ),
}


const_rhs_test_cases = {
    "const_rhs_small": McuTestCase(
        CortexMBmmConstantRhs((1, 3, 4)),
        (torch.randn(1, 2, 3),),
    ),
}


matmul_test_cases = {
    "matmul_rank3": McuTestCase(
        CortexMMatmul(),
        (torch.randn(2, 4, 8), torch.randn(2, 8, 4)),
    ),
    "matmul_rank4": McuTestCase(
        CortexMMatmul(),
        (torch.randn(1, 4, 16, 8), torch.randn(1, 4, 8, 16)),
    ),
    "matmul_broadcast_rank3": McuTestCase(
        CortexMMatmulBroadcast(),
        (torch.randn(1, 4, 8), torch.randn(2, 8, 4)),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_batch_matmul(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", const_rhs_test_cases)
def test_dialect_batch_matmul_const_rhs(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", matmul_test_cases)
def test_dialect_matmul(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_batch_matmul(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)


@parametrize("test_case", const_rhs_test_cases)
def test_implementation_batch_matmul_const_rhs(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)
