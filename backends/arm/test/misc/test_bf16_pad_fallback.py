# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for BF16 aten.pad fallback without the TOSA bf16
extension.
"""

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


class Pad(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (1, 1)) * 2


def _pad_node(dtype: torch.dtype):
    ep = torch.export.export(Pad(), (torch.ones(1, 4, 8, dtype=dtype),))
    node = next(
        node for node in ep.graph.nodes if node.target == torch.ops.aten.pad.default
    )
    return ep, node


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("enable_bf16", (False, True))
def test_vgf_pad_preservation_respects_bf16_extension(dtype, enable_bf16):
    spec = "TOSA-1.0+FP+INT+int4+int16"
    if enable_bf16:
        spec += "+bf16"

    ep, node = _pad_node(dtype)
    keep_ops, filter_fn = VgfPartitioner(VgfCompileSpec(spec)).ops_to_not_decompose(ep)

    assert torch.ops.aten.pad.default in keep_ops
    assert filter_fn is not None
    # FP32/FP16 are supported by the FP profile. BF16 requires +bf16.
    assert filter_fn(node) is (dtype != torch.bfloat16 or enable_bf16)


@pytest.mark.parametrize("enable_bf16", (False, True))
def test_tosa_pad_preservation_respects_bf16_extension(enable_bf16):
    spec = "TOSA-1.0+FP" + ("+bf16" if enable_bf16 else "")
    ep, node = _pad_node(torch.bfloat16)
    keep_ops, filter_fn = TOSAPartitioner(TosaCompileSpec(spec)).ops_to_not_decompose(
        ep
    )

    assert torch.ops.aten.pad.default in keep_ops
    assert filter_fn is not None
    assert filter_fn(node) is enable_bf16


def test_vgf_bf16_pad_without_extension_exports_to_pte():
    # The entire BF16 graph stays portable with
    # this explicitly BF16-disabled spec.
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    ep = torch.export.export(Pad(), (x,))
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[VgfPartitioner(VgfCompileSpec("TOSA-1.0+FP+INT+int4+int16"))],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    targets = [
        str(node.target)
        for node in edge.exported_program().graph.nodes
        if node.op == "call_function"
    ]
    assert any("constant_pad_nd" in target for target in targets)
    assert not any("aten.pad" in target for target in targets)
    assert not any("executorch_call_delegate" in target for target in targets)
    # The unfixed version raises: Missing out variants: {'aten::pad'}.
    edge.to_executorch()
