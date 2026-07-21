# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pass-level tests for ``FoldScaleIntoQuantizePass``.

These run the pass on the post-``FoldAndAnnotateQParamsPass`` graph (its real
input state) and check the graph rewrite and its guards directly, without the
full lowering that ``test_attention_scale.py`` exercises end to end.
"""

import copy
import math

import torch
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.fold_scale_into_quantize_pass import (
    FoldScaleIntoQuantizePass,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

_QD = torch.ops.quantized_decomposed
_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_EDGE_CFG = EdgeCompileConfig(
    _check_ir_validity=False,
    _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
)


class _AttnDiv(torch.nn.Module):
    def forward(self, q, k):
        return torch.softmax(
            torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1
        )


class _AttnMul(torch.nn.Module):
    def forward(self, q, k):
        return torch.softmax(
            torch.bmm(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1])), dim=-1
        )


class _PoolScale(torch.nn.Module):
    # A constant scale feeding a SharedQspec pool -- the passthrough case where
    # folding must not disturb the shared scale.
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x / 4.0, 2)


class _ScaleFirstMul(torch.nn.Module):
    # Constant as the FIRST operand: mul(const, x). The constant lifts to a
    # placeholder (empty args), so the pass must not read its args[0] before
    # confirming the operand is a dequantize.
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.125))

    def forward(self, q, k):
        return torch.softmax(self.scale * torch.bmm(q, k.transpose(-2, -1)), dim=-1)


class _MultiHeadMatmulAttn(torch.nn.Module):
    # Rank-4 multi-head attention with q @ k^T scaled by 1/sqrt(head_dim), like
    # SAM's mask-decoder attention. MatmulToBmmPass rewrites the matmul to a
    # quantizable bmm at annotation, so the scale lands in the requantize sandwich
    # the fold removes -- the realistic path a bare rank-3 torch.bmm skips.
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)

    def _heads(self, x):
        b, n, c = x.shape
        return x.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        return (attn @ v).transpose(1, 2).reshape(x.shape)


_ATTN_INPUTS = (torch.rand(1, 8, 16), torch.rand(1, 8, 16))
_POOL_INPUTS = (torch.rand(1, 1, 8, 8),)
_MHA_INPUT = (torch.rand(1, 8, 32),)


def _to_edge(model, inputs):
    gm = export(model.eval(), inputs, strict=True).module()
    gm = prepare_pt2e(gm, CortexMQuantizer())
    gm(*inputs)
    gm = convert_pt2e(gm)
    return to_edge(export(gm, inputs, strict=True), compile_config=_EDGE_CFG)


def _run(edge, extra_passes):
    passes = [RemoveGetItemPass, FoldAndAnnotateQParamsPass, *extra_passes]
    return CortexMPassManager(
        copy.deepcopy(edge).exported_program(), passes=passes
    ).transform()


def _count(gm, needle):
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and needle in str(n.target)
    )


def _find(gm, needle):
    return next(n for n in gm.graph.nodes if needle in str(n.target))


def test_div_folds_and_removes_the_sandwich():
    edge = _to_edge(_AttnDiv(), _ATTN_INPUTS)
    annotated = _run(edge, [])
    folded = _run(edge, [FoldScaleIntoQuantizePass])

    assert _count(annotated.graph_module, "aten.div.Tensor") == 1
    assert _count(folded.graph_module, "aten.div.Tensor") == 0
    # The dequantize/quantize that wrapped the div go with it (one of each).
    assert _count(folded.graph_module, "dequantize_per_tensor") == (
        _count(annotated.graph_module, "dequantize_per_tensor") - 1
    )
    # bmm now feeds softmax directly.
    assert "bmm" in str(_find(folded.graph_module, "_softmax").args[0].target)


def test_mul_folds():
    edge = _to_edge(_AttnMul(), _ATTN_INPUTS)
    assert _count(_run(edge, []).graph_module, "aten.mul.Tensor") == 1
    assert (
        _count(_run(edge, [FoldScaleIntoQuantizePass]).graph_module, "aten.mul.Tensor")
        == 0
    )


def test_matmul_multihead_attention_scale_folds():
    # SAM-faithful path: the rank-4 q @ k^T matmul becomes a quantizable bmm via
    # MatmulToBmmPass at annotation, so the /sqrt(head_dim) scale lands in a
    # dequantize -> div -> quantize sandwich the fold removes. A bare rank-3
    # torch.bmm (the tests above) has its output quantized directly, so it does
    # not exercise this matmul->bmm dependency -- if that pass regresses, the
    # scale would stay fp32 and this test fails.
    edge = _to_edge(_MultiHeadMatmulAttn(), _MHA_INPUT)
    assert _count(_run(edge, []).graph_module, "aten.div.Tensor") == 1
    assert (
        _count(_run(edge, [FoldScaleIntoQuantizePass]).graph_module, "aten.div.Tensor")
        == 0
    )


def test_fold_is_bit_exact():
    # Removing the sandwich is bit-exact iff quantize(dequantize(q) / c) == q for
    # every int8 at the calibrated qparams; verify against the real kernels.
    edge = _to_edge(_AttnDiv(), _ATTN_INPUTS)
    div = _find(_run(edge, []).graph_module, "aten.div.Tensor")
    dq, quant = div.args[0], next(iter(div.users))
    s_in, zp_in = dq.args[1], dq.args[2]
    s_out, zp_out = quant.args[1], quant.args[2]

    c = math.sqrt(_ATTN_INPUTS[0].shape[-1])
    v = torch.arange(-128, 128, dtype=torch.int8)
    deq = _QD.dequantize_per_tensor.default(v, s_in, zp_in, -128, 127, torch.int8)
    requant = _QD.quantize_per_tensor.default(
        deq / c, s_out, zp_out, -128, 127, torch.int8
    )
    assert torch.equal(requant, v)


def test_sharedqspec_consumer_scale_is_untouched():
    # Adrian's case: a constant scale feeding a SharedQspec pool. The fold
    # rewrites no scale, so the pool's shared in/out qparams are identical before
    # and after -- the shared cluster can never be corrupted -- and the divide
    # still folds.
    edge = _to_edge(_PoolScale(), _POOL_INPUTS)
    annotated = _run(edge, [])
    folded = _run(edge, [FoldScaleIntoQuantizePass])

    assert _count(folded.graph_module, "aten.div.Tensor") == 0
    pool_before = _find(annotated.graph_module, "max_pool2d")
    pool_after = _find(folded.graph_module, "max_pool2d")
    # Pin the premise: max_pool2d is a SharedQspec op (input scale == output
    # scale). Without this the test would silently stop guarding Adrian's case.
    assert (
        get_input_qparams(pool_before)[0].scale
        == get_output_qparams(pool_before)[0].scale
    )
    assert (
        get_input_qparams(pool_before)[0].scale
        == get_input_qparams(pool_after)[0].scale
    )
    assert (
        get_output_qparams(pool_before)[0].scale
        == get_output_qparams(pool_after)[0].scale
    )


def _perturb_softmax_quantize(program, arg_index, fn):
    for node in program.graph_module.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is _QUANTIZE
            and any("softmax" in str(u.target) for u in node.users)
        ):
            args = list(node.args)
            args[arg_index] = fn(args[arg_index])
            node.args = tuple(args)


def test_skips_when_scale_not_absorbed():
    program = _run(_to_edge(_AttnDiv(), _ATTN_INPUTS), [])
    _perturb_softmax_quantize(program, 1, lambda s: s * 1.5)
    result = FoldScaleIntoQuantizePass(exported_program=program).call(
        program.graph_module
    )
    assert not result.modified
    assert _count(program.graph_module, "aten.div.Tensor") == 1


def test_skips_on_zero_point_mismatch():
    program = _run(_to_edge(_AttnDiv(), _ATTN_INPUTS), [])
    _perturb_softmax_quantize(program, 2, lambda z: z + 1)
    result = FoldScaleIntoQuantizePass(exported_program=program).call(
        program.graph_module
    )
    assert not result.modified
    assert _count(program.graph_module, "aten.div.Tensor") == 1


def test_constant_first_operand_does_not_crash():
    # `const * tensor` puts the constant (a placeholder with empty args) at
    # args[0]; the pass must not read its args[0] before confirming it is a
    # dequantize. It should skip, not raise (regression).
    program = _run(_to_edge(_ScaleFirstMul(), _ATTN_INPUTS), [])
    result = FoldScaleIntoQuantizePass(exported_program=program).call(
        program.graph_module
    )
    assert not result.modified
    assert _count(program.graph_module, "aten.mul.Tensor") == 1
