# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WebGPU op-test cases.

Declarative per-op suites for the manifest-driven op-test framework, mirroring the
Vulkan op-test authoring ergonomics. Each op reuses its `nn.Module` + generators from
the per-op `test_*.py`; new ops append a `@register_op_test` entry in their own tests-diff.
"""

import torch

from executorch.backends.webgpu.test.op_tests.test_suite import (
    Case,
    InputSpec,
    M1,
    M2,
    register_op_test,
    S,
    S1,
    S2,
    WebGPUTestSuite,
    XS,
)
from executorch.backends.webgpu.test.ops.test_add import (
    AddChainedModule,
    AddModule,
    AddSelfModule,
)
from executorch.backends.webgpu.test.ops.test_argmax import (
    argmax_tie_gen,
    ArgmaxModule,
    argmin_tie_gen,
    ArgminModule,
)
from executorch.backends.webgpu.test.ops.test_avg_pool2d import AvgPool2dModule
from executorch.backends.webgpu.test.ops.test_bitwise import (
    BitwiseAndModule,
    BitwiseNotModule,
    bw_gen_a,
    bw_gen_b,
)
from executorch.backends.webgpu.test.ops.test_cat import (
    CatModule,
    CONFIGS as _CAT_CONFIGS,
)
from executorch.backends.webgpu.test.ops.test_compare import (
    compare_gen_a,
    compare_gen_b,
    CompareModule,
)
from executorch.backends.webgpu.test.ops.test_conv1d_dw import Conv1dDWModule
from executorch.backends.webgpu.test.ops.test_conv1d_pw import Conv1dPwModule
from executorch.backends.webgpu.test.ops.test_conv_with_clamp import ConvWithClampModule
from executorch.backends.webgpu.test.ops.test_flip import FlipModule
from executorch.backends.webgpu.test.ops.test_floor_divide import FloorDivideModule
from executorch.backends.webgpu.test.ops.test_grid_priors import GridPriorsModule
from executorch.backends.webgpu.test.ops.test_grid_sampler_2d import GridSampler2dModule
from executorch.backends.webgpu.test.ops.test_group_norm import GroupNormModule
from executorch.backends.webgpu.test.ops.test_index_select import IndexSelectModule
from executorch.backends.webgpu.test.ops.test_linear_dq8ca_q4gsw import (
    make_linear_dq8ca_q4gsw_module,
)
from executorch.backends.webgpu.test.ops.test_linear_q8ta_q8csw import (
    make_linear_q8ta_q8csw_module,
)
from executorch.backends.webgpu.test.ops.test_linear_qcs4w import (
    make_qcs4w_linear_module,
)
from executorch.backends.webgpu.test.ops.test_logical_and import (
    la_gen_a,
    la_gen_b,
    LogicalAndModule,
)
from executorch.backends.webgpu.test.ops.test_logical_or import (
    BitwiseOrModule,
    lo_gen_a,
    lo_gen_b,
    LogicalOrModule,
)
from executorch.backends.webgpu.test.ops.test_minimum import MinimumModule
from executorch.backends.webgpu.test.ops.test_mul import (
    CONFIGS as _MUL_CONFIGS,
    MulModule,
)
from executorch.backends.webgpu.test.ops.test_permute import (
    CONFIGS as _PERMUTE_CONFIGS,
    PermuteModule,
)
from executorch.backends.webgpu.test.ops.test_pixel_shuffle import PixelShuffleModule
from executorch.backends.webgpu.test.ops.test_pow import PowModule
from executorch.backends.webgpu.test.ops.test_q8ta_add import Q8taAddModule
from executorch.backends.webgpu.test.ops.test_q8ta_conv2d import make_q8ta_conv2d_module
from executorch.backends.webgpu.test.ops.test_q8ta_conv2d_dw import (
    make_q8ta_conv2d_dw_module,
)
from executorch.backends.webgpu.test.ops.test_q8ta_conv2d_pw import (
    make_q8ta_conv2d_pw_module,
)
from executorch.backends.webgpu.test.ops.test_q8ta_conv2d_transposed import (
    make_q8ta_conv2d_transposed_module,
)
from executorch.backends.webgpu.test.ops.test_q8ta_linear import make_q8ta_linear_module
from executorch.backends.webgpu.test.ops.test_q8ta_pixel_shuffle import (
    Q8taPixelShuffleModule,
)
from executorch.backends.webgpu.test.ops.test_q8ta_relu import Q8taReluModule
from executorch.backends.webgpu.test.ops.test_quant import (
    DequantizeConstModule,
    QuantizeModule,
)
from executorch.backends.webgpu.test.ops.test_reduce import AmaxModule, AminModule
from executorch.backends.webgpu.test.ops.test_repeat import RepeatModule
from executorch.backends.webgpu.test.ops.test_rms_norm import (
    _CASES,
    _linspace_weight,
    _ramp,
    RmsNormModule,
)
from executorch.backends.webgpu.test.ops.test_rope_interleaved import (
    RopeInterleavedModule,
)
from executorch.backends.webgpu.test.ops.test_select import (
    CONFIGS as _SELECT_CONFIGS,
    SelectModule,
)
from executorch.backends.webgpu.test.ops.test_sigmoid import (
    _det_input as _sigmoid_det_input,
    _wide_det_input as _sigmoid_wide_det_input,
    N as _SIGMOID_N,
    SigmoidChainedModule,
    SigmoidModule,
)

from executorch.backends.webgpu.test.ops.test_slice import (
    CONFIGS as _SLICE_CONFIGS,
    SliceModule,
)

from executorch.backends.webgpu.test.ops.test_squeeze import (
    CONFIGS as _SQUEEZE_CONFIGS,
    SqueezeModule,
)

from executorch.backends.webgpu.test.ops.test_unary_activations import (
    _lin as _unary_lin,
    CLAMP_CONFIGS,
    ClampModule,
    HARDTANH_CONFIGS,
    HardtanhModule,
    POW_SCALAR_CONFIGS,
    PowScalarModule,
    UNARY_G1,
    UnaryModule,
)
from executorch.backends.webgpu.test.ops.test_unsqueeze import (
    CONFIGS as _UNSQUEEZE_CONFIGS,
    UnsqueezeModule,
)
from executorch.backends.webgpu.test.ops.test_view_copy import (
    CONFIGS as _VIEW_CONFIGS,
    ViewModule,
)

# rms_norm coverage is exactly the 15 cases the native test covered.
RMS_NORM_CASES = _CASES


def _add_factory(variant: str = "regular") -> torch.nn.Module:
    return {
        "regular": AddModule,
        "self": AddSelfModule,
        "chained": AddChainedModule,
    }[variant]()


@register_op_test("add")
def _add_suite() -> WebGPUTestSuite:
    # Same-shape numeric coverage only: broadcast adds stay export-smoke in
    # ops/test_add.py because the kernel can't broadcast.
    return WebGPUTestSuite(
        module_factory=_add_factory,
        cases=[
            Case(
                name="regular_2d",
                construct={"variant": "regular"},
                inputs=((M1, M2), (M1, M2)),
            ),
            Case(
                name="regular_3d",
                construct={"variant": "regular"},
                inputs=((S, S1, S2), (S, S1, S2)),
            ),
            Case(
                name="regular_4d",
                construct={"variant": "regular"},
                inputs=((XS, S, S1, S2), (XS, S, S1, S2)),
            ),
            Case(name="self", construct={"variant": "self"}, inputs=((M1, M2),)),
            # "scalar" (x+3.0) is intentionally OMITTED — the WebGPU add kernel can't
            # do scalar/broadcast adds (0x30 at runtime); it stays export-smoke.
            Case(
                name="chained",
                construct={"variant": "chained"},
                inputs=((M1, M2), (M1, M2)),
            ),
        ],
    )


def _rms_norm_factory(hidden: int, eps: float, weight_fn) -> torch.nn.Module:
    model = RmsNormModule(hidden, eps=eps)
    with torch.no_grad():
        model.weight.copy_(weight_fn(hidden))
    return model


@register_op_test("rms_norm")
def _rms_norm_suite() -> WebGPUTestSuite:
    cases = []
    for c in RMS_NORM_CASES:
        shape = c["shape"]
        hidden = shape[-1]
        weight_fn = c.get("weight_fn", _linspace_weight)
        input_fn = c.get("input_fn", _ramp)
        cases.append(
            Case(
                name=c["name"],
                construct={"hidden": hidden, "eps": 1e-6, "weight_fn": weight_fn},
                inputs=(InputSpec(shape=shape, gen=input_fn),),
            )
        )
    return WebGPUTestSuite(module_factory=_rms_norm_factory, cases=cases)


@register_op_test("mul")
def _mul_suite() -> WebGPUTestSuite:
    # Full numeric coverage incl. broadcast (binary_mul.wgsl over a TensorMeta UBO); fp64 golden.
    return WebGPUTestSuite(
        module_factory=lambda: MulModule(),
        cases=[
            Case(name=name, inputs=(sa, sb)) for name, (sa, sb) in _MUL_CONFIGS.items()
        ],
    )


def _fn_config_suite(module_cls, configs) -> WebGPUTestSuite:
    """Builder for ops whose per-case spec is a (shape, fn) pair (view/select/slice).
    The fn is a `construct` kwarg baked into the .pte module, never a serialized input.
    """
    return WebGPUTestSuite(
        module_factory=lambda fn: module_cls(fn),
        cases=[
            Case(name=n, construct={"fn": fn}, inputs=(shape,))
            for n, (shape, fn) in configs.items()
        ],
        golden_dtype="float32",  # gather/copy: fp64 bit-identical, skip dual-oracle
    )


@register_op_test("minimum")
def _minimum_suite() -> WebGPUTestSuite:
    # Same-shape numeric coverage (flat binary kernel; broadcast stays smoke).
    return WebGPUTestSuite(
        module_factory=lambda: MinimumModule(),
        cases=[
            Case(name="2d", inputs=((M1, M2), (M1, M2))),
            Case(name="3d", inputs=((S, S1, S2), (S, S1, S2))),
        ],
    )


def _compare_suite(op: str) -> WebGPUTestSuite:
    # Elementwise fp32 comparison -> bool (byte-exact golden). The two inputs use
    # DIFFERENT discrete-range seeds so a!=b (real lt/gt mix) while colliding
    # often (eq/le/ge ties); all shapes have numel % 4 == 0 (bool output packs 4
    # bytes/word). Same-shape only (flat kernel; broadcast=smoke).
    def case(name, shape):
        return Case(
            name=name,
            inputs=(
                InputSpec(shape=shape, gen=compare_gen_a),
                InputSpec(shape=shape, gen=compare_gen_b),
            ),
        )

    return WebGPUTestSuite(
        module_factory=lambda: CompareModule(op),
        cases=[case("2d", (4, 8)), case("3d", (2, 3, 8)), case("sq", (16, 16))],
        golden_dtype="bool",
    )


@register_op_test("eq")
def _eq_suite() -> WebGPUTestSuite:
    return _compare_suite("eq")


@register_op_test("lt")
def _lt_suite() -> WebGPUTestSuite:
    return _compare_suite("lt")


@register_op_test("le")
def _le_suite() -> WebGPUTestSuite:
    return _compare_suite("le")


@register_op_test("gt")
def _gt_suite() -> WebGPUTestSuite:
    return _compare_suite("gt")


@register_op_test("ge")
def _ge_suite() -> WebGPUTestSuite:
    return _compare_suite("ge")


@register_op_test("logical_and")
def _logical_and_suite() -> WebGPUTestSuite:
    # out = (a>0) && (b>0): two bool masks derived on-GPU from float inputs via
    # gt.Tensor (baked zeros), AND'd -> bool. Distinct a/b seeds so the masks
    # differ (AND ~25% True, a real mix an OR mutant fails); all shapes numel %
    # 4 == 0 (bool packs 4/word). float32 oracle (byte-exact bool golden).
    def case(name, shape):
        return Case(
            name=name,
            construct={"shape": shape},
            inputs=(
                InputSpec(shape=shape, gen=la_gen_a),
                InputSpec(shape=shape, gen=la_gen_b),
            ),
        )

    return WebGPUTestSuite(
        module_factory=lambda shape: LogicalAndModule(shape),
        cases=[case("2d", (4, 8)), case("3d", (2, 3, 8)), case("sq", (16, 16))],
        golden_dtype="float32",
    )


@register_op_test("bitwise_and")
def _bitwise_and_suite() -> WebGPUTestSuite:
    # bool bitwise AND == logical_and for canonical 0/1 (shares the handler).
    # Two masks derived on-GPU from float inputs via gt.Tensor (baked zeros),
    # distinct a/b seeds (AND ~25% True); all shapes numel % 4 == 0.
    def case(name, shape):
        return Case(
            name=name,
            construct={"shape": shape},
            inputs=(
                InputSpec(shape=shape, gen=bw_gen_a),
                InputSpec(shape=shape, gen=bw_gen_b),
            ),
        )

    return WebGPUTestSuite(
        module_factory=lambda shape: BitwiseAndModule(shape),
        cases=[case("2d", (4, 8)), case("3d", (2, 3, 8)), case("sq", (16, 16))],
        golden_dtype="float32",
    )


@register_op_test("bitwise_not")
def _bitwise_not_suite() -> WebGPUTestSuite:
    # bool NOT (1-x): one mask derived on-GPU from a float input via gt.Tensor
    # (baked zeros), inverted -> bool (~50% True); all shapes numel % 4 == 0.
    def case(name, shape):
        return Case(
            name=name,
            construct={"shape": shape},
            inputs=(InputSpec(shape=shape, gen=bw_gen_a),),
        )

    return WebGPUTestSuite(
        module_factory=lambda shape: BitwiseNotModule(shape),
        cases=[case("2d", (4, 8)), case("3d", (2, 3, 8)), case("sq", (16, 16))],
        golden_dtype="float32",
    )


@register_op_test("logical_or")
def _logical_or_suite() -> WebGPUTestSuite:
    # out = (a>0) || (b>0): two bool masks derived on-GPU from float inputs via
    # gt.Tensor (baked zeros), OR'd -> bool. Distinct a/b seeds (~50% each,
    # independent -> OR ~75% True, a real mix an AND mutant fails); all shapes
    # numel % 4 == 0. float32 oracle (byte-exact bool golden).
    def case(name, shape):
        return Case(
            name=name,
            construct={"shape": shape},
            inputs=(
                InputSpec(shape=shape, gen=lo_gen_a),
                InputSpec(shape=shape, gen=lo_gen_b),
            ),
        )

    return WebGPUTestSuite(
        module_factory=lambda shape: LogicalOrModule(shape),
        cases=[case("2d", (4, 8)), case("3d", (2, 3, 8)), case("sq", (16, 16))],
        golden_dtype="float32",
    )


@register_op_test("bitwise_or")
def _bitwise_or_suite() -> WebGPUTestSuite:
    # bool bitwise OR == logical_or for canonical 0/1 (shares the handler).
    def case(name, shape):
        return Case(
            name=name,
            construct={"shape": shape},
            inputs=(
                InputSpec(shape=shape, gen=lo_gen_a),
                InputSpec(shape=shape, gen=lo_gen_b),
            ),
        )

    return WebGPUTestSuite(
        module_factory=lambda shape: BitwiseOrModule(shape),
        cases=[case("2d", (4, 8)), case("3d", (2, 3, 8)), case("sq", (16, 16))],
        golden_dtype="float32",
    )


@register_op_test("pow")
def _pow_suite() -> WebGPUTestSuite:
    # Positive base avoids pow(neg, frac)=NaN; exponent spans negative+positive.
    return WebGPUTestSuite(
        module_factory=lambda: PowModule(),
        cases=[
            Case(
                name="2d",
                inputs=(
                    InputSpec(shape=(M1, M2), gen=_unary_lin(0.1, 3.0)),
                    InputSpec(shape=(M1, M2), gen=_unary_lin(-2.0, 3.0)),
                ),
            ),
            Case(
                name="3d",
                inputs=(
                    InputSpec(shape=(S, S1, S2), gen=_unary_lin(0.1, 3.0)),
                    InputSpec(shape=(S, S1, S2), gen=_unary_lin(-2.0, 3.0)),
                ),
            ),
        ],
    )


def _floor_div_golden(module, inputs):
    # Vulkan-faithful oracle: floor(a/b) in fp32 (Vulkan glsl OPERATOR floor(X/Y)),
    # NOT torch's fmod-corrected div_floor which can differ by 1 at fp boundaries.
    return torch.floor(inputs[0] / inputs[1])


@register_op_test("floor_divide")
def _floor_divide_suite() -> WebGPUTestSuite:
    # aten.div.Tensor_mode; divisor bounded away from zero. golden_fn = the
    # Vulkan floor(a/b) formula (same formula as the fp32 kernel).
    return WebGPUTestSuite(
        module_factory=lambda: FloorDivideModule(),
        cases=[
            Case(
                name="2d",
                inputs=(
                    InputSpec(shape=(M1, M2), gen=_unary_lin(-8.0, 8.0)),
                    InputSpec(shape=(M1, M2), gen=_unary_lin(0.5, 4.0)),
                ),
                golden_fn=_floor_div_golden,
            ),
            Case(
                name="3d",
                inputs=(
                    InputSpec(shape=(S, S1, S2), gen=_unary_lin(-8.0, 8.0)),
                    InputSpec(shape=(S, S1, S2), gen=_unary_lin(0.5, 4.0)),
                ),
                golden_fn=_floor_div_golden,
            ),
        ],
    )


def _reduce_suite(module_cls) -> WebGPUTestSuite:
    # Last-dim reduction; both keepdim variants over a 2d and a 3d shape.
    return WebGPUTestSuite(
        module_factory=lambda keepdim: module_cls(keepdim),
        cases=[
            Case(name="keepdim_2d", construct={"keepdim": True}, inputs=((M1, M2),)),
            Case(name="nodim_2d", construct={"keepdim": False}, inputs=((M1, M2),)),
            Case(
                name="keepdim_3d",
                construct={"keepdim": True},
                inputs=((S, S1, S2),),
            ),
            Case(
                name="nodim_3d",
                construct={"keepdim": False},
                inputs=((S, S1, S2),),
            ),
        ],
    )


@register_op_test("amax")
def _amax_suite() -> WebGPUTestSuite:
    return _reduce_suite(AmaxModule)


@register_op_test("amin")
def _amin_suite() -> WebGPUTestSuite:
    return _reduce_suite(AminModule)


@register_op_test("flip")
def _flip_suite() -> WebGPUTestSuite:
    # Reverse various dims; pure data movement -> float32 oracle.
    return WebGPUTestSuite(
        module_factory=lambda dims: FlipModule(dims),
        cases=[
            Case(name="last", construct={"dims": [-1]}, inputs=((M1, M2),)),
            Case(name="dim0", construct={"dims": [0]}, inputs=((M1, M2),)),
            Case(name="both_3d", construct={"dims": [0, 2]}, inputs=((S, S1, S2),)),
            Case(name="mid_3d", construct={"dims": [1]}, inputs=((S, S1, S2),)),
            Case(
                name="multi_4d",
                construct={"dims": [1, 3]},
                inputs=((XS, S, S1, S2),),
            ),
        ],
        golden_dtype="float32",
    )


@register_op_test("repeat")
def _repeat_suite() -> WebGPUTestSuite:
    # Tile along dims; pure data movement -> float32 oracle.
    return WebGPUTestSuite(
        module_factory=lambda repeats: RepeatModule(repeats),
        cases=[
            Case(name="tile_1d", construct={"repeats": [2]}, inputs=((XS,),)),
            Case(name="tile_2d", construct={"repeats": [2, 2]}, inputs=((XS, S),)),
            Case(
                name="prepend_3d",
                construct={"repeats": [1, 3, 2]},
                inputs=((XS, S),),
            ),
            Case(
                name="prepend_ext",
                construct={"repeats": [2, 3, 1]},
                inputs=((XS, S),),
            ),
            Case(
                name="tile_3d",
                construct={"repeats": [2, 1, 2]},
                inputs=((XS, S, S1),),
            ),
        ],
        golden_dtype="float32",
    )


@register_op_test("conv1d_pw")
def _conv1d_pw_suite() -> WebGPUTestSuite:
    # Pointwise 1D conv (aten.convolution, K=1 matmul); fp64 oracle.
    def mk(in_channels, out_channels, bias):
        return Conv1dPwModule(in_channels, out_channels, bias)

    def case(name, N, ic, oc, L, bias):
        return Case(
            name=name,
            construct={"in_channels": ic, "out_channels": oc, "bias": bias},
            inputs=((N, ic, L),),
        )

    return WebGPUTestSuite(
        module_factory=mk,
        cases=[
            case("ic4_oc6", 1, 4, 6, 5, True),
            case("square", 1, 3, 3, 7, True),
            case("oc2_nobias", 1, 5, 2, 4, False),
            case("ic8_oc8", 1, 8, 8, 3, True),
            case("batch2", 2, 3, 4, 5, True),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("conv1d_dw")
def _conv1d_dw_suite() -> WebGPUTestSuite:
    # Depthwise 1D conv (aten.convolution, depthwise config); fp64 oracle.
    def mk(C, kernel, stride, padding, dilation, bias):
        return Conv1dDWModule(C, kernel, stride, padding, dilation, bias)

    def case(name, C, L, kernel, stride, padding, dilation, bias):
        return Case(
            name=name,
            construct={
                "C": C,
                "kernel": kernel,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "bias": bias,
            },
            inputs=((1, C, L),),
        )

    return WebGPUTestSuite(
        module_factory=mk,
        cases=[
            case("k3s1p1", 4, 8, 3, 1, 1, 1, True),
            case("k3s2p1", 4, 8, 3, 2, 1, 1, True),
            case("dil2", 3, 10, 3, 1, 2, 2, True),
            case("k5_nobias", 5, 7, 5, 1, 0, 1, False),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("apply_rotary_emb_interleaved")
def _rope_interleaved_suite() -> WebGPUTestSuite:
    # Pair-interleaved rope (direct custom-op call); CPU eager golden.
    def case(name, in_shape, freqs_shape):
        return Case(name=name, construct={}, inputs=(in_shape, freqs_shape))

    return WebGPUTestSuite(
        module_factory=lambda: RopeInterleavedModule(),
        cases=[
            case("bnc", (1, 4, 8), (4, 8)),
            case("batch", (2, 3, 8), (3, 8)),
            case("c4", (1, 5, 4), (5, 4)),
            case("c16", (1, 2, 16), (2, 16)),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("grid_sampler_2d")
def _grid_sampler_2d_suite() -> WebGPUTestSuite:
    # Bilinear grid sample (border, align_corners); real fp math -> fp64 oracle.
    return WebGPUTestSuite(
        module_factory=lambda: GridSampler2dModule(),
        cases=[
            Case(name="sq", construct={}, inputs=((1, 2, 4, 4), (1, 3, 3, 2))),
            Case(
                name="wide_in",
                construct={},
                inputs=((1, 1, 3, 5), (1, 4, 4, 2)),
            ),
            Case(name="batch", construct={}, inputs=((2, 3, 4, 4), (2, 2, 6, 2))),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("pixel_shuffle")
def _pixel_shuffle_suite() -> WebGPUTestSuite:
    # Channel->space rearrange; pure data movement -> float32 oracle.
    return WebGPUTestSuite(
        module_factory=lambda r: PixelShuffleModule(r),
        cases=[
            Case(name="r2", construct={"r": 2}, inputs=((1, 8, 2, 3),)),
            Case(name="r2_batch", construct={"r": 2}, inputs=((2, 4, 3, 3),)),
            Case(name="r3", construct={"r": 3}, inputs=((1, 9, 2, 2),)),
            Case(name="r2_3d", construct={"r": 2}, inputs=((4, 2, 2),)),
        ],
        golden_dtype="float32",
    )


@register_op_test("avg_pool2d")
def _avg_pool2d_suite() -> WebGPUTestSuite:
    # Windowed spatial average; real fp math -> float64 oracle at 1e-3.
    def mk(kernel, stride, padding, count_include_pad, ceil_mode, divisor):
        return AvgPool2dModule(
            kernel, stride, padding, count_include_pad, ceil_mode, divisor
        )

    def case(name, k, s, p, cip, ceil_mode, divisor, shape):
        return Case(
            name=name,
            construct={
                "kernel": k,
                "stride": s,
                "padding": p,
                "count_include_pad": cip,
                "ceil_mode": ceil_mode,
                "divisor": divisor,
            },
            inputs=(shape,),
        )

    return WebGPUTestSuite(
        module_factory=mk,
        cases=[
            case("basic", [2, 2], [2, 2], [0, 0], True, False, None, (1, 2, 4, 4)),
            case("pad_cip", [3, 3], [2, 2], [1, 1], True, False, None, (1, 2, 5, 5)),
            case("pad_nocip", [3, 3], [2, 2], [1, 1], False, False, None, (1, 2, 5, 5)),
            case("asym", [3, 2], [2, 3], [1, 1], True, False, None, (2, 3, 5, 7)),
            case("divisor", [2, 2], [2, 2], [0, 0], True, False, 3, (1, 1, 4, 4)),
            # ceil_mode: last window overhangs -> exercises the overhang divisor
            # branch (beh/bew > 0) + the ceil output-size (3x3 vs floor 2x2).
            case("ceil_cip", [2, 2], [2, 2], [0, 0], True, True, None, (1, 1, 5, 5)),
            case("ceil_nocip", [3, 3], [2, 2], [0, 0], False, True, None, (1, 2, 5, 5)),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("native_group_norm")
def _group_norm_suite() -> WebGPUTestSuite:
    # 2-pass norm returning (out, mean, rstd); the multi-output golden verifies
    # both the reduce (mean/rstd) and normalize (out) passes. float64 oracle.
    return WebGPUTestSuite(
        module_factory=lambda num_channels, num_groups: GroupNormModule(
            num_channels, num_groups
        ),
        cases=[
            Case(
                name="c4_g2",
                construct={"num_channels": 4, "num_groups": 2},
                inputs=((2, 4, 3, 5),),
            ),
            Case(
                name="c6_g3",
                construct={"num_channels": 6, "num_groups": 3},
                inputs=((1, 6, 2, 2),),
            ),
            Case(
                name="c4_g1",
                construct={"num_channels": 4, "num_groups": 1},
                inputs=((1, 4, 3, 3),),
            ),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("index_select")
def _index_select_suite() -> WebGPUTestSuite:
    # Gather rows along dim via a baked int index; float32 oracle. Only the float
    # input is a runtime tensor (the index is a graph constant).
    return WebGPUTestSuite(
        module_factory=lambda dim, index: IndexSelectModule(dim, index),
        cases=[
            Case(
                name="dim0_1d",
                construct={"dim": 0, "index": [0, 2, 4, 1]},
                inputs=((S,),),
            ),
            Case(
                name="dim0_2d",
                construct={"dim": 0, "index": [3, 0, 1]},
                inputs=((XS + 1, S1),),
            ),
            Case(
                name="dim1_2d",
                construct={"dim": 1, "index": [5, 2, 0, 2]},
                inputs=((XS + 1, S1),),
            ),
            Case(
                name="dim1_3d",
                construct={"dim": 1, "index": [4, 0, 2]},
                inputs=((XS, S, S1),),
            ),
            Case(
                name="dim2_3d",
                construct={"dim": 2, "index": [6, 1, 4]},
                inputs=((XS, S, S1),),
            ),
        ],
        golden_dtype="float32",
    )


@register_op_test("conv_with_clamp")
def _conv_with_clamp_suite() -> WebGPUTestSuite:
    # nn.Conv2d + F.relu6 -> delegated et_vk.conv_with_clamp (fp32 conv + clamp
    # [0,6]). Golden = fp32 eager. Covers k3/stride/dilation/no_bias (groups==1).
    def case(name, shape, ic, oc, k, stride, padding, dilation, bias=True):
        return Case(
            name=name,
            construct={
                "ic": ic,
                "oc": oc,
                "k": k,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "bias": bias,
            },
            inputs=(shape,),
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: ConvWithClampModule(**kw),
        cases=[
            case("k3p1", (1, 4, 8, 8), 4, 8, 3, 1, 1, 1),
            case("stride2", (1, 3, 10, 10), 3, 6, 3, 2, 1, 1),
            case("dil2", (2, 3, 9, 9), 3, 5, 3, 1, 2, 2),
            case("no_bias", (1, 4, 8, 8), 4, 8, 3, 1, 1, 1, bias=False),
            # Fully axis-asymmetric (Kh!=Kw, H!=W, sh!=sw, ph!=pw, dh!=dw) so an
            # H<->W index swap would diverge from the golden.
            case("asym", (1, 3, 7, 9), 3, 5, (2, 3), (1, 2), (1, 0), (2, 1)),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("grid_priors")
def _grid_priors_suite() -> WebGPUTestSuite:
    # Detection anchor-grid op: out [H*W, 2] from the input's H/W (values unused)
    # + baked stride/offset. Pure computed output -> float32 oracle. `offset0`
    # covers offset=0; the shapes span square + non-square H/W.
    def case(name, shape, stride, offset):
        return Case(
            name=name,
            construct={"stride": stride, "offset": offset},
            inputs=(shape,),
        )

    return WebGPUTestSuite(
        module_factory=lambda stride, offset: GridPriorsModule(stride, offset),
        cases=[
            case("s8", (1, 3, 8, 10), 8, 0.5),
            case("s16", (1, 3, 4, 4), 16, 0.0),
            case("offset0", (1, 3, 5, 7), 4, 0.0),
        ],
        golden_dtype="float32",
    )


@register_op_test("view_copy")
def _view_copy_suite() -> WebGPUTestSuite:
    return _fn_config_suite(ViewModule, _VIEW_CONFIGS)


@register_op_test("select")
def _select_suite() -> WebGPUTestSuite:
    return _fn_config_suite(SelectModule, _SELECT_CONFIGS)


def _sigmoid_full_range(_shape) -> torch.Tensor:
    # Reuses the monolith's saturation-tail input (linspace(-12, 12)).
    return _sigmoid_det_input()


def _sigmoid_wide_range(_shape) -> torch.Tensor:
    return _sigmoid_wide_det_input()


def _sigmoid_factory(variant: str = "regular") -> torch.nn.Module:
    return {
        "regular": SigmoidModule,
        "chained": SigmoidChainedModule,
    }[variant]()


@register_op_test("sigmoid")
def _sigmoid_suite() -> WebGPUTestSuite:
    # sigmoid has no CONFIGS table; cover unary shapes directly (tol 1e-4).
    return WebGPUTestSuite(
        module_factory=_sigmoid_factory,
        cases=[
            Case(name="vec", inputs=((M1,),)),
            Case(name="mat", inputs=((M1, M2),)),
            Case(name="rank3", inputs=((S1, M1, M2),)),
            Case(name="rank4", inputs=((S1, S2, S2, M2),)),
            # Saturation tails sigmoid(+-12) (~6e-6 / 0.999994) that randn shapes miss.
            Case(
                name="saturation",
                inputs=(InputSpec(shape=(_SIGMOID_N,), gen=_sigmoid_full_range),),
            ),
            Case(
                name="wide_saturation",
                inputs=(InputSpec(shape=(_SIGMOID_N,), gen=_sigmoid_wide_range),),
            ),
            Case(
                name="chained",
                construct={"variant": "chained"},
                inputs=(InputSpec(shape=(_SIGMOID_N,), gen=_sigmoid_full_range),),
            ),
        ],
        atol=1e-4,
        rtol=1e-4,
    )


@register_op_test("squeeze")
def _squeeze_suite() -> WebGPUTestSuite:
    # CONFIGS: name -> (shape, dim) where dim is an int or a tuple.
    return WebGPUTestSuite(
        module_factory=lambda dim: SqueezeModule(dim),
        cases=[
            Case(name=n, construct={"dim": dim}, inputs=(shape,))
            for n, (shape, dim) in _SQUEEZE_CONFIGS.items()
        ],
        golden_dtype="float32",  # reshape copies values; fp64 bit-identical
    )


@register_op_test("unsqueeze")
def _unsqueeze_suite() -> WebGPUTestSuite:
    # CONFIGS: name -> (shape, dim).
    return WebGPUTestSuite(
        module_factory=lambda dim: UnsqueezeModule(dim),
        cases=[
            Case(name=n, construct={"dim": dim}, inputs=(shape,))
            for n, (shape, dim) in _UNSQUEEZE_CONFIGS.items()
        ],
        golden_dtype="float32",  # reshape copies values; fp64 bit-identical
    )


@register_op_test("slice")
def _slice_suite() -> WebGPUTestSuite:
    return _fn_config_suite(SliceModule, _SLICE_CONFIGS)


@register_op_test("permute")
def _permute_suite() -> WebGPUTestSuite:
    # CONFIGS: name -> (shape, perm-tuple).
    return WebGPUTestSuite(
        module_factory=lambda perm: PermuteModule(perm),
        cases=[
            Case(name=n, construct={"perm": perm}, inputs=(shape,))
            for n, (shape, perm) in _PERMUTE_CONFIGS.items()
        ],
        golden_dtype="float32",  # permutation reorders values; fp64 bit-identical
    )


@register_op_test("cat")
def _cat_suite() -> WebGPUTestSuite:
    # CONFIGS: name -> (list_of_input_shapes, dim). Variadic input count per case.
    return WebGPUTestSuite(
        module_factory=lambda dim: CatModule(dim),
        cases=[
            Case(name=n, construct={"dim": dim}, inputs=tuple(shapes))
            for n, (shapes, dim) in _CAT_CONFIGS.items()
        ],
        golden_dtype="float32",  # concatenation copies values; fp64 bit-identical
    )


from executorch.backends.webgpu.test.ops.test_gelu import (
    _det_input as _gelu_det_input,
    GeluModule,
    N as _GELU_N,
)


def _gelu_full_range(_shape) -> torch.Tensor:
    # Reuse the deterministic linspace(-6, 6) spanning negatives/zero/positives.
    return _gelu_det_input()


@register_op_test("gelu")
def _gelu_suite() -> WebGPUTestSuite:
    # erf ("none") is the Florence-2/BART + PyTorch default; tanh is the approx.
    return WebGPUTestSuite(
        module_factory=lambda approximate: GeluModule(approximate),
        cases=[
            Case(name="erf_vec", construct={"approximate": "none"}, inputs=((M1,),)),
            Case(name="erf_mat", construct={"approximate": "none"}, inputs=((M1, M2),)),
            Case(
                name="erf_rank3",
                construct={"approximate": "none"},
                inputs=((S1, M1, M2),),
            ),
            Case(
                name="tanh_mat", construct={"approximate": "tanh"}, inputs=((M1, M2),)
            ),
            Case(
                name="erf_range",
                construct={"approximate": "none"},
                inputs=(InputSpec(shape=(_GELU_N,), gen=_gelu_full_range),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_layer_norm import (
    _ramp as _ln_ramp,
    make_layer_norm,
)


@register_op_test("layer_norm")
def _layer_norm_suite() -> WebGPUTestSuite:
    # LayerNorm over the last dim (BART + DaViT); affine + no-affine, widths
    # below/equal/above the 64-wide workgroup reduction.
    return WebGPUTestSuite(
        module_factory=make_layer_norm,
        cases=[
            Case(
                name="affine_mat",
                construct={"normalized_shape": 128},
                inputs=((4, 128),),
            ),
            Case(
                name="affine_rank3",
                construct={"normalized_shape": 768},
                inputs=((1, 16, 768),),
            ),
            Case(
                name="no_affine",
                construct={"normalized_shape": 128, "affine": False},
                inputs=((4, 128),),
            ),
            Case(
                name="width_lt_wg",
                construct={"normalized_shape": 32},
                inputs=((8, 32),),
            ),
            Case(
                name="width_gt_wg",
                construct={"normalized_shape": 132},
                inputs=((4, 132),),
            ),
            Case(
                name="bart_hidden",
                construct={"normalized_shape": 1024},
                inputs=(InputSpec(shape=(1, 8, 1024), gen=_ln_ramp),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_linear_fp32 import (
    _ramp as _lin_ramp,
    make_linear,
)


@register_op_test("linear_fp32")
def _linear_fp32_suite() -> WebGPUTestSuite:
    # fp32 linear (BART + DaViT projections); bias + no-bias, and shapes whose
    # M*N exceeds the 65535 1D ceiling to exercise the 2D-dispatch spill.
    return WebGPUTestSuite(
        module_factory=make_linear,
        cases=[
            Case(
                name="bias_mat",
                construct={"in_features": 64, "out_features": 32},
                inputs=((4, 64),),
            ),
            Case(
                name="no_bias",
                construct={"in_features": 64, "out_features": 32, "bias": False},
                inputs=((4, 64),),
            ),
            Case(
                name="rank3",
                construct={"in_features": 768, "out_features": 768},
                inputs=(InputSpec(shape=(1, 16, 768), gen=_lin_ramp),),
            ),
            Case(
                name="tall_m",
                construct={"in_features": 128, "out_features": 64},
                inputs=((256, 128),),
            ),
            Case(
                name="bart_proj",
                construct={"in_features": 1024, "out_features": 1024},
                inputs=(InputSpec(shape=(1, 8, 1024), gen=_lin_ramp),),
            ),
            Case(
                name="odd_k",
                construct={"in_features": 63, "out_features": 32},
                inputs=((4, 63),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_conv2d import _chw_ramp, make_conv


@register_op_test("conv2d")
def _conv2d_suite() -> WebGPUTestSuite:
    # DaViT patch-embed / downsample convs + conv_transpose2d (same registration,
    # folded by the `transposed` arg). NCHW fp32. Routing coverage (all vs the
    # same fp64 golden): patch_embed/conv3x3_pad1/strided/gemm_batched are
    # groups==1 → im2col tiled GEMM (gemm_batched pins the B>1 output write);
    # grouped_vec4 (groups=2, icpg=4) → direct vec4 kernel; depthwise (groups=8,
    # icpg=1) → direct scalar; transpose2x → conv_transpose2d.
    return WebGPUTestSuite(
        module_factory=make_conv,
        cases=[
            Case(
                name="conv3x3_pad1",
                construct={"in_ch": 8, "out_ch": 16, "kernel": 3, "padding": 1},
                inputs=(InputSpec(shape=(1, 8, 16, 16), gen=_chw_ramp),),
            ),
            Case(
                name="patch_embed",
                construct={"in_ch": 3, "out_ch": 64, "kernel": 16, "stride": 16},
                inputs=(InputSpec(shape=(1, 3, 32, 32), gen=_chw_ramp),),
            ),
            Case(
                name="strided",
                construct={
                    "in_ch": 3,
                    "out_ch": 8,
                    "kernel": 3,
                    "stride": 2,
                    "padding": 1,
                },
                inputs=(InputSpec(shape=(1, 3, 16, 16), gen=_chw_ramp),),
            ),
            Case(
                name="depthwise",
                construct={
                    "in_ch": 8,
                    "out_ch": 8,
                    "kernel": 3,
                    "padding": 1,
                    "groups": 8,
                },
                inputs=(InputSpec(shape=(1, 8, 8, 8), gen=_chw_ramp),),
            ),
            Case(
                name="grouped_vec4",
                construct={
                    "in_ch": 8,
                    "out_ch": 8,
                    "kernel": 3,
                    "padding": 1,
                    "groups": 2,
                },
                inputs=(InputSpec(shape=(1, 8, 8, 8), gen=_chw_ramp),),
            ),
            Case(
                name="gemm_batched",
                construct={"in_ch": 8, "out_ch": 16, "kernel": 3, "padding": 1},
                inputs=(InputSpec(shape=(2, 8, 16, 16), gen=_chw_ramp),),
            ),
            Case(
                name="transpose2x",
                construct={
                    "in_ch": 4,
                    "out_ch": 4,
                    "kernel": 2,
                    "stride": 2,
                    "transposed": True,
                },
                inputs=(InputSpec(shape=(1, 4, 4, 4), gen=_chw_ramp),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_et_vk_sdpa import SdpaModule


def _sdpa_randn(shape):
    g = torch.Generator().manual_seed(sum(int(x) for x in shape))
    return torch.randn(*shape, generator=g)


def _sdpa_mask(b, h, sq, skv):
    g = torch.Generator().manual_seed(7)
    return torch.randn(b, h, sq, skv, generator=g).clamp(-1.0, 0.0)


@register_op_test("et_vk_sdpa")
def _et_vk_sdpa_suite() -> WebGPUTestSuite:
    # Non-causal fused attention (Florence-2 vision + BART, via the et_vk source
    # transform). Covers self-attn, an asymmetric S_q != S_kv (cross-attn) case,
    # an additive mask (BART), and D=128 (Voxtral/DaViT). The QK dispatch is
    # occupancy-routed: chattn_davit (num_rows = B*H*S_q = 256 < the 4096 floor)
    # exercises the per-entry QK kernel; selfattn_siglip (num_rows = 6912) is the
    # per-row guard. Both branches must match the same fp32 golden.
    def qkv(b, h, sq, skv, d):
        return (
            InputSpec(shape=(b, h, sq, d), gen=_sdpa_randn),
            InputSpec(shape=(b, h, skv, d), gen=_sdpa_randn),
            InputSpec(shape=(b, h, skv, d), gen=_sdpa_randn),
        )

    return WebGPUTestSuite(
        module_factory=lambda mask=None: SdpaModule(mask),
        cases=[
            Case(name="selfattn_small", inputs=qkv(1, 4, 8, 8, 16)),
            Case(name="selfattn_siglip", inputs=qkv(1, 12, 576, 576, 64)),
            Case(name="asym_qpool", inputs=qkv(1, 8, 4, 16, 16)),
            Case(
                name="masked_bart",
                construct={"mask": _sdpa_mask(1, 4, 8, 8)},
                inputs=qkv(1, 4, 8, 8, 16),
            ),
            Case(name="d128_voxtral", inputs=qkv(1, 4, 6, 6, 128)),
            Case(name="chattn_davit", inputs=qkv(1, 8, 32, 256, 64)),
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_embedding import (
    _det_weight as _emb_det_weight,
    EmbeddingModule,
)


def _emb_idx_small(_shape):
    return torch.tensor([0, 3, 15, 7], dtype=torch.long)


def _emb_idx_bart(_shape):
    return torch.tensor([[1, 5, 1023, 0, 42]], dtype=torch.long)


@register_op_test("embedding")
def _embedding_suite() -> WebGPUTestSuite:
    # fp32 token/pos embedding lookup (BART). int32 indices via the op-test
    # framework's int-input path.
    return WebGPUTestSuite(
        module_factory=lambda num_embeddings, embed_dim: EmbeddingModule(
            _emb_det_weight(num_embeddings, embed_dim)
        ),
        cases=[
            Case(
                name="small",
                construct={"num_embeddings": 16, "embed_dim": 8},
                inputs=(InputSpec(shape=(4,), gen=_emb_idx_small),),
            ),
            Case(
                name="bart_tok",
                construct={"num_embeddings": 1024, "embed_dim": 768},
                inputs=(InputSpec(shape=(1, 5), gen=_emb_idx_bart),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_addmm import (
    _randn as _addmm_randn,
    AddmmModule,
)


@register_op_test("addmm")
def _addmm_suite() -> WebGPUTestSuite:
    return WebGPUTestSuite(
        module_factory=lambda n: AddmmModule(n),
        cases=[
            Case(
                name="small",
                construct={"n": 32},
                inputs=(
                    InputSpec(shape=(4, 16), gen=_addmm_randn),
                    InputSpec(shape=(16, 32), gen=_addmm_randn),
                ),
            ),
            Case(
                name="bart",
                construct={"n": 768},
                inputs=(
                    InputSpec(shape=(16, 768), gen=_addmm_randn),
                    InputSpec(shape=(768, 768), gen=_addmm_randn),
                ),
            ),
            Case(
                name="odd_k",
                construct={"n": 32},
                inputs=(
                    InputSpec(shape=(4, 15), gen=_addmm_randn),
                    InputSpec(shape=(15, 32), gen=_addmm_randn),
                ),
            ),
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_constant_pad_nd import (
    _randn as _pad_randn,
    PadModule,
)


@register_op_test("constant_pad_nd")
def _constant_pad_nd_suite() -> WebGPUTestSuite:
    return WebGPUTestSuite(
        module_factory=lambda pad: PadModule(pad),
        cases=[
            Case(
                name="last2",
                construct={"pad": [1, 2]},
                inputs=(InputSpec(shape=(3, 8), gen=_pad_randn),),
            ),
            Case(
                name="rank3",
                construct={"pad": [1, 1, 2, 0]},
                inputs=(InputSpec(shape=(2, 4, 8), gen=_pad_randn),),
            ),
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_upsample_nearest2d import (
    _det_input as _upsample_det_input,
    UpsampleNearest2dModule,
)


@register_op_test("upsample_nearest2d")
def _upsample_nearest2d_suite() -> WebGPUTestSuite:
    # SAM2 FPN 2x upsample chain (36->72->144) + a cheap tiny eyeball case.
    return WebGPUTestSuite(
        module_factory=lambda scale_factor: UpsampleNearest2dModule(scale_factor),
        cases=[
            Case(
                name="fpn_36_72",
                construct={"scale_factor": 2.0},
                inputs=(InputSpec(shape=(1, 8, 36, 36), gen=_upsample_det_input),),
            ),
            Case(
                name="fpn_72_144",
                construct={"scale_factor": 2.0},
                inputs=(InputSpec(shape=(1, 8, 72, 72), gen=_upsample_det_input),),
            ),
            Case(
                name="tiny",
                construct={"scale_factor": 2.0},
                inputs=(InputSpec(shape=(1, 4, 5, 7), gen=_upsample_det_input),),
            ),
            Case(
                # Non-integer, non-2x ratio (5->8): floor(oh*5/8) and the
                # half-pixel-center formula round((oh+0.5)*5/8-0.5) diverge at
                # oh=3,6 (1 vs 2, 3 vs 4) — locks in the legacy "nearest"
                # formula (see upsample_nearest2d.wgsl) against a genuinely
                # discriminating ratio, not just the 2x cases above where both
                # formulas happen to agree.
                name="non_2x_ratio",
                construct={"scale_factor": 1.6},
                inputs=(InputSpec(shape=(1, 2, 5, 5), gen=_upsample_det_input),),
            ),
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_leaky_relu import LeakyReluModule


@register_op_test("leaky_relu")
def _leaky_relu_suite() -> WebGPUTestSuite:
    # Real-ESRGAN SRVGGNetCompact body activation. The det input spans negatives,
    # exercising the negative_slope branch; a 4D and a 2D case.
    return WebGPUTestSuite(
        module_factory=lambda negative_slope: LeakyReluModule(negative_slope),
        cases=[
            Case(
                name="default_slope",
                construct={"negative_slope": 0.01},
                inputs=(InputSpec(shape=(1, 16, 8, 8), gen=_upsample_det_input),),
            ),
            Case(
                name="slope_0_2",
                construct={"negative_slope": 0.2},
                inputs=(InputSpec(shape=(3, 32), gen=_upsample_det_input),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_upsample_bilinear2d import (
    UpsampleBilinear2dModule,
)


@register_op_test("upsample_bilinear2d")
def _upsample_bilinear2d_suite() -> WebGPUTestSuite:
    # DPT/Depth-Anything bilinear resize head. Both align_corners branches +
    # a non-integer ratio (5->8) that discriminates the two source-index
    # formulas (see upsample_bilinear2d.wgsl).
    return WebGPUTestSuite(
        module_factory=lambda scale_factor, align_corners: UpsampleBilinear2dModule(
            scale_factor, align_corners
        ),
        cases=[
            Case(
                name="af_false_2x",
                construct={"scale_factor": 2.0, "align_corners": False},
                inputs=(InputSpec(shape=(1, 8, 36, 36), gen=_upsample_det_input),),
            ),
            Case(
                name="af_false_non_2x",
                construct={"scale_factor": 1.6, "align_corners": False},
                inputs=(InputSpec(shape=(1, 2, 5, 5), gen=_upsample_det_input),),
            ),
            Case(
                name="af_false_tiny",
                construct={"scale_factor": 2.0, "align_corners": False},
                inputs=(InputSpec(shape=(1, 4, 5, 7), gen=_upsample_det_input),),
            ),
            Case(
                name="af_true_2x",
                construct={"scale_factor": 2.0, "align_corners": True},
                inputs=(InputSpec(shape=(1, 4, 7, 7), gen=_upsample_det_input),),
            ),
            Case(
                name="af_true_non_2x",
                construct={"scale_factor": 1.6, "align_corners": True},
                inputs=(InputSpec(shape=(1, 2, 5, 5), gen=_upsample_det_input),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_batch_norm import (
    _det_input as _bn_det_input,
    BatchNorm2dModule,
)


@register_op_test("batch_norm")
def _batch_norm_suite() -> WebGPUTestSuite:
    # MODNet decoder / CNN-backbone inference batch norm (eval -> the no-training
    # variant). Covers affine + non-affine (optional weight/bias) and an odd H*W.
    # Only the `out` ValueList entry is compared (out_index 0).
    return WebGPUTestSuite(
        module_factory=lambda num_features, affine: BatchNorm2dModule(
            num_features, affine
        ).eval(),
        cases=[
            Case(
                name="affine_c8",
                construct={"num_features": 8, "affine": True},
                inputs=(InputSpec(shape=(1, 8, 12, 12), gen=_bn_det_input),),
            ),
            Case(
                name="no_affine_c8",
                construct={"num_features": 8, "affine": False},
                inputs=(InputSpec(shape=(1, 8, 12, 12), gen=_bn_det_input),),
            ),
            Case(
                name="c16_odd",
                construct={"num_features": 16, "affine": True},
                inputs=(InputSpec(shape=(1, 16, 5, 7), gen=_bn_det_input),),
            ),
        ],
        atol=1e-3,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_split_with_sizes_copy import (
    _det_input as _split_det_input,
    SplitWithSizesModule,
)


@register_op_test("split_with_sizes_copy")
def _split_with_sizes_copy_suite() -> WebGPUTestSuite:
    # YOLO Detect-head split of concatenated predictions. Multi-output: the
    # framework compares chunk 0 (out_index 0) while each case runs all N
    # per-chunk dispatches. Covers a 3-way channel split, a dim-0 split, and
    # a last-dim split. copy is bit-exact -> float32 golden.
    return WebGPUTestSuite(
        module_factory=lambda sizes, dim, out_order=None: SplitWithSizesModule(
            sizes, dim, out_order
        ),
        cases=[
            Case(
                name="three_dim1",
                construct={"sizes": [2, 3, 3], "dim": 1},
                inputs=(InputSpec(shape=(1, 8, 4, 4), gen=_split_det_input),),
            ),
            Case(
                name="two_dim0",
                construct={"sizes": [3, 2], "dim": 0},
                inputs=(InputSpec(shape=(5, 4), gen=_split_det_input),),
            ),
            Case(
                name="dim_last",
                construct={"sizes": [4, 4], "dim": -1},
                inputs=(InputSpec(shape=(2, 8), gen=_split_det_input),),
            ),
            # Reorder so chunk 1 (running offset > 0) is output 0 -> verifies the
            # per-chunk start accumulation (out_index 0 is all the framework checks).
            Case(
                name="offset_chunk1_first",
                construct={"sizes": [2, 3, 3], "dim": 1, "out_order": [1, 0, 2]},
                inputs=(InputSpec(shape=(1, 8, 4, 4), gen=_split_det_input),),
            ),
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_max_pool2d import (
    _det_input as _maxpool_det_input,
    MaxPool2dModule,
)


@register_op_test("max_pool2d")
def _max_pool2d_suite() -> WebGPUTestSuite:
    # SAM2 Hiera q_pool (native shape) + real Hiera channel count (ch768) +
    # a padding case + a tiny eyeball case.
    return WebGPUTestSuite(
        module_factory=lambda kernel_size, stride, padding: MaxPool2dModule(
            kernel_size, stride, padding
        ),
        cases=[
            Case(
                name="q_pool",
                construct={"kernel_size": 2, "stride": 2, "padding": 0},
                inputs=(InputSpec(shape=(1, 8, 12, 12), gen=_maxpool_det_input),),
            ),
            Case(
                name="ch768",
                construct={"kernel_size": 2, "stride": 2, "padding": 0},
                inputs=(InputSpec(shape=(1, 768, 14, 14), gen=_maxpool_det_input),),
            ),
            Case(
                name="pad1",
                construct={"kernel_size": 3, "stride": 2, "padding": 1},
                inputs=(InputSpec(shape=(1, 8, 7, 7), gen=_maxpool_det_input),),
            ),
            Case(
                name="tiny",
                construct={"kernel_size": 2, "stride": 2, "padding": 0},
                inputs=(InputSpec(shape=(1, 4, 5, 5), gen=_maxpool_det_input),),
            ),
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_relu import (
    _det_input as _relu_det_input,
    ReluModule,
)


@register_op_test("relu")
def _relu_suite() -> WebGPUTestSuite:
    # SAM2/SAM3 mask-decoder MLP path; tiny + a decoder_mlp-shaped case.
    return WebGPUTestSuite(
        module_factory=lambda: ReluModule(),
        cases=[
            Case(
                name="tiny",
                construct={},
                inputs=(InputSpec(shape=(1, 4, 8), gen=_relu_det_input),),
            ),
            Case(
                name="decoder_mlp",
                construct={},
                inputs=(InputSpec(shape=(1, 576, 768), gen=_relu_det_input),),
            ),
        ],
        atol=1e-4,
        rtol=1e-3,
    )


from executorch.backends.webgpu.test.ops.test_sub import (
    CONFIGS as _SUB_CONFIGS,
    SubModule,
)


@register_op_test("sub")
def _sub_suite() -> WebGPUTestSuite:
    # Full numeric coverage incl. the spatial broadcast + alpha (binary_sub.wgsl
    # over a TensorMeta UBO); fp64 golden. Mirrors _mul_suite. alpha is a
    # construct kwarg baked into the .pte, never a serialized input.
    return WebGPUTestSuite(
        module_factory=lambda: SubModule(),
        cases=[
            Case(name=name, inputs=(sa, sb)) for name, (sa, sb) in _SUB_CONFIGS.items()
        ],
    )


def _unary_g1_factory(torch_fn, gen):
    # A per-op no-param-activation suite (mat + rank3), reused across UNARY_G1.
    def _suite() -> WebGPUTestSuite:
        return WebGPUTestSuite(
            module_factory=lambda: UnaryModule(torch_fn),
            cases=[
                Case(name="mat", inputs=(InputSpec(shape=(M1, M2), gen=gen),)),
                Case(name="rank3", inputs=(InputSpec(shape=(S1, M1, M2), gen=gen),)),
            ],
        )

    return _suite


for _g1_op, (_g1_fn, _g1_gen) in UNARY_G1.items():
    register_op_test(_g1_op)(_unary_g1_factory(_g1_fn, _g1_gen))


@register_op_test("clamp")
def _clamp_suite() -> WebGPUTestSuite:
    # min_none exercises the None -> -inf substitution in clamp_impl.
    return WebGPUTestSuite(
        module_factory=lambda lo, hi: ClampModule(lo, hi),
        cases=[
            Case(
                name=n,
                construct={"lo": lo, "hi": hi},
                inputs=(InputSpec(shape=(M1, M2), gen=_unary_lin(-6.0, 6.0)),),
            )
            for n, (lo, hi) in CLAMP_CONFIGS.items()
        ],
    )


@register_op_test("hardtanh")
def _hardtanh_suite() -> WebGPUTestSuite:
    return WebGPUTestSuite(
        module_factory=lambda lo, hi: HardtanhModule(lo, hi),
        cases=[
            Case(
                name=n,
                construct={"lo": lo, "hi": hi},
                inputs=(InputSpec(shape=(M1, M2), gen=_unary_lin(-6.0, 6.0)),),
            )
            for n, (lo, hi) in HARDTANH_CONFIGS.items()
        ],
    )


@register_op_test("pow_scalar")
def _pow_scalar_suite() -> WebGPUTestSuite:
    # Positive-base configs (exponent baked). Plus a negative-base integer-exponent
    # case: pow(x, 2) == x*x. WGSL pow(neg) is undefined, so the shader special-cases
    # it (fixes F.normalize's q^2 in Swin-V2 cosine attention).
    return WebGPUTestSuite(
        module_factory=lambda exponent: PowScalarModule(exponent),
        cases=[
            Case(
                name=n,
                construct={"exponent": e},
                inputs=(InputSpec(shape=(M1, M2), gen=_unary_lin(0.1, 4.0)),),
            )
            for n, e in POW_SCALAR_CONFIGS.items()
        ]
        + [
            Case(
                name="neg_base_sq",
                construct={"exponent": 2.0},
                inputs=(InputSpec(shape=(M1, M2), gen=_unary_lin(-3.0, 3.0)),),
            )
        ],
    )


# int8 stress: values at the sign-extend edges (-128/127/0/-1), numel % 4 == 0.
_DEQUANT_EDGE = [-128, -127, -1, 0, 1, 63, 126, 127, -64, -50, 50, 100, 7, -8, 42, -42]


@register_op_test("quantize_per_tensor")
def _quantize_suite() -> WebGPUTestSuite:
    # Lone quantize (int8 output), byte-exact vs torch int8 -- a round-trip folds
    # to identity (ET cancels dq(q(x))). golden_dtype float32: golden = int8 eager.
    def case(name, shape, scale, zp):
        return Case(
            name=name,
            construct={"scale": scale, "zero_point": zp},
            inputs=(shape,),
        )

    # scale 0.05 -> inv_scale 20; x = k*0.025 lands on half-integer k*0.5 (ties),
    # where WGSL round() and torch (both ties-to-even) must still agree byte-exact.
    def _ties(shape):
        assert torch.Size(shape).numel() == 16, "ties input requires numel == 16"
        return (torch.arange(-8, 8, dtype=torch.float32) * 0.025).reshape(shape)

    return WebGPUTestSuite(
        module_factory=lambda scale, zero_point: QuantizeModule(scale, zero_point),
        cases=[
            case("basic", (4, 8), 0.05, 0),
            case("zp_nonzero", (2, 2, 4), 0.1, 5),
            case("small_scale", (16,), 0.02, -13),
            Case(
                name="ties",
                construct={"scale": 0.05, "zero_point": 0},
                inputs=(InputSpec(shape=(16,), gen=_ties),),
            ),
        ],
        golden_dtype="float32",
    )


# span the int8 sign-extend + requant-clamp edges (-128/127); numel % 4 == 0.
_Q8_A = [-128, 127, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
_Q8_B = [-128, 127, 0, 1, -1, 50, -50, 100, -100, 3, -3, 42, -42, 9, -9, 63]


@register_op_test("q8ta_add")
def _q8ta_add_suite() -> WebGPUTestSuite:
    # int8 add (baked int8 consts), byte-exact vs CPU eager. `alpha` case pins the
    # a + alpha*b term (the Vulkan glsl buffer path drops alpha).
    def case(name, **kw):
        return Case(name=name, construct={"a_vals": _Q8_A, "b_vals": _Q8_B, **kw})

    return WebGPUTestSuite(
        module_factory=lambda **kw: Q8taAddModule(**kw),
        cases=[
            case("basic"),
            case("alpha", alpha=2.0),
            case("nonzero_zp", a_zp=5, b_zp=-4, out_zp=7),
        ],
        golden_dtype="float32",
    )


@register_op_test("q8ta_relu")
def _q8ta_relu_suite() -> WebGPUTestSuite:
    # int8 relu (baked int8 const), byte-exact vs CPU eager. Inputs whose dequant
    # is negative are relu-clamped to 0 (pins max(x,0)); edges -128/127 covered.
    def case(name, **kw):
        return Case(name=name, construct={"x_vals": _Q8_A, **kw})

    return WebGPUTestSuite(
        module_factory=lambda **kw: Q8taReluModule(**kw),
        cases=[
            case("basic"),
            case("diff_qparams", output_scale=0.08, output_zp=5),
            case("nonzero_zp", input_zp=10, output_zp=-20),
        ],
        golden_dtype="float32",
    )


@register_op_test("q8ta_pixel_shuffle")
def _q8ta_pixel_shuffle_suite() -> WebGPUTestSuite:
    # int8 pixel_shuffle (baked int8 const), byte-exact vs CPU eager. Same-qparams
    # is a pure gather; diff_qparams exercises the dequant->requant rescale.
    def case(name, n_ch, h, w, **kw):
        n = n_ch * h * w  # [1, n_ch, h, w], n_ch = C*r*r
        vals = [((i % 251) - 125) for i in range(n)]  # spread across int8 range
        return Case(
            name=name, construct={"x_vals": vals, "shape": (1, n_ch, h, w), **kw}
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: Q8taPixelShuffleModule(**kw),
        cases=[
            case("basic", 4, 2, 2),  # r=2, C=1 -> [1,1,4,4]
            case("c2", 8, 2, 2),  # r=2, C=2 -> [1,2,4,4]
            case("r3", 9, 2, 2, upscale_factor=3),  # r=3, C=1 -> [1,1,6,6]
            case("diff_qparams", 4, 3, 3, output_scale=0.08, output_zp=5),
            case("nonzero_zp", 4, 2, 2, input_zp=10, output_zp=-20),
        ],
        golden_dtype="float32",
    )


@register_op_test("linear_qcs4w")
def _linear_qcs4w_suite() -> WebGPUTestSuite:
    # VulkanQuantizer weight-only 4-bit nn.Linear -> delegated et_vk.linear_qcs4w.
    # Golden = converted eager (fp32 per-channel fake-quant). K even (2 nibbles/
    # byte), N*ceil(K/2) % 4 == 0 (u32-packed weight). M==1 = decode/gemv shape.
    def case(name, m, k, n, **kw):
        return Case(
            name=name, construct={"k": k, "n": n, "m": m, **kw}, inputs=((m, k),)
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_qcs4w_linear_module(**kw),
        cases=[
            case("basic", 4, 32, 16),
            case("gemv", 1, 32, 16),  # M==1
            case("k64", 2, 64, 8),
            case("n32", 3, 32, 32),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("linear_q8ta_q8csw")
def _linear_q8ta_q8csw_suite() -> WebGPUTestSuite:
    # XNNPACK-static nn.Linear with output_activation=None -> delegated
    # quantize_per_tensor -> linear_q8ta_q8csw (fp32 out). Golden = converted
    # eager (fp32 fake-quant). All cases use bias=True: a bias-less TERMINAL
    # linear mis-fuses to an int8 output the schema (no output scale/zp) cannot
    # compute -> the handler fail-louds on it; bias keeps the output fp32. N is a
    # multiple of 4 (the AOT pads the quantized weight's N to a mult of 4).
    def case(name, m, k, n, **kw):
        return Case(
            name=name,
            construct={"k": k, "n": n, "m": m, "bias": True, **kw},
            inputs=((m, k),),
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_linear_q8ta_q8csw_module(**kw),
        cases=[
            case("basic", 4, 32, 16),
            case("gemv", 1, 32, 16),  # M==1
            case("k48", 2, 48, 8),  # different K, smaller N
            case("n32", 3, 32, 32),  # larger N
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("linear_dq8ca_q4gsw")
def _linear_dq8ca_q4gsw_suite() -> WebGPUTestSuite:
    # torchao Int8DynamicActivationIntxWeightConfig(int4, PerGroup) -> delegated
    # choose_qparams_affine (per-row act scale/zp) -> linear_dq8ca_q4gsw (fp32
    # out). Golden = converted eager (fp32 fake-quant). q4gsw needs K % gs == 0,
    # K % 8 == 0, N % 8 == 0.
    def case(name, m, k, n, **kw):
        return Case(
            name=name, construct={"k": k, "n": n, "m": m, **kw}, inputs=((m, k),)
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_linear_dq8ca_q4gsw_module(**kw),
        cases=[
            case("basic", 4, 64, 16),
            case("gemv", 1, 64, 16),  # M==1 decode
            case("k128", 2, 128, 8),  # more groups, smaller N
            case("n32", 3, 64, 32),  # larger N
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("q8ta_linear")
def _q8ta_linear_suite() -> WebGPUTestSuite:
    # XNNPACK-static-quantized nn.Linear -> delegated quantize->q8ta_linear->
    # dequantize (C0 + the new op). Golden = converted eager (fp32 fake-quant).
    def case(name, m, k, n, **kw):
        return Case(
            name=name, construct={"k": k, "n": n, "m": m, **kw}, inputs=((m, k),)
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_q8ta_linear_module(**kw),
        cases=[
            case("basic", 4, 16, 8),
            case("gemv", 1, 16, 8),  # M==1
            case("k32", 8, 32, 4),
            case("no_bias", 4, 16, 8, bias=False),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("q8ta_conv2d_pw")
def _q8ta_conv2d_pw_suite() -> WebGPUTestSuite:
    # Golden = XNNPACK-static-PT2E converted eager (fp32); W%4==0 for output pack.
    def case(name, ic, oc, h, w, n=1, **kw):
        return Case(
            name=name,
            construct={"ic": ic, "oc": oc, "h": h, "w": w, "n": n, **kw},
            inputs=((n, ic, h, w),),
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_q8ta_conv2d_pw_module(**kw),
        cases=[
            case("basic", 4, 8, 6, 8),
            case("ic8", 8, 4, 4, 4),
            case("no_bias", 4, 8, 6, 8, bias=False),
            case("batch2", 4, 8, 6, 8, n=2),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("q8ta_conv2d_dw")
def _q8ta_conv2d_dw_suite() -> WebGPUTestSuite:
    # Golden = XNNPACK-static-PT2E converted eager (fp32); depthwise, C%4==0,
    # W_out%4==0 for output packing.
    def case(name, c, k, h, w, n=1, **kw):
        return Case(
            name=name,
            construct={"c": c, "k": k, "h": h, "w": w, "n": n, **kw},
            inputs=((n, c, h, w),),
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_q8ta_conv2d_dw_module(**kw),
        cases=[
            case("k3", 8, 3, 8, 8),
            case("no_bias", 8, 3, 8, 8, bias=False),
            case("stride2", 8, 3, 8, 8, stride=2),
            case("dil2", 8, 3, 8, 8, dilation=2, padding=2),
            case("batch2", 8, 3, 8, 8, n=2),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("q8ta_conv2d")
def _q8ta_conv2d_suite() -> WebGPUTestSuite:
    # Golden = XNNPACK-static-PT2E converted eager (fp32); general (groups==1)
    # conv, full IC reduction. W_out%4==0 for output packing.
    def case(name, ic, oc, k, h, w, n=1, **kw):
        return Case(
            name=name,
            construct={"ic": ic, "oc": oc, "k": k, "h": h, "w": w, "n": n, **kw},
            inputs=((n, ic, h, w),),
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_q8ta_conv2d_module(**kw),
        cases=[
            case("k3", 8, 8, 3, 8, 8),
            case("oc_gt", 4, 8, 3, 8, 8),
            case("no_bias", 8, 8, 3, 8, 8, bias=False),
            case("stride2", 8, 8, 3, 8, 8, stride=2),
            case("dil2", 8, 8, 3, 8, 8, dilation=2, padding=2),
            case("oc6", 8, 6, 3, 8, 8),
            case("ic3", 3, 8, 3, 8, 8),
            case("asym", 8, 8, (3, 5), 8, 8, padding=(1, 2)),
            case("batch2", 8, 8, 3, 8, 8, n=2),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("q8ta_conv2d_transposed")
def _q8ta_conv2d_transposed_suite() -> WebGPUTestSuite:
    # Golden = XNNPACK-static-PT2E converted eager (fp32); transposed conv
    # (groups==1, dilation==1). W_out%4==0 for output packing.
    def case(name, ic, oc, k, h, w, n=1, **kw):
        return Case(
            name=name,
            construct={"ic": ic, "oc": oc, "k": k, "h": h, "w": w, "n": n, **kw},
            inputs=((n, ic, h, w),),
        )

    return WebGPUTestSuite(
        module_factory=lambda **kw: make_q8ta_conv2d_transposed_module(**kw),
        cases=[
            case("s2", 8, 8, 2, 8, 8, stride=2),
            case("no_bias", 8, 8, 2, 8, 8, stride=2, bias=False),
            case("k3", 8, 8, 3, 8, 8, stride=2, padding=1, output_padding=1),
            case("oc6", 8, 6, 2, 8, 8, stride=2),
            case("ic3", 3, 8, 3, 8, 8, stride=2, padding=1, output_padding=1),
            case("batch2", 8, 8, 2, 8, 8, stride=2, n=2),
            case("asym", 8, 8, (3, 2), 8, 8, stride=2),
        ],
        golden_dtype="float32",
        atol=1e-3,
        rtol=1e-3,
    )


@register_op_test("dequantize_per_tensor")
def _dequant_const_suite() -> WebGPUTestSuite:
    # Independent GPU dequantize check: baked int8 const vs torch dequant (breaks
    # the round-trip's compensating-bug blind spot). golden = CPU eager.
    return WebGPUTestSuite(
        module_factory=lambda scale, zero_point: DequantizeConstModule(
            scale, zero_point, _DEQUANT_EDGE
        ),
        cases=[
            Case(name="edges", construct={"scale": 0.05, "zero_point": 0}, inputs=()),
            Case(name="zp", construct={"scale": 0.1, "zero_point": 7}, inputs=()),
        ],
        golden_dtype="float32",
    )


@register_op_test("argmax")
def _argmax_suite() -> WebGPUTestSuite:
    # Last-dim argmax -> int64 index. randn puts the max at an INTERIOR index
    # (exercises the walk); the tie case pins the strict-`>` first-occurrence.
    return WebGPUTestSuite(
        module_factory=lambda: ArgmaxModule(),
        cases=[
            Case(name="2d", inputs=((M1, M2),)),
            Case(name="3d", inputs=((S, S1, S2),)),
            Case(name="tie", inputs=(InputSpec(shape=(3, 6), gen=argmax_tie_gen),)),
        ],
        golden_dtype="float32",
    )


@register_op_test("argmin")
def _argmin_suite() -> WebGPUTestSuite:
    # Last-dim argmin -> int64 index; randn interior min + strict-`<` tie case.
    return WebGPUTestSuite(
        module_factory=lambda: ArgminModule(),
        cases=[
            Case(name="2d", inputs=((M1, M2),)),
            Case(name="3d", inputs=((S, S1, S2),)),
            Case(name="tie", inputs=(InputSpec(shape=(3, 6), gen=argmin_tie_gen),)),
        ],
        golden_dtype="float32",
    )
