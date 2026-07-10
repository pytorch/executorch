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
from executorch.backends.webgpu.test.ops.test_cat import (
    CatModule,
    CONFIGS as _CAT_CONFIGS,
)
from executorch.backends.webgpu.test.ops.test_mul import (
    CONFIGS as _MUL_CONFIGS,
    MulModule,
)
from executorch.backends.webgpu.test.ops.test_permute import (
    CONFIGS as _PERMUTE_CONFIGS,
    PermuteModule,
)
from executorch.backends.webgpu.test.ops.test_rms_norm import (
    _CASES,
    _linspace_weight,
    _ramp,
    RmsNormModule,
)
from executorch.backends.webgpu.test.ops.test_select import (
    CONFIGS as _SELECT_CONFIGS,
    SelectModule,
)
from executorch.backends.webgpu.test.ops.test_sigmoid import (
    _det_input as _sigmoid_det_input,
    N as _SIGMOID_N,
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


@register_op_test("view_copy")
def _view_copy_suite() -> WebGPUTestSuite:
    return _fn_config_suite(ViewModule, _VIEW_CONFIGS)


@register_op_test("select")
def _select_suite() -> WebGPUTestSuite:
    return _fn_config_suite(SelectModule, _SELECT_CONFIGS)


def _sigmoid_full_range(_shape) -> torch.Tensor:
    # Reuses the monolith's saturation-tail input (linspace(-12, 12)).
    return _sigmoid_det_input()


@register_op_test("sigmoid")
def _sigmoid_suite() -> WebGPUTestSuite:
    # sigmoid has no CONFIGS table; cover unary shapes directly (tol 1e-4).
    return WebGPUTestSuite(
        module_factory=lambda: SigmoidModule(),
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
            Case(name="tanh_mat", construct={"approximate": "tanh"}, inputs=((M1, M2),)),
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
            Case(name="affine_mat", construct={"normalized_shape": 128}, inputs=((4, 128),)),
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
                name="width_lt_wg", construct={"normalized_shape": 32}, inputs=((8, 32),)
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
from executorch.backends.webgpu.test.ops.test_conv2d import (
    _chw_ramp,
    make_conv,
)


@register_op_test("conv2d")
def _conv2d_suite() -> WebGPUTestSuite:
    # DaViT patch-embed / downsample convs + conv_transpose2d (same registration,
    # folded by the `transposed` arg). NCHW fp32.
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
from executorch.backends.webgpu.test.ops.test_et_vk_sdpa import (
    SdpaModule,
)


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
    # an additive mask (BART), and D=128 (Voxtral/DaViT) through the vec4 kernels.
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
        ],
        golden_dtype="float32",
        atol=1e-4,
        rtol=1e-3,
    )
from executorch.backends.webgpu.test.ops.test_embedding import (
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
            num_embeddings, embed_dim
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
            Case(name="small", construct={"n": 32}, inputs=(InputSpec(shape=(4, 16), gen=_addmm_randn), InputSpec(shape=(16, 32), gen=_addmm_randn))),
            Case(name="bart", construct={"n": 768}, inputs=(InputSpec(shape=(16, 768), gen=_addmm_randn), InputSpec(shape=(768, 768), gen=_addmm_randn))),
            Case(name="odd_k", construct={"n": 32}, inputs=(InputSpec(shape=(4, 15), gen=_addmm_randn), InputSpec(shape=(15, 32), gen=_addmm_randn))),
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
            Case(name="last2", construct={"pad": [1, 2]}, inputs=(InputSpec(shape=(3, 8), gen=_pad_randn),)),
            Case(name="rank3", construct={"pad": [1, 1, 2, 0]}, inputs=(InputSpec(shape=(2, 4, 8), gen=_pad_randn),)),
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
