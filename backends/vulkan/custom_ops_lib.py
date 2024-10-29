# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.library

namespace = "et_vk"
lib = torch.library.Library(namespace, "DEF")

#############
## prepack ##
#############


def prepack_impl(x: torch.Tensor):
    return x


name = "prepack"
lib.define(f"{name}(Tensor x) -> Tensor")
lib.impl(name, prepack_impl, "CompositeExplicitAutograd")
prepack_op = getattr(getattr(torch.ops, namespace), name)

#####################
## conv_with_clamp ##
#####################


def conv_with_clamp_impl(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    output_min=-float("inf"),
    output_max=float("inf"),
):
    return torch.clamp(
        torch.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ),
        output_min,
        output_max,
    )


name = "conv_with_clamp"
lib.define(
    f"{name}(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max) -> Tensor"
)
lib.impl(name, conv_with_clamp_impl, "CompositeExplicitAutograd")
conv_with_clamp_op = getattr(getattr(torch.ops, namespace), name)

#########################
## conv_with_clamp.out ##
#########################


def conv_with_clamp_out_impl(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    output_min=-float("inf"),
    output_max=float("inf"),
    out=None,
):
    out = conv_with_clamp_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_min,
        output_max,
    )
    return out


name = "conv_with_clamp.out"
lib.define(
    f"{name}(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.impl(name, conv_with_clamp_out_impl, "CompositeExplicitAutograd")

#################
## grid_priors ##
#################


# The dimension of x should be larger than 1
def grid_priors_impl(
    x,
    stride,
    offset,
):
    height, width = x.shape[-2:]
    # Need to specify device of torch.arange to avoid executorch exporting error
    shift_x = (torch.arange(0, width, device=x.device) + offset) * stride
    shift_y = (torch.arange(0, height, device=x.device) + offset) * stride
    # Need to specify indexing parameter ('ij' is the default value) to avoid executorch exporting error
    shift_xx, shift_yy = torch.meshgrid([shift_y, shift_x], indexing="ij")
    shift_xx = shift_xx.reshape(-1)
    shift_yy = shift_yy.reshape(-1)
    shifts = torch.stack((shift_yy, shift_xx), dim=-1)
    return shifts


name = "grid_priors"
lib.define(f"{name}(Tensor self, int stride, float offset) -> Tensor")
lib.impl(name, grid_priors_impl, "CompositeExplicitAutograd")
grid_priors_op = getattr(getattr(torch.ops, namespace), name)


# When lowering to executorch, ops are converted from default to out variant. Hence, custom ops define both variants.
def grid_priors_out_impl(
    x,
    stride,
    offset,
    out,
):
    out = grid_priors_impl(x, stride, offset)
    return out


name = "grid_priors_out"
lib.define(
    f"{name}(Tensor self, int stride, float offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.impl(name, grid_priors_out_impl, "CompositeExplicitAutograd")

########################
## linear_weight_int4 ##
########################


def linear_weight_int4_impl(
    x: torch.Tensor,
    weights_4x8: torch.Tensor,
    groupsize: int,
    scales_and_zeros: torch.Tensor,
    inner_k_tiles: int,
):
    original_x_size = x.size()
    out_features = weights_4x8.size(0)
    x = x.reshape(-1, original_x_size[-1])
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
        weights_4x8, inner_k_tiles
    )
    out = torch.ops.aten._weight_int4pack_mm(
        x, weight_int4pack, groupsize, scales_and_zeros
    )
    out_shape = original_x_size[:-1] + (out_features,)
    return out.reshape(out_shape)


name = "linear_weight_int4"
lib.define(
    f"{name}(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros, int inner_k_tiles) -> Tensor"
)
lib.impl(name, linear_weight_int4_impl, "CompositeExplicitAutograd")
linear_weight_int4_op = getattr(getattr(torch.ops, namespace), name)

######################
## apply_rotary_emb ##
######################


# Note that this implementation is copied from executorch.examples.models.llama.rope
# but it is copied here to avoid introducing a dependency on the llama code.
def apply_rotary_emb_impl(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
):
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        freqs_cis_ndim = freqs_cis.ndim
        if freqs_cis_ndim == 3:
            # freqs_cis: (seq_len, n_heads, head_dim // 2)
            assert freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1])
            shape = [
                d if (i == ndim - 3 or i == ndim - 2 or i == ndim - 1) else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            # freqs_cis: (seq_len, head_dim // 2)
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(shape)

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


name = "apply_rotary_emb"
lib.define(
    f"{name}(Tensor xq, Tensor xk, Tensor freqs_cos, Tensor freqs_sin) -> (Tensor, Tensor)"
)
lib.impl(name, apply_rotary_emb_impl, "CompositeExplicitAutograd")
apply_rotary_emb_op = getattr(getattr(torch.ops, namespace), name)
