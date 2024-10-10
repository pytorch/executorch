# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.library

namespace = "et_vk"
lib = torch.library.Library(namespace, "DEF")


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
