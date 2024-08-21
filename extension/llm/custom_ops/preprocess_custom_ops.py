# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import List

import torch

from torch.library import impl, Library

preprocess_op_lib = Library("preprocess", "DEF")

# Register and define pad and out variant.
# Note: pad doesn't require an explicit meta kernel because
# CompositeExplicitAutograd automatically registers the implementation to meta,
# and meta kernels do not go through functionalization. The implementation
# does not export due to issues during functionalization.
# See: https://github.com/pytorch/pytorch/issues/120288
preprocess_op_lib.define("pad(Tensor image, SymInt[] padding) -> Tensor")


@impl(preprocess_op_lib, "pad", dispatch_key="CompositeExplicitAutograd")
def pad_impl(
    image: torch.Tensor,
    padding: List[int],
) -> torch.Tensor:
    output = torch.empty(
        [image.shape[0], image.shape[1] + padding[3], image.shape[2] + padding[1]],
        dtype=image.dtype,
        device=image.device,
        requires_grad=False,
    )
    output = torch.fill(output, 0)
    output.narrow(1, 0, image.shape[1]).narrow(2, 0, image.shape[2]).copy_(image)
    return output


preprocess_op_lib.define(
    "pad.out(Tensor image, SymInt[] padding, *, Tensor(a!) out) -> Tensor(a!)"
)


@impl(preprocess_op_lib, "pad.out", dispatch_key="CompositeExplicitAutograd")
def pad_out_impl(
    image: torch.Tensor,
    padding: List[int],
    out: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        [image.shape[0], image.shape[1] + padding[3], image.shape[2] + padding[1]],
        dtype=image.dtype,
        device=image.device,
        requires_grad=False,
    )
    out = torch.fill(out, 0)
    out.narrow(1, 0, image.shape[1]).narrow(2, 0, image.shape[2]).copy_(image)
    return out


# Register and define tile_crop and out variant.
preprocess_op_lib.define("tile_crop(Tensor input, int tile_size) -> Tensor")


@impl(preprocess_op_lib, "tile_crop", dispatch_key="CompositeExplicitAutograd")
def tile_crop_impl(input: torch.Tensor, tile_size: int) -> torch.Tensor:
    c = input.shape[0]
    h = input.shape[1]
    w = input.shape[2]
    tiles_height = h // tile_size
    tiles_width = w // tile_size
    tile_cropped = input.view(c, tiles_height, tile_size, tiles_width, tile_size)
    transposed = tile_cropped.permute(1, 3, 0, 2, 4)
    tiles = transposed.contiguous().view(
        tiles_height * tiles_width, c, tile_size, tile_size
    )
    return tiles


preprocess_op_lib.define(
    "tile_crop.out(Tensor input, int tile_size, *, Tensor(a!) out) -> Tensor(a!)"
)


@impl(preprocess_op_lib, "tile_crop.out", dispatch_key="CompositeExplicitAutograd")
def tile_crop_out_impl(
    input: torch.Tensor, tile_size: int, out: torch.Tensor
) -> torch.Tensor:
    out = input.clone()
    c = out.shape[0]
    h = out.shape[1]
    w = out.shape[2]
    tiles_height = h // tile_size
    tiles_width = w // tile_size
    out = out.view(c, tiles_height, tile_size, tiles_width, tile_size)
    out = out.permute(1, 3, 0, 2, 4)
    out = out.contiguous().view(tiles_height * tiles_width, c, tile_size, tile_size)
    return out


# Register meta kernel to prevent export tracing into the tile_crop impl.
@torch.library.register_fake("preprocess::tile_crop")
def tile_crop(output: torch.Tensor, tile_size: int) -> torch.Tensor:
    # Returned tensor is of size [n, 3, 224, 224], where n is the number of tiles.
    # We should export with n = max_num_tiles. Set 50 for now.
    return torch.empty([50, output.size(0), 224, 224])
