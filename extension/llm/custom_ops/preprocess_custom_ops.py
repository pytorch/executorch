# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from torch.library import impl, Library

preprocess_op_lib = Library("preprocess", "DEF")

# Register and define tile_crop and out variant.
preprocess_op_lib.define("tile_crop(Tensor input, int tile_size) -> Tensor")

# Keep this in sync with model config.
MAX_NUM_TILES = 4


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
    # Returned tensor is of size [n, 3, 224, 224], where n = number of tiles.
    # Use an unbacked symint to create an upper-bounded dynamic shape output.
    # Otherwise, output is set to a static shape, and we can only output
    # tensors of shape [MAX_NUM_TILES, 3, 224, 224].
    ctx = torch._custom_ops.get_ctx()
    s0 = ctx.create_unbacked_symint()
    torch._constrain_as_size(s0, 0, MAX_NUM_TILES)
    return torch.empty([s0, output.size(0), tile_size, tile_size])
