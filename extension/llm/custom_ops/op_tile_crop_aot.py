# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch

try:
    tile_crop = torch.ops.preprocess.tile_crop.default
    assert tile_crop is not None
except:
    libs = list(Path(__file__).parent.resolve().glob("*custom_ops_aot_lib.*"))
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    logging.info(f"Loading custom ops library: {libs[0]}")
    torch.ops.load_library(libs[0])
    tile_crop = torch.ops.preprocess.tile_crop.default
    assert tile_crop is not None

preprocess_ops_lib = torch.library.Library("preprocess", "IMPL")

MAX_NUM_TILES = 4


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
