# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from typing import List, Optional

import torch

from executorch.examples.models.model_base import EagerModelBase
from torch.export import Dim
from torch.library import impl, Library
from torchvision.transforms import v2
from torchvision.transforms._functional_tensor import resize

preprocess_op_lib = Library("preprocess", "DEF")

# Register and define pad and out variant.
preprocess_op_lib.define("pad(Tensor image, Tensor canvas) -> Tensor")


@impl(preprocess_op_lib, "pad", dispatch_key="CompositeExplicitAutograd")
def pad_impl(image: torch.Tensor, canvas: torch.Tensor) -> torch.Tensor:
    canvas.narrow(1, 0, image.shape[1]).narrow(2, 0, image.shape[2]).copy_(image)
    return canvas


preprocess_op_lib.define(
    "pad.out(Tensor image, Tensor canvas, *, Tensor(a!) out) -> Tensor(a!)"
)


@impl(preprocess_op_lib, "pad.out", dispatch_key="CompositeImplicitAutograd")
def pad_out_impl(
    image: torch.Tensor, canvas: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    out.narrow(1, 0, image.shape[1]).narrow(2, 0, image.shape[2]).copy_(image)
    return out


# Register and define reshape and out variant.
preprocess_op_lib.define("reshape(Tensor input, int c, int tile_size) -> Tensor")


@impl(preprocess_op_lib, "reshape", dispatch_key="CompositeExplicitAutograd")
def reshape_impl(input: torch.Tensor, c: int, tile_size: int) -> torch.Tensor:
    h = input.shape[1]
    w = input.shape[2]
    tiles_height = h // tile_size
    tiles_width = w // tile_size
    reshaped = input.view(c, tiles_height, tile_size, tiles_width, tile_size)
    transposed = reshaped.permute(1, 3, 0, 2, 4)
    tiles = transposed.contiguous().view(
        tiles_height * tiles_width, c, tile_size, tile_size
    )
    return tiles


preprocess_op_lib.define(
    "reshape.out(Tensor input, int c, int tile_size, *, Tensor(a!) out) -> Tensor(a!)"
)


@impl(preprocess_op_lib, "reshape.out", dispatch_key="CompositeExplicitAutograd")
def reshape_out_impl(
    input: torch.Tensor, c: int, tile_size: int, out: torch.Tensor
) -> torch.Tensor:
    out = input.clone()
    h = out.shape[1]
    w = out.shape[2]
    tiles_height = h // tile_size
    tiles_width = w // tile_size
    out = out.view(c, tiles_height, tile_size, tiles_width, tile_size)
    out = out.permute(1, 3, 0, 2, 4)
    out = out.contiguous().view(tiles_height * tiles_width, c, tile_size, tile_size)
    return out


# Register fake to prevent export tracing into the reshape impl.
@torch.library.register_fake("preprocess::reshape")
def reshape(output: torch.Tensor, c: int, tile_size: int) -> torch.Tensor:
    return torch.empty_like(output)


@dataclass
class PreprocessConfig:
    tile_size: int
    channels: int
    image_mean: List[float]
    image_std: List[float]
    resample: str
    normalize: bool


class Preprocess(torch.nn.Module):
    def __init__(self, config: Optional[PreprocessConfig] = None):
        super().__init__()
        if config is None:
            self.config = PreprocessConfig(
                tile_size=224,
                channels=3,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                resample="bilinear",
                normalize=True,
            )
        else:
            self.config = config

    def forward(
        self, image: torch.Tensor, target_size: torch.Tensor, canvas_size: torch.Tensor
    ):
        # Resize
        ts0, ts1 = target_size.tolist()
        torch._check(ts0 >= 2)
        torch._check(ts0 <= 4000)
        torch._check(ts1 >= 2)
        torch._check(ts1 <= 4000)

        image = resize(
            image,
            size=[ts0, ts1],
            interpolation=self.config.resample,
            antialias=True,
        )

        # Pad
        cs0, cs1 = canvas_size.tolist()
        torch._check(cs0 >= 2)
        torch._check(cs0 <= 4000)
        torch._check(cs1 >= 2)
        torch._check(cs1 <= 4000)
        sizes = [3, cs0, cs1]

        output = torch.empty(
            sizes, dtype=image.dtype, device=image.device, requires_grad=False
        )
        output = torch.fill(output, 0)
        output = torch.ops.preprocess.pad.default(image, output)

        # Normalize
        if self.config.normalize:
            output = v2.functional.normalize(
                output, self.config.image_mean, self.config.image_std
            )

        # Split
        tiles = torch.ops.preprocess.reshape.default(
            output, self.config.channels, self.config.tile_size
        )
        return tiles


class PreprocessModel(EagerModelBase):
    def __init__(self):
        super().__init__()

    def get_eager_model(self):
        model = Preprocess()
        return model

    def get_example_inputs(self):
        image = torch.ones(3, 800, 600)
        target_size = torch.tensor([448, 336])
        canvas_size = torch.tensor([448, 448])
        return (image, target_size, canvas_size)

    def get_dynamic_shapes(self):
        img_h = Dim("img_h", min=1, max=4000)
        img_w = Dim("img_w", min=1, max=4000)

        dynamic_shapes = {
            "image": {1: img_h, 2: img_w},
            "target_size": None,
            "canvas_size": None,
        }
        return dynamic_shapes
