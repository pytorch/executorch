# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from executorch.extension.llm.custom_ops import op_tile_crop_aot  # noqa
from torch.export import Dim
from torchtune.models.clip.inference._transform import _CLIPImageTransform

from ...model_base import EagerModelBase


@dataclass
class PreprocessConfig:
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None
    resample: str = "bilinear"
    max_num_tiles: int = 4
    tile_size: int = 224
    antialias: bool = False
    # Used for reference eager model from torchtune.
    resize_to_max_canvas: bool = False
    possible_resolutions: Optional[List[Tuple[int, int]]] = None


class CLIPImageTransformModel(EagerModelBase):
    def __init__(
        self,
        config: PreprocessConfig,
    ):
        super().__init__()

        # Eager model.
        self.model = _CLIPImageTransform(
            image_mean=config.image_mean,
            image_std=config.image_std,
            resample=config.resample,
            max_num_tiles=config.max_num_tiles,
            tile_size=config.tile_size,
            antialias=config.antialias,
        )

        # Replace non-exportable ops with custom ops.
        self.model.tile_crop = torch.ops.preprocess.tile_crop.default

    def get_eager_model(self) -> torch.nn.Module:
        return self.model

    def get_example_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.ones(3, 800, 600)
        target_size = torch.tensor([448, 336])
        canvas_size = torch.tensor([448, 448])
        return (image, target_size, canvas_size)

    def get_dynamic_shapes(self) -> Dict[str, Dict[int, Dim]]:
        img_h = Dim("img_h", min=1, max=4000)
        img_w = Dim("img_w", min=1, max=4000)

        dynamic_shapes = {
            "image": {1: img_h, 2: img_w},
            "target_size": None,
            "canvas_size": None,
        }
        return dynamic_shapes
