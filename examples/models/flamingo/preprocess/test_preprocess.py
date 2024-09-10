# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import PIL
import torch

from executorch.extension.pybindings import portable_lib  # noqa # usort: skip
from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa # usort: skip
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

from parameterized import parameterized
from PIL import Image

from torchtune.models.clip.inference._transform import (
    _CLIPImageTransform,
    CLIPImageTransform,
)

from torchtune.modules.transforms.vision_utils.get_canvas_best_fit import (
    find_supported_resolutions,
    get_canvas_best_fit,
)

from torchtune.modules.transforms.vision_utils.get_inscribed_size import (
    get_inscribed_size,
)
from torchvision.transforms.v2 import functional as F

from .export_preprocess_lib import export_preprocess, lower_to_executorch_preprocess


@dataclass
class PreprocessConfig:
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None
    resize_to_max_canvas: bool = True
    resample: str = "bilinear"
    antialias: bool = False
    tile_size: int = 224
    max_num_tiles: int = 4
    possible_resolutions = None


class TestImageTransform(unittest.TestCase):
    """
    This unittest checks that the exported image transform model produces the
    same output as the reference model.

    Reference model: CLIPImageTransform
        https://github.com/pytorch/torchtune/blob/main/torchtune/models/clip/inference/_transforms.py#L115
    Eager and exported models: _CLIPImageTransform
        https://github.com/pytorch/torchtune/blob/main/torchtune/models/clip/inference/_transforms.py#L26
    """

    def setUp(self):
        np.random.seed(0)

    def prepare_inputs(
        self, image: Image.Image, config: PreprocessConfig
    ) -> Tuple[torch.Tensor]:
        """
        Prepare inputs for eager and exported models:
        - Convert PIL image to tensor.
        - Calculate the best resolution; a canvas with height and width divisible by tile_size.
        - Calculate the inscribed size; the size of the image inscribed within best_resolution,
            without distortion.

        These calculations are done by the reference model inside __init__ and __call__
        https://github.com/pytorch/torchtune/blob/main/torchtune/models/clip/inference/_transforms.py#L115
        """
        image_tensor = F.to_dtype(
            F.grayscale_to_rgb_image(F.to_image(image)), scale=True
        )

        # The above converts the PIL image into a torchvision tv_tensor.
        # Convert the tv_tensor into a torch.Tensor.
        image_tensor = image_tensor + 0

        # Ensure tensor is contiguous for executorch.
        image_tensor = image_tensor.contiguous()

        # Calculate possible resolutions.
        possible_resolutions = config.possible_resolutions
        if possible_resolutions is None:
            possible_resolutions = find_supported_resolutions(
                max_num_tiles=config.max_num_tiles, tile_size=config.tile_size
            )
        possible_resolutions = torch.tensor(possible_resolutions).reshape(-1, 2)

        # Limit resizing.
        max_size = None if config.resize_to_max_canvas else config.tile_size

        # Find the best canvas to fit the image without distortion.
        best_resolution = get_canvas_best_fit(
            image=image_tensor,
            possible_resolutions=possible_resolutions,
            resize_to_max_canvas=config.resize_to_max_canvas,
        )
        best_resolution = torch.tensor(best_resolution)

        # Find the dimensions of the image, such that it is inscribed within best_resolution
        # without distortion.
        inscribed_size = get_inscribed_size(
            image_tensor.shape[-2:], best_resolution, max_size
        )
        inscribed_size = torch.tensor(inscribed_size)

        return image_tensor, inscribed_size, best_resolution

    # This test setup mirrors the one in torchtune:
    # https://github.com/pytorch/torchtune/blob/main/tests/torchtune/models/clip/test_clip_image_transform.py
    # The values are slightly different, as torchtune uses antialias=True,
    # and this test uses antialias=False, which is exportable (has a portable kernel).
    @parameterized.expand(
        [
            (
                (100, 400, 3),  # image_size
                torch.Size([2, 3, 224, 224]),  # expected shape
                False,  # resize_to_max_canvas
                [0.2230, 0.1763],  # expected_tile_means
                [1.0, 1.0],  # expected_tile_max
                [0.0, 0.0],  # expected_tile_min
                [1, 2],  # expected_aspect_ratio
            ),
            (
                (1000, 300, 3),  # image_size
                torch.Size([4, 3, 224, 224]),  # expected shape
                True,  # resize_to_max_canvas
                [0.5005, 0.4992, 0.5004, 0.1651],  # expected_tile_means
                [0.9976, 0.9940, 0.9936, 0.9906],  # expected_tile_max
                [0.0037, 0.0047, 0.0039, 0.0],  # expected_tile_min
                [4, 1],  # expected_aspect_ratio
            ),
            (
                (200, 200, 3),  # image_size
                torch.Size([4, 3, 224, 224]),  # expected shape
                True,  # resize_to_max_canvas
                [0.5012, 0.5020, 0.5010, 0.4991],  # expected_tile_means
                [0.9921, 0.9925, 0.9969, 0.9908],  # expected_tile_max
                [0.0056, 0.0069, 0.0059, 0.0032],  # expected_tile_min
                [2, 2],  # expected_aspect_ratio
            ),
            (
                (600, 200, 3),  # image_size
                torch.Size([3, 3, 224, 224]),  # expected shape
                False,  # resize_to_max_canvas
                [0.4472, 0.4468, 0.3031],  # expected_tile_means
                [1.0, 1.0, 1.0],  # expected_tile_max
                [0.0, 0.0, 0.0],  # expected_tile_min
                [3, 1],  # expected_aspect_ratio
            ),
        ]
    )
    def test_preprocess(
        self,
        image_size: Tuple[int],
        expected_shape: torch.Size,
        resize_to_max_canvas: bool,
        expected_tile_means: List[float],
        expected_tile_max: List[float],
        expected_tile_min: List[float],
        expected_ar: List[int],
    ) -> None:
        config = PreprocessConfig(resize_to_max_canvas=resize_to_max_canvas)

        reference_model = CLIPImageTransform(
            image_mean=config.image_mean,
            image_std=config.image_std,
            resize_to_max_canvas=config.resize_to_max_canvas,
            resample=config.resample,
            antialias=config.antialias,
            tile_size=config.tile_size,
            max_num_tiles=config.max_num_tiles,
            possible_resolutions=None,
        )

        eager_model = _CLIPImageTransform(
            image_mean=config.image_mean,
            image_std=config.image_std,
            resample=config.resample,
            antialias=config.antialias,
            tile_size=config.tile_size,
            max_num_tiles=config.max_num_tiles,
        )

        exported_model = export_preprocess(
            image_mean=config.image_mean,
            image_std=config.image_std,
            resample=config.resample,
            antialias=config.antialias,
            tile_size=config.tile_size,
            max_num_tiles=config.max_num_tiles,
        )

        executorch_model = lower_to_executorch_preprocess(exported_model)
        executorch_module = _load_for_executorch_from_buffer(executorch_model.buffer)

        # Prepare image input.
        image = (
            np.random.randint(0, 256, np.prod(image_size))
            .reshape(image_size)
            .astype(np.uint8)
        )
        image = PIL.Image.fromarray(image)

        # Run reference model.
        reference_output = reference_model(image=image)
        reference_image = reference_output["image"]
        reference_ar = reference_output["aspect_ratio"].tolist()

        # Check output shape and aspect ratio matches expected values.
        self.assertEqual(reference_image.shape, expected_shape)
        self.assertEqual(reference_ar, expected_ar)

        # Check pixel values within expected range [0, 1]
        self.assertTrue(0 <= reference_image.min() <= reference_image.max() <= 1)

        # Check mean, max, and min values of the tiles match expected values.
        for i, tile in enumerate(reference_image):
            self.assertAlmostEqual(
                tile.mean().item(), expected_tile_means[i], delta=1e-4
            )
            self.assertAlmostEqual(tile.max().item(), expected_tile_max[i], delta=1e-4)
            self.assertAlmostEqual(tile.min().item(), expected_tile_min[i], delta=1e-4)

        # Check num tiles matches the product of the aspect ratio.
        expected_num_tiles = reference_ar[0] * reference_ar[1]
        self.assertEqual(expected_num_tiles, reference_image.shape[0])

        # Pre-work for eager and exported models. The reference model performs these
        # calculations and passes the result to _CLIPImageTransform, the exportable model.
        image_tensor, inscribed_size, best_resolution = self.prepare_inputs(
            image=image, config=config
        )

        # Run eager model and check it matches reference model.
        eager_image, eager_ar = eager_model(
            image_tensor, inscribed_size, best_resolution
        )
        eager_ar = eager_ar.tolist()
        self.assertTrue(torch.allclose(reference_image, eager_image))
        self.assertEqual(reference_ar, eager_ar)

        # Run exported model and check it matches reference model.
        exported_image, exported_ar = exported_model.module()(
            image_tensor, inscribed_size, best_resolution
        )
        exported_ar = exported_ar.tolist()
        self.assertTrue(torch.allclose(reference_image, exported_image))
        self.assertEqual(reference_ar, exported_ar)

        # Run executorch model and check it matches reference model.
        et_image, et_ar = executorch_module.forward(
            (image_tensor, inscribed_size, best_resolution)
        )
        self.assertTrue(torch.allclose(reference_image, et_image))
        self.assertEqual(reference_ar, et_ar.tolist())
