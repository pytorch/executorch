# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Tuple

import numpy as np
import PIL
import pytest
import torch

# Import these first. Otherwise, the custom ops are not registered.
from executorch.extension.pybindings import portable_lib  # noqa # usort: skip
from executorch.extension.llm.custom_ops import op_tile_crop_aot  # noqa # usort: skip

from executorch.examples.models.llama3_2_vision.preprocess.model import (
    CLIPImageTransformModel,
    PreprocessConfig,
)

from executorch.exir import EdgeCompileConfig, to_edge

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

from PIL import Image
from torch._inductor.package import package_aoti

from torchtune.models.clip.inference._transform import CLIPImageTransform

from torchtune.modules.transforms.vision_utils.get_canvas_best_fit import (
    find_supported_resolutions,
    get_canvas_best_fit,
)

from torchtune.modules.transforms.vision_utils.get_inscribed_size import (
    get_inscribed_size,
)

from torchvision.transforms.v2 import functional as F


def initialize_models(resize_to_max_canvas: bool) -> Dict[str, Any]:
    config = PreprocessConfig(resize_to_max_canvas=resize_to_max_canvas)

    reference_model = CLIPImageTransform(
        image_mean=config.image_mean,
        image_std=config.image_std,
        resample=config.resample,
        antialias=config.antialias,
        tile_size=config.tile_size,
        max_num_tiles=config.max_num_tiles,
        resize_to_max_canvas=config.resize_to_max_canvas,
        possible_resolutions=None,
    )

    # Eager model.
    model = CLIPImageTransformModel(config)

    # Exported model.
    exported_model = torch.export.export(
        model.get_eager_model(),
        model.get_example_inputs(),
        dynamic_shapes=model.get_dynamic_shapes(),
        strict=False,
    )

    # AOTInductor model.
    so = torch._export.aot_compile(
        exported_model.module(),
        args=model.get_example_inputs(),
        options={"aot_inductor.package": True},
        dynamic_shapes=model.get_dynamic_shapes(),
    )
    aoti_path = "preprocess.pt2"
    package_aoti(aoti_path, so)

    edge_program = to_edge(
        exported_model, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    executorch_model = edge_program.to_executorch()

    # Re-export as ExecuTorch edits the ExportedProgram.
    exported_model = torch.export.export(
        model.get_eager_model(),
        model.get_example_inputs(),
        dynamic_shapes=model.get_dynamic_shapes(),
        strict=False,
    )

    return {
        "config": config,
        "reference_model": reference_model,
        "model": model,
        "exported_model": exported_model,
        "aoti_path": aoti_path,
        "executorch_model": executorch_model,
    }


# From https://github.com/pytorch/torchtune/blob/main/tests/test_utils.py#L231
def assert_expected(
    actual: Any,
    expected: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_device: bool = True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


class TestImageTransform:
    """
    This test checks that the exported image transform model produces the
    same output as the reference model.

    Reference model: CLIPImageTransform
        https://github.com/pytorch/torchtune/blob/main/torchtune/models/clip/inference/_transforms.py#L115
    Eager and exported models: _CLIPImageTransform
        https://github.com/pytorch/torchtune/blob/main/torchtune/models/clip/inference/_transforms.py#L26
    """

    models_no_resize = initialize_models(resize_to_max_canvas=False)
    models_resize = initialize_models(resize_to_max_canvas=True)

    @pytest.fixture(autouse=True)
    def setup_function(self):
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

    def run_preprocess(
        self,
        image_size: Tuple[int],
        expected_shape: torch.Size,
        resize_to_max_canvas: bool,
        expected_tile_means: List[float],
        expected_tile_max: List[float],
        expected_tile_min: List[float],
        expected_ar: List[int],
    ) -> None:
        models = self.models_resize if resize_to_max_canvas else self.models_no_resize
        # Prepare image input.
        image = (
            np.random.randint(0, 256, np.prod(image_size))
            .reshape(image_size)
            .astype(np.uint8)
        )
        image = PIL.Image.fromarray(image)

        # Run reference model.
        reference_model = models["reference_model"]
        reference_output = reference_model(image=image)
        reference_image = reference_output["image"]
        reference_ar = reference_output["aspect_ratio"].tolist()

        # Check output shape and aspect ratio matches expected values.
        assert (
            reference_image.shape == expected_shape
        ), f"Expected shape {expected_shape} but got {reference_image.shape}"

        assert (
            reference_ar == expected_ar
        ), f"Expected ar {reference_ar} but got {expected_ar}"

        # Check pixel values within expected range [0, 1]
        assert (
            0 <= reference_image.min() <= reference_image.max() <= 1
        ), f"Expected pixel values in range [0, 1] but got {reference_image.min()} to {reference_image.max()}"

        # Check mean, max, and min values of the tiles match expected values.
        for i, tile in enumerate(reference_image):
            assert_expected(
                tile.mean().item(), expected_tile_means[i], rtol=0, atol=1e-4
            )
            assert_expected(tile.max().item(), expected_tile_max[i], rtol=0, atol=1e-4)
            assert_expected(tile.min().item(), expected_tile_min[i], rtol=0, atol=1e-4)

        # Check num tiles matches the product of the aspect ratio.
        expected_num_tiles = reference_ar[0] * reference_ar[1]
        assert (
            expected_num_tiles == reference_image.shape[0]
        ), f"Expected {expected_num_tiles} tiles but got {reference_image.shape[0]}"

        # Pre-work for eager and exported models. The reference model performs these
        # calculations and passes the result to _CLIPImageTransform, the exportable model.
        image_tensor, inscribed_size, best_resolution = self.prepare_inputs(
            image=image, config=models["config"]
        )

        # Run eager model and check it matches reference model.
        eager_model = models["model"].get_eager_model()
        eager_image, eager_ar = eager_model(
            image_tensor, inscribed_size, best_resolution
        )
        eager_ar = eager_ar.tolist()
        assert_expected(eager_image, reference_image, rtol=0, atol=1e-4)
        assert (
            reference_ar == eager_ar
        ), f"Eager model: expected {reference_ar} but got {eager_ar}"

        # Run exported model and check it matches reference model.
        exported_model = models["exported_model"]
        exported_image, exported_ar = exported_model.module()(
            image_tensor, inscribed_size, best_resolution
        )
        exported_ar = exported_ar.tolist()
        assert_expected(exported_image, reference_image, rtol=0, atol=1e-4)
        assert (
            reference_ar == exported_ar
        ), f"Exported model: expected {reference_ar} but got {exported_ar}"

        # Run executorch model and check it matches reference model.
        executorch_model = models["executorch_model"]
        executorch_module = _load_for_executorch_from_buffer(executorch_model.buffer)
        et_image, et_ar = executorch_module.forward(
            (image_tensor, inscribed_size, best_resolution)
        )
        assert_expected(et_image, reference_image, rtol=0, atol=1e-4)
        assert (
            reference_ar == et_ar.tolist()
        ), f"Executorch model: expected {reference_ar} but got {et_ar.tolist()}"

        # Run aoti model and check it matches reference model.
        aoti_path = models["aoti_path"]
        aoti_model = torch._inductor.aoti_load_package(aoti_path)
        aoti_image, aoti_ar = aoti_model(image_tensor, inscribed_size, best_resolution)
        assert_expected(aoti_image, reference_image, rtol=0, atol=1e-4)
        assert (
            reference_ar == aoti_ar.tolist()
        ), f"AOTI model: expected {reference_ar} but got {aoti_ar.tolist()}"

    # This test setup mirrors the one in torchtune:
    # https://github.com/pytorch/torchtune/blob/main/tests/torchtune/models/clip/test_clip_image_transform.py
    # The values are slightly different, as torchtune uses antialias=True,
    # and this test uses antialias=False, which is exportable (has a portable kernel).
    def test_preprocess1(self):
        self.run_preprocess(
            (100, 400, 3),  # image_size
            torch.Size([2, 3, 224, 224]),  # expected shape
            False,  # resize_to_max_canvas
            [0.2230, 0.1763],  # expected_tile_means
            [1.0, 1.0],  # expected_tile_max
            [0.0, 0.0],  # expected_tile_min
            [1, 2],  # expected_aspect_ratio
        )

    def test_preprocess2(self):
        self.run_preprocess(
            (1000, 300, 3),  # image_size
            torch.Size([4, 3, 224, 224]),  # expected shape
            True,  # resize_to_max_canvas
            [0.5005, 0.4992, 0.5004, 0.1651],  # expected_tile_means
            [0.9976, 0.9940, 0.9936, 0.9906],  # expected_tile_max
            [0.0037, 0.0047, 0.0039, 0.0],  # expected_tile_min
            [4, 1],  # expected_aspect_ratio
        )

    def test_preprocess3(self):
        self.run_preprocess(
            (200, 200, 3),  # image_size
            torch.Size([4, 3, 224, 224]),  # expected shape
            True,  # resize_to_max_canvas
            [0.5012, 0.5020, 0.5010, 0.4991],  # expected_tile_means
            [0.9921, 0.9925, 0.9969, 0.9908],  # expected_tile_max
            [0.0056, 0.0069, 0.0059, 0.0032],  # expected_tile_min
            [2, 2],  # expected_aspect_ratio
        )

    def test_preprocess4(self):
        self.run_preprocess(
            (600, 200, 3),  # image_size
            torch.Size([3, 3, 224, 224]),  # expected shape
            False,  # resize_to_max_canvas
            [0.4472, 0.4468, 0.3031],  # expected_tile_means
            [1.0, 1.0, 1.0],  # expected_tile_max
            [0.0, 0.0, 0.0],  # expected_tile_min
            [3, 1],  # expected_aspect_ratio
        )
