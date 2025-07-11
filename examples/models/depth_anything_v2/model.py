# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.examples.models.model_base import EagerModelBase


class DepthAnythingV2Model(EagerModelBase):
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        self.model_name = model_name

    def _load_model(self):
        """Load the Depth Anything V2 model from HuggingFace"""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError(
                "transformers is required for DepthAnythingV2Model. "
                "Install with: pip install transformers"
            )

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModelForDepthEstimation.from_pretrained(self.model_name)

        return model

    def get_eager_model(self) -> torch.nn.Module:
        return DepthAnythingV2Wrapper(self.model_name)

    def get_example_inputs(self):
        """Get example inputs for the model"""
        # Standard input size for Depth Anything V2 models
        # The model expects images of size (3, 518, 518) based on the processor configuration
        return (torch.randn(1, 3, 518, 518),)

    def get_dynamic_shapes(self):
        """Dynamic shapes for variable input sizes"""
        from torch.export import Dim

        batch_size = Dim("batch_size", min=1, max=8)
        height = Dim("height", min=224, max=1024)
        width = Dim("width", min=224, max=1024)

        return ({0: batch_size, 2: height, 3: width},)


class DepthAnythingV2Wrapper(torch.nn.Module):
    """
    Wrapper for Depth Anything V2 model that handles preprocessing and provides a clean interface.
    """

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError(
                "transformers is required for DepthAnythingV2Model. "
                "Install with: pip install transformers"
            )

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

        # Set to evaluation mode
        self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for depth estimation.

        Args:
            pixel_values: Input image tensor of shape (batch_size, 3, height, width)
                         Values should be normalized to [0, 1] range

        Returns:
            predicted_depth: Depth map tensor of shape (batch_size, height, width)
        """
        # The model expects inputs to be preprocessed
        # pixel_values should already be properly normalized and sized

        # Remove torch.no_grad() for export compatibility
        outputs = self.model(pixel_values=pixel_values)
        predicted_depth = outputs.predicted_depth

        # The model outputs depth in a specific format - we may need to interpolate
        # to match the input image size
        if predicted_depth.shape[-2:] != pixel_values.shape[-2:]:
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return predicted_depth

    def preprocess_image(self, image):
        """
        Preprocess a PIL image for the model.
        This method is not used in the forward pass but can be helpful for testing.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"]
