# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from segment_anything import sam_model_registry  # @manual

from ..model_base import EagerModelBase


class SegmentAnythingModel(EagerModelBase):
    def __init__(self):
        # Use the smallest variant (w/ 3.5B params) by default
        self.model_type = "vit_b"

    def get_eager_model(self) -> torch.nn.Module:
        logging.info(f"Loading segment-anything {self.model_type} model")
        self.sam_model = sam_model_registry[self.model_type]()
        logging.info(f"Loaded segment-anything {self.model_type} model")
        return self.sam_model

    def get_example_inputs(self):
        embed_size = self.sam_model.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        batched_input = [
            # NOTE: SAM can take any of the following inputs independently. For
            # example, if you want to gen an inference model with point-only inputs,
            # just comment out the other inputs.
            {  # multi-points input
                "image": torch.randn(3, 224, 224),
                "original_size": (1500, 2250),
                "point_coords": torch.randint(low=0, high=224, size=(1, 2, 2)),
                "point_labels": torch.randint(low=0, high=4, size=(1, 2)),
            },
            {  # multi-boxes input
                "image": torch.randn(3, 224, 224),
                "original_size": (1500, 2250),
                "boxes": torch.randn(2, 4),
            },
            {  # mask input
                "image": torch.randn(3, 224, 224),
                "original_size": (1500, 2250),
                "mask_input": torch.randn(1, 1, *mask_input_size),
            },
            {  # comb input
                "image": torch.randn(3, 224, 224),
                "original_size": (1500, 2250),
                "point_coords": torch.randint(low=0, high=224, size=(3, 5, 2)),
                "point_labels": torch.randint(low=0, high=4, size=(3, 5)),
                "boxes": torch.randn(3, 4),
                "mask_input": torch.randn(3, 1, *mask_input_size),
            },
        ]
        multimask_output = False
        return (batched_input, multimask_output)
