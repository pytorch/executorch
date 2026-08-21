# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import torch

from executorch.examples.models.mlperf_tiny import ResNet8

logging.basicConfig(level=logging.INFO)


class ModelSource(Enum):
    MLPERF_TINY = 0

MODEL_NAME_TO_MODEL_CLASS = {
    "image_classification": ResNet8,
}

class ModelManager:
    def get_model(self, model_name: str, **kwargs) -> torch.nn.Module:
        if model_name not in MODEL_NAME_TO_MODEL_CLASS:
            raise ValueError(f"Model {model_name} not supported!")

        logging.info(f"Loading MLPerf Tiny model {model_name}...")
        model = MODEL_NAME_TO_MODEL_CLASS[model_name](**kwargs)
        model.eval()
        logging.info("Model loaded successfully.")
        return model