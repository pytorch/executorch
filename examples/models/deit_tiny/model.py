# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

try:
    import timm  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "timm package is required for builtin 'deit_tiny'. Install timm."
    ) from e

from ..model_base import EagerModelBase


class DeiTTinyModel(EagerModelBase):

    def __init__(self):  # type: ignore[override]
        pass

    def get_eager_model(self) -> torch.nn.Module:  # type: ignore[override]
        logging.info("Loading timm deit_tiny_patch16_224 model")
        model = timm.models.deit.deit_tiny_patch16_224(pretrained=True)
        logging.info("Loaded timm deit_tiny_patch16_224 model")
        return model

    def get_example_inputs(self):  # type: ignore[override]
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)


__all__ = ["DeiTTinyModel"]
