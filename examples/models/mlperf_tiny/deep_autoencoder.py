# Copyright (C) 2020 Hitachi, Ltd. All right reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch port of the MLPerf Tiny anomaly detection deep autoencoder."""

import torch
import torch.nn as nn

from executorch.examples.models.model_base import EagerModelBase


class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 640) -> None:
        super().__init__()
        hidden = [128, 128, 128, 128, 8, 128, 128, 128, 128]
        layers = []
        in_dim = input_dim
        for dim in hidden:
            layers.append(nn.Linear(in_dim, dim, bias=True))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        self.encoder_decoder = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_decoder(x)
        x = self.output_layer(x)
        return x


class DeepAutoEncoderModel(EagerModelBase):

    def get_eager_model(self) -> torch.nn.Module:
        return DeepAutoEncoder().eval()

    def get_example_inputs(self):
        return (torch.rand(1, 640) * 2 - 1,)
