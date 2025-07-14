# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn


class PositionEmbeddingRandomCustom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords.unsqueeze(0)
        coords = torch.matmul(
            coords, self.positional_encoding_gaussian_matrix.unsqueeze(0)
        )
        coords = coords.squeeze(0)
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords_0 = coords[:, :, 0] / image_size[1]
        coords_1 = coords[:, :, 1] / image_size[0]
        coords = torch.stack((coords_0, coords_1), dim=-1)

        return self._pe_encoding(coords.to(torch.float))  # B x N x C


def _replace_pos_emb_with_custom_op(module: torch.nn.Module):
    from efficient_sam.efficient_sam_decoder import PositionEmbeddingRandom  # B007

    for _, child in module.named_children():
        if isinstance(child, PositionEmbeddingRandom):
            child._pe_encoding = PositionEmbeddingRandomCustom._pe_encoding.__get__(
                child, PositionEmbeddingRandom
            )
            child.forward_with_coords = (
                PositionEmbeddingRandomCustom.forward_with_coords.__get__(
                    child, PositionEmbeddingRandom
                )
            )
        else:
            _replace_pos_emb_with_custom_op(child)


def replace_pos_emb_with_custom_op(module: torch.nn.Module) -> torch.nn.Module:

    _replace_pos_emb_with_custom_op(module)
    return module
