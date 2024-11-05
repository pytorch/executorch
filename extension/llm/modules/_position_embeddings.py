# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# An torch.export() friendly version of torchtune's positional embeddings.
# Added torch._check() to make sure guards on symints are enforced.
# See https://github.com/pytorch/torchtune/blob/main/torchtune/models/clip/_position_embeddings.py

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(
        self,
        max_num_tiles: int,
        embed_dim: int,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.embed_dim = embed_dim

        scale = embed_dim**-0.5
        self.embedding = nn.Parameter(
            scale * torch.randn(max_num_tiles, max_num_tiles, 1, embed_dim)
        )
        self.gate = nn.Parameter(torch.zeros(1))

        # Register load hook to interpolate positional embeddings
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    # TODO: Switch to public method after 2.5 is stable
    @torch.no_grad()
    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ):
        """
        Interpolates positional embeddings to accomodate different number of tiles,
        in case the model was instantiated with different
        settings than the one you are loading the state dict from.

        For more info, check self._dynamic_resize function.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: if the shape of the loaded embedding is not compatible with the current embedding.
            ValueError: if max_num_tiles_x, max_num_tiles_y are not equal.
            ValueError: if after interpolation, the shape of the loaded embedding is not compatible with the current embedding.
        """

        embedding = state_dict.get(prefix + "embedding")

        if embedding is not None:

            # ckpt pos emb
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,
                tgt_num_tokens,
                tgt_emb,
            ) = self.embedding.shape

            # instantiated pos emb
            (
                inpt_max_num_tiles_x,
                inpt_max_num_tiles_y,
                inpt_num_tokens,
                inpt_emb,
            ) = state_dict[prefix + "embedding"].shape

            # sanity check
            if inpt_num_tokens != tgt_num_tokens or inpt_emb != tgt_emb:
                raise ValueError(
                    "Expected embedding shape to be (..., num_tokens, tgt_emb) to match"
                    f" but found shapes {self.embedding.shape} and {state_dict[prefix + 'embedding'].shape}"
                )

            if inpt_max_num_tiles_x != inpt_max_num_tiles_y:
                raise ValueError(
                    "Expected max_num_tiles_x, max_num_tiles_y to be equal but found, but found"
                    f"(max_num_tiles_x, max_num_tiles_y, 1, embed_dim) = {self.embedding.shape}"
                )

            # resize ckpt to match instantiated shape
            embedding_new = self._resize_position_embedding(
                embedding, tgt_max_num_tiles=tgt_max_num_tiles_x
            )

            # update state dict
            state_dict[prefix + "embedding"] = embedding_new
            if embedding_new.shape != self.embedding.shape:
                raise ValueError(
                    "Expected embedding shape and embedding_new.shape to match"
                    f" but found shapes {self.embedding.shape} and {embedding_new.shape}"
                )

    @staticmethod
    def _resize_position_embedding(
        embedding: torch.Tensor, tgt_max_num_tiles: int
    ) -> torch.Tensor:
        """
        Interpolates positional embeddings to accomodate a different max_num_tiles. These
        are the only dimensions that changes during interpolation.

        Args:
            embedding (torch.Tensor): torch.Tensor with shape (max_num_tiles, max_num_tiles, 1, embed_dim
            tgt_max_num_tiles (int): The number of tiles to resize to.

        Returns:
            torch.Tensor: The resized embedding.

        Example:
            >>> import torch
            >>> # create dummy embedding
            >>> embedding = torch.arange(2*2*2*2).reshape(2, 2, 2, 2).float()
            >>> resized_embed = _dynamic_resize(embedding, tgt_max_num_tiles=1)
            >>> print(resized_embed.shape)
            >>> torch.Size([1, 1, 2, 2])
        """
        # set max_num_tiles to the last dimension
        embedding = embedding.permute(2, 3, 0, 1)

        embedding = F.interpolate(
            embedding,
            size=(tgt_max_num_tiles, tgt_max_num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # permute to the original shape
        embedding = embedding.permute(2, 3, 0, 1)
        return embedding

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        args:
            x (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, n_tiles, n_tokens, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        torch._check(n_tiles <= self.max_num_tiles)

        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_tiles_h = n_tiles_h.item()
            n_tiles_w = n_tiles_w.item()

            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            torch._check_is_size(n_tiles_h)
            torch._check_is_size(n_tiles_w)
            torch._check(n_tiles_h >= 1)
            torch._check(n_tiles_w >= 1)
            torch._check(n_tiles_h <= self.max_num_tiles)
            torch._check(n_tiles_w <= self.max_num_tiles)
            padded_embedding = F.pad(self.embedding, (0, 0, 0, 0, 0, 1, 0, 1))
            pos_embed = padded_embedding[:n_tiles_h, :n_tiles_w, :, :]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.clone()
            pos_embed = pos_embed.reshape(n_non_padded_tiles, 1, self.embed_dim)

            x = F.pad(x, (0, 0, 0, 0, 0, 1, 0, 0))
            torch._check_is_size(n_non_padded_tiles)
            torch._check(n_non_padded_tiles < x.size(1))
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()
            x = x[:, :n_tiles, :, :]

        return x


def replace_tile_positional_embedding(model: nn.Module) -> nn.Module:
    """
    Replace the tile positional embedding from torchtune with an export-friendly one.
    Recursively searches the submodules of the model and replaces the tile positional embedding if found.
    Args:
        model (nn.Module): The model to replace the tile positional embedding in.

    Returns:
        nn.Module: The model after replacing the tile positional embedding.

    """
    from torchtune.models.clip._position_embeddings import (
        TilePositionalEmbedding as TuneTilePositionalEmbedding,
    )

    for name, module in model.named_children():
        if isinstance(module, TuneTilePositionalEmbedding):
            logging.info(
                f"Replacing tile positional embedding in {name} with export-friendly one."
            )
            max_num_tiles, _, _, embed_dim = module.embedding.shape
            mod = TilePositionalEmbedding(
                max_num_tiles=max_num_tiles,
                embed_dim=embed_dim,
            )
            mod.load_state_dict(module.state_dict())
            setattr(
                model,
                name,
                mod,
            )
        else:
            replace_tile_positional_embedding(module)
    return model
