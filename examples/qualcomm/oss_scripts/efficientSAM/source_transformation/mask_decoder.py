# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple

import torch
import torch.nn as nn


class MaskDecoderCustom(nn.Module):
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings (the batch dimension is broadcastable).
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        (
            batch_size,
            max_num_queries,
            sparse_embed_dim_1,
            sparse_embed_dim_2,
        ) = sparse_prompt_embeddings.shape

        (
            _,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        ) = image_embeddings.shape

        # QNN don't support dim greater than 4
        image_embeddings_expanded = image_embeddings.expand(max_num_queries, -1, -1, -1)
        image_embeddings_tiled = image_embeddings_expanded.contiguous().view(
            batch_size * max_num_queries,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        )
        sparse_prompt_embeddings = sparse_prompt_embeddings.reshape(
            batch_size * max_num_queries, sparse_embed_dim_1, sparse_embed_dim_2
        )
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings_tiled,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        if multimask_output and self.num_multimask_outputs > 1:
            return masks[:, 1:, :], iou_pred[:, 1:]
        else:
            return masks[:, :1, :], iou_pred[:, :1]

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        # QNN don't support dim greater than 4,
        pos_src = image_pe.expand([tokens.shape[0]] + [*image_pe.shape[1:]])
        b, c, h, w = image_embeddings.shape
        hs, src = self.transformer(image_embeddings, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)

        for upscaling_layer in self.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)
        hyper_in_list: List[torch.Tensor] = []
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


def _replace_maskdecoder_with_custom_op(module: torch.nn.Module):
    from efficient_sam.efficient_sam_decoder import MaskDecoder  # B007

    for _, child in module.named_children():
        if isinstance(child, MaskDecoder):
            child.forward = MaskDecoderCustom.forward.__get__(child, MaskDecoder)
            child.predict_masks = MaskDecoderCustom.predict_masks.__get__(
                child, MaskDecoder
            )
        else:
            _replace_maskdecoder_with_custom_op(child)


def replace_maskdecoder_with_custom_op(module: torch.nn.Module) -> torch.nn.Module:

    _replace_maskdecoder_with_custom_op(module)
    return module
