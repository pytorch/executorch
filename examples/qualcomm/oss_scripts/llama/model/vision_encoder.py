# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.examples.qualcomm.utils import replace_module_with_custom_class
from torch import nn

from transformers.models.idefics3.modeling_idefics3 import (
    BaseModelOutput,
    Idefics3Config,
    Idefics3Connector,
    Idefics3Encoder,
    Idefics3PreTrainedModel,
    Idefics3VisionConfig,
    Idefics3VisionEmbeddings,
)
from transformers.models.internvl.modeling_internvl import (
    InternVLConfig,
    InternVLMultiModalProjector,
    InternVLVisionModel,
)


# Custom implementation based on `transformers/models/idefics3/modeling_idefics3/Idefics3VisionEmbeddings.py` (Transformers v5.0.0rc1)
#
# Qualcomm optimization:
# Precompute and register positional IDs as a buffer to avoid computation during forward passes.
class CustomIdefics3VisionEmbeddings(Idefics3VisionEmbeddings):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # ========================== Qualcomm Changed: Precompute position ids ==========================
        S = self.num_patches_per_side
        H = self.image_size // self.patch_size
        W = self.image_size // self.patch_size

        # Assume batch size of 1 for precomputation
        B = 1

        # Create boundaries for bucketize
        boundaries = torch.arange(1.0 / S, 1.0, 1.0 / S)  # [S-1]

        # Assume full attention mask (all True) for precomputation
        patch_attention_mask = torch.ones((B, H, W), dtype=torch.bool)

        # Calculate nb_h and nb_w (number of True values in first column/row)
        nb_h = patch_attention_mask[:, :, 0].sum(dim=1)  # [B]
        nb_w = patch_attention_mask[:, 0, :].sum(dim=1)  # [B]
        nb_h_clamped = torch.clamp(nb_h, min=1).float()  # [B]
        nb_w_clamped = torch.clamp(nb_w, min=1).float()  # [B]

        # Create indices
        h_idx = torch.arange(H, dtype=torch.float32).unsqueeze(0)  # [1, H]
        w_idx = torch.arange(W, dtype=torch.float32).unsqueeze(0)  # [1, W]

        # Calculate fractional positions
        frac_h = (h_idx / nb_h_clamped.unsqueeze(1)) * (1.0 - 1e-6)  # [B, H]
        frac_w = (w_idx / nb_w_clamped.unsqueeze(1)) * (1.0 - 1e-6)  # [B, W]

        # Bucketize to get position indices
        bucket_h = torch.bucketize(frac_h, boundaries, right=True)  # [B, H]
        bucket_w = torch.bucketize(frac_w, boundaries, right=True)  # [B, W]

        # Create position grid: pos = h * S + w
        pos_grid = bucket_h.unsqueeze(2) * S + bucket_w.unsqueeze(1)  # [B, H, W]
        pos_full = pos_grid.reshape(B, H * W)  # [B, H*W]

        # Apply attention mask
        mask_flat = patch_attention_mask.view(B, H * W)  # [B, H*W]
        position_ids = torch.where(
            mask_flat, pos_full, torch.zeros_like(pos_full)
        )  # [B, H*W]

        # Register the precomputed position_ids
        self.register_buffer("position_ids", position_ids, persistent=False)
        # ===============================================================================================

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # pixel_values: [B, 3, max_im_h, max_im_w]
        B, _, max_im_h, max_im_w = pixel_values.shape

        # 1) patch embedding: [B, C, H, W] -> [B, H*W, C]
        patch_embeds = self.patch_embedding(pixel_values)  # [B, C, H, W]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # ========================== Qualcomm Changed: Use the precomputed position_ids ==========================
        position_ids = self.position_ids.to(pixel_values.device)
        # ========================================================================================================

        # Expand to match batch size if needed
        if B > 1 and position_ids.size(0) == 1:
            position_ids = position_ids.expand(B, -1)

        # Add positional embedding
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


# Custom implementation based on `transformers/models/idefics3/modeling_idefics3/Idefics3VisionTransformer.py` (Transformers v5.0.0rc1)
#
# Qualcomm changes:
# Assume the image is non-empty and skip attention mask propagation to the encoder
class CustomIdefics3VisionTransformer(Idefics3PreTrainedModel):
    config: Idefics3VisionConfig

    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)
        self.embeddings = Idefics3VisionEmbeddings(config)
        self.encoder = Idefics3Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values=pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Custom implementation based on `transformers/models/idefics3/modeling_idefics3.py` (Transformers v5.0.0rc1).
#
# Qualcomm optimization:
# - Dynamic shape support is removed; computations are now static for efficiency.
# - After image preprocessing, we assume all patch images are valid (non-empty),
#   so attention masks are no longer required in the Vision Transformer.
class Idefics3VisionEncoder(Idefics3PreTrainedModel):
    def __init__(
        self, config: Idefics3Config, img_resized_h: int = 512, img_resized_w: int = 512
    ):
        super().__init__(config)
        self.vision_model = CustomIdefics3VisionTransformer._from_config(
            config.vision_config
        )
        self.connector = Idefics3Connector(config)
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

        replace_module_with_custom_class(
            self.vision_model,
            target_class=Idefics3VisionEmbeddings,
            custom_class=CustomIdefics3VisionEmbeddings,
            strict=True,
            extra_custom_kwargs={"config": config.vision_config},
        )

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # HTP Prepare failed when pixel_values has 5D dimension, so we squeeze the batch dimension here.
        pixel_values = pixel_values[0]
        return (pixel_values.squeeze(0),)

    def get_example_inputs(self):
        # pixel values - use config dimensions instead of hardcoded values
        return (
            torch.randn(
                (1, 3, self.img_resized_h, self.img_resized_w), dtype=torch.float32
            ),
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.LongTensor = None,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        pixel_values = pixel_values[None, ...]
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(
            batch_size * num_images, *pixel_values.shape[2:]
        )

        # ========================== Qualcomm Changed ==========================
        # Since dynamic shapes are unsupported, we assume all patches are valid and don't need `patch_attention_mask`.

        # # Remove padding images - padding images are full 0.
        # nb_values_per_image = pixel_values.shape[1:].numel()
        # real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        # pixel_values = pixel_values[real_images_inds].contiguous()

        # # Handle the vision attention mask
        # if pixel_attention_mask is None:
        #     pixel_attention_mask = torch.ones(
        #         size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
        #         dtype=torch.bool,
        #         device=pixel_values.device,
        #     )
        # else:
        #     # Remove padding images from the mask
        #     pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
        #     pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        # patch_size = self.config.vision_config.patch_size
        # patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        # patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        # patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
        # ======================================================================

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values)
        image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states.last_hidden_state)
        return image_hidden_states


# Copy from transformers/models/internvl/modeling_internvl.py (Transformers v5.0.0rc1).
class InternVL3VisionEncoder(torch.nn.Module):
    def __init__(
        self, config: InternVLConfig, img_resized_h: int = 448, img_resized_w: int = 448
    ):
        super(InternVL3VisionEncoder, self).__init__()
        self.vision_tower = InternVLVisionModel(config.vision_config)
        self.multi_modal_projector = InternVLMultiModalProjector(config)
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        return pixel_values

    def get_example_inputs(self):
        # pixel values - use config dimensions instead of hardcoded values
        return (
            torch.randn(
                (1, 3, self.img_resized_h, self.img_resized_w), dtype=torch.float32
            ),
        )

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor of shape (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError(
                "Height and width must be divisible by scale_factor for proper downsampling."
            )

        # Reshape to allow downsampling
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(
            batch_size,
            int(height * scale_factor),
            int(width * scale_factor),
            int(channels / (scale_factor**2)),
        )

        # Swap height and width back for proper orientation
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int` or `list[int]`):
                Layer index or list of layer indices to extract features from.
        Returns:
            vision_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`.
        """
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        downsample_ratio = self.config.downsample_ratio
        if vision_feature_layer == -1:
            vision_features = self.vision_tower(
                pixel_values=pixel_values
            ).last_hidden_state
        else:
            vision_features = self.vision_model(
                pixel_values=pixel_values
            ).hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        # Calculate dimensions based on vision features
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(
            batch_size, feature_size, feature_size, -1
        )

        # Apply downsampling using pixel shuffle
        vision_features = self.pixel_shuffle(
            vision_features, scale_factor=downsample_ratio
        )

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(
            batch_size, -1, vision_features.shape[-1]
        )

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features)
        return vision_features
