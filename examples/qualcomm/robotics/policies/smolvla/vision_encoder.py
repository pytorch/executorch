# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Mapping

import torch

from executorch.examples.qualcomm.utils import replace_module_with_custom_class
from executorch.examples.qualcomm.robotics.policies.smolvla.config import (
    SmolVLAConfig,
)
from transformers.models.idefics3.modeling_idefics3 import BaseModelOutput
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMConfig,
    SmolVLMConnector,
    SmolVLMEncoder,
    SmolVLMPreTrainedModel,
    SmolVLMVisionConfig,
    SmolVLMVisionEmbeddings,
)


class CustomSmolVLMVisionEmbeddings(SmolVLMVisionEmbeddings):
    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__(config)
        num_patches = (config.image_size // config.patch_size) ** 2
        position_ids = torch.arange(num_patches, dtype=torch.long).unsqueeze(0)
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        position_ids = self.position_ids.to(pixel_values.device)
        if batch_size > 1 and position_ids.size(0) == 1:
            position_ids = position_ids.expand(batch_size, -1)
        return embeddings + self.position_embedding(position_ids)


class CustomSmolVLMVisionTransformer(SmolVLMPreTrainedModel):
    config: SmolVLMVisionConfig

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__(config)
        self.embeddings = SmolVLMVisionEmbeddings(config)
        self.encoder = SmolVLMEncoder(config)
        self.patch_size = config.patch_size
        self.post_layernorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.FloatTensor) -> BaseModelOutput:
        hidden_states = self.embeddings(pixel_values=pixel_values)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return BaseModelOutput(
            last_hidden_state=self.post_layernorm(encoder_outputs.last_hidden_state)
        )


class SmolVLMVisionEncoder(SmolVLMPreTrainedModel):
    def __init__(
        self, config: SmolVLMConfig, img_resized_h: int = 512, img_resized_w: int = 512
    ):
        super().__init__(config)
        self.vision_model = CustomSmolVLMVisionTransformer._from_config(
            config.vision_config
        )
        self.connector = SmolVLMConnector(config)
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

        replace_module_with_custom_class(
            self.vision_model,
            target_class=SmolVLMVisionEmbeddings,
            custom_class=CustomSmolVLMVisionEmbeddings,
            strict=True,
            extra_custom_kwargs={"config": config.vision_config},
        )
    
    @classmethod
    def from_hf_policy(
        cls, config: SmolVLAConfig, policy_state_dict: Mapping[str, torch.Tensor]
    ) -> "SmolVLMVisionEncoder":
        model = cls(config.vlm_config, config.image_size, config.image_size)
        
        # Load vision encoder weight from huggingface
        vision_policy_prefix = "model.vlm_with_expert.vlm.model."
        vision_encoder_weight_from_hf =  {
            key.removeprefix(vision_policy_prefix): value
            for key, value in policy_state_dict.items()
            if key.startswith(vision_policy_prefix + "vision_model.")
            or key.startswith(vision_policy_prefix + "connector.")
        }
        model.load_state_dict(
            vision_encoder_weight_from_hf
        )
        return model

    def get_example_inputs(self):
        return (
            torch.randn(
                (1, 3, self.img_resized_h, self.img_resized_w), dtype=torch.float32
            ),
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ):
        pixel_values = pixel_values[None, ...]
        batch_size, num_images, _, _, _ = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        image_hidden_states = self.vision_model(pixel_values=pixel_values)
        image_embeddings = self.connector(image_hidden_states.last_hidden_state)
        # VLAFlowMatching.embed_prefix scales image embeddings by sqrt(hidden size).
        return image_embeddings * math.sqrt(image_embeddings.shape[-1])

