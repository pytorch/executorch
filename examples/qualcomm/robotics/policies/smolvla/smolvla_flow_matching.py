# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping

import torch
from torch import nn

from executorch.examples.qualcomm.robotics.policies.smolvla.action_expert import (
    SmolVLAActionExpert,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.config import (
    SmolVLAConfig,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.prefix_decoder import (
    SmolVLAPrefixModel,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.vision_encoder import (
    SmolVLMVisionEncoder,
)


class SmolVLAFlowMatching(nn.Module):
    """Compose the three graphs into one action chunk, as qnn_smolvla_runner does on device.

    vision x cameras -> [camera embeddings ; prefix tail] -> prefix K/V
    -> num_steps Euler steps of the Action Expert.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        prefix_model: nn.Module,
        action_expert: nn.Module,
        *,
        num_steps: int = 10,
    ) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.vision_encoder = vision_encoder
        self.prefix_model = prefix_model
        self.action_expert = action_expert
        self.num_steps = num_steps

    @classmethod
    def from_hf_policy(
        cls, config: SmolVLAConfig, policy_state_dict: Mapping[str, torch.Tensor]
    ) -> "SmolVLAFlowMatching":
        vision_encoder = SmolVLMVisionEncoder.from_hf_policy(config, policy_state_dict)
        prefix_model = SmolVLAPrefixModel.from_hf_policy(config, policy_state_dict)
        action_expert = SmolVLAActionExpert.from_hf_policy(config, policy_state_dict)
        return cls(
            vision_encoder=vision_encoder.to(torch.float32).eval(),
            prefix_model=prefix_model.to(torch.float32).eval(),
            action_expert=action_expert.to(torch.float32).eval(),
            num_steps=config.num_steps,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_tail: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        noise: torch.Tensor,
        action_attention_mask: torch.Tensor,
        suffix_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        camera_embeddings = [
            self.vision_encoder(pixel_values[:, camera])
            for camera in range(pixel_values.shape[1])
        ]
        prefix_embeddings = torch.cat((*camera_embeddings, prefix_tail), dim=1)
        k_prefix, v_prefix = self.prefix_model(
            prefix_embeddings, prefix_attention_mask, prefix_position_ids
        )

        actions = noise
        for step in range(self.num_steps):
            velocity = self.action_expert(
                actions,
                torch.tensor([step], dtype=torch.int32),
                k_prefix,
                v_prefix,
                action_attention_mask,
                suffix_position_ids,
            )
            actions = actions - velocity / self.num_steps
        return actions
