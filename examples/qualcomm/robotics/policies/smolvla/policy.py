# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SmolVLA policy, mirroring lerobot SmolVLAPolicy.

The action queue lives with the rollout on Terminal B, as in lerobot remote
inference, so this policy exposes predict_action_chunk() but no select_action().
SmolVLAFlowMatching only exposes forward(); this policy owns the conversion
from a LeRobot batch to the forward() contract.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from executorch.examples.qualcomm.robotics.policies.smolvla.config import (
    SmolVLAConfig,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.prefix_decoder import (
    make_additive_attention_mask,
)


@dataclass(frozen=True)
class SmolVLAInputs:
    """Inputs of SmolVLAFlowMatching.forward(), in argument order.

    The field names are also the qnn_smolvla_runner raw input file names.
    """

    pixel_values: Tensor  # [1, cameras, 3, H, W] float32 in [-1, 1]
    prefix_tail: Tensor  # [1, language_tokens + 1, hidden] float32
    prefix_attention_mask: Tensor  # [1, prefix, prefix] float32, 0 attends
    prefix_position_ids: Tensor  # [1, prefix] int32
    noise: Tensor  # [1, chunk_size, max_action_dim] float32
    action_attention_mask: Tensor  # [1, chunk_size, prefix + chunk_size] float32, 0 attends
    suffix_position_ids: Tensor  # [1, chunk_size] int32

    def __post_init__(self) -> None:
        expected = {
            "pixel_values": (5, torch.float32),
            "prefix_tail": (3, torch.float32),
            "prefix_attention_mask": (3, torch.float32),
            "prefix_position_ids": (2, torch.int32),
            "noise": (3, torch.float32),
            "action_attention_mask": (3, torch.float32),
            "suffix_position_ids": (2, torch.int32),
        }
        for (name, (ndim, dtype)), value in zip(expected.items(), self.as_model_args()):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor, got {type(value)!r}")
            if value.ndim != ndim or value.dtype != dtype or value.shape[0] != 1:
                raise ValueError(
                    f"{name} must be a batch-1 rank-{ndim} {dtype} tensor, "
                    f"got {value.dtype} {tuple(value.shape)}"
                )

    def as_model_args(self) -> tuple[Tensor, ...]:
        return (
            self.pixel_values,
            self.prefix_tail,
            self.prefix_attention_mask,
            self.prefix_position_ids,
            self.noise,
            self.action_attention_mask,
            self.suffix_position_ids,
        )


class SmolVLAPolicy:
    """Predict SmolVLA action chunks from policy-preprocessed LeRobot batches.

    The language and state embeddings run eagerly on the host through
    prefix_embedding; model only receives what the three graphs consume.
    """

    def __init__(
        self,
        model: nn.Module | None,
        prefix_embedding: nn.Module,
        config: SmolVLAConfig,
        *,
        seed: int,
    ) -> None:
        self.model = model
        self.prefix_embedding = prefix_embedding
        self.config = config
        self._generator = torch.Generator().manual_seed(seed)

    def prepare_images(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Mirrors SmolVLAPolicy.prepare_images: latest frame, resize with pad, [0, 1] -> [-1, 1]."""
        images = []
        for key in self.config.camera_keys:
            image = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if image.shape[1] != 3:
                raise ValueError(f"{key} must have three channels, got {image.shape[1]}")
            image = resize_with_pad(
                image.float(), self.config.image_size, self.config.image_size
            )
            images.append(image * 2.0 - 1.0)
        pixel_values = torch.stack(images, dim=1)
        image_masks = torch.ones(1, self.config.num_cameras, dtype=torch.bool)
        return pixel_values, image_masks

    def prepare_state(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Mirrors SmolVLAPolicy.prepare_state: latest frame, pad to state_dim."""
        state = batch[self.config.state_key]
        state = state[:, -1] if state.ndim == 3 else state
        if state.shape[-1] > self.config.state_dim:
            raise ValueError(
                f"state width must be <= {self.config.state_dim}, got {state.shape[-1]}"
            )
        return F.pad(state.float(), (0, self.config.state_dim - state.shape[-1]))

    def sample_noise(self) -> Tensor:
        """Mirrors VLAFlowMatching.sample_noise with a seeded generator."""
        return torch.randn(
            (1, self.config.chunk_size, self.config.max_action_dim),
            generator=self._generator,
        )

    @torch.inference_mode()
    def predict_action_chunk(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Mirrors SmolVLAPolicy.predict_action_chunk: prepare, run the model, unpad."""
        actions = self.model(*self.prepare_inputs(batch).as_model_args())
        return actions[:, :, : self.config.action_feature[0]]

    @torch.inference_mode()
    def prepare_inputs(self, batch: Mapping[str, Tensor]) -> SmolVLAInputs:
        """Build the SmolVLAFlowMatching.forward() inputs, mirroring VLAFlowMatching.sample_actions."""
        config = self.config
        pixel_values, image_masks = self.prepare_images(batch)
        language_mask = batch[config.language_mask_key].to(torch.bool)
        prefix_tail = self.prefix_embedding(
            batch[config.language_tokens_key].to(torch.long), self.prepare_state(batch)
        ).float()

        prefix_pad_mask = torch.cat(
            (
                image_masks.repeat_interleave(config.image_tokens_per_camera, dim=1),
                language_mask,
                torch.ones(1, 1, dtype=torch.bool),
            ),
            dim=1,
        )
        if prefix_pad_mask.shape != (1, config.prefix_length):
            raise ValueError(
                f"Prefix mask must have shape [1, {config.prefix_length}], "
                f"got {tuple(prefix_pad_mask.shape)}"
            )
        # Image and language tokens attend bidirectionally; the state token starts a new block.
        prefix_blocks = torch.zeros(1, config.prefix_length, dtype=torch.int32)
        prefix_blocks[:, -1] = 1
        prefix_visibility = make_att_2d_masks(prefix_pad_mask, prefix_blocks)

        suffix_causal = torch.tril(
            torch.ones(config.chunk_size, config.chunk_size, dtype=torch.bool)
        ).unsqueeze(0)
        action_visibility = torch.cat(
            (
                prefix_pad_mask[:, None, :].expand(-1, config.chunk_size, -1),
                suffix_causal,
            ),
            dim=2,
        )
        return SmolVLAInputs(
            pixel_values=pixel_values,
            prefix_tail=prefix_tail,
            prefix_attention_mask=make_additive_attention_mask(prefix_visibility),
            prefix_position_ids=(torch.cumsum(prefix_pad_mask, dim=1) - 1).to(
                torch.int32
            ),
            noise=self.sample_noise(),
            action_attention_mask=make_additive_attention_mask(action_visibility),
            suffix_position_ids=(
                prefix_pad_mask.sum(dim=1, keepdim=True)
                + torch.arange(config.chunk_size)
            ).to(torch.int32),
        )


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Mirrors lerobot SmolVLA make_att_2d_masks (big_vision prefix-LM masks)."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def resize_with_pad(images: Tensor, height: int, width: int) -> Tensor:
    """Resize without distortion and pad on the left and top."""
    current_height, current_width = images.shape[-2:]
    if (current_height, current_width) == (height, width):
        return images
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    return F.pad(resized, (width - resized_width, 0, height - resized_height, 0))
