"""Gemma4 vision and audio encoder wrappers for ExecuTorch export.

The HF Gemma4ImageProcessor pre-patchifies images into:
  pixel_values:        [B, N_max_patches, 3*patch_size^2=768]
  pixel_position_ids:  [B, N_max_patches, 2]  (x,y; padding = -1,-1)

These thin wrappers accept that format, call through the HF submodules,
and return a plain tensor suitable for torch.export (no **kwargs, no
dataclass outputs).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoderExport(nn.Module):
    """Wraps Gemma4VisionModel + embed_vision projector for torch.export.

    Inputs  (from HF Gemma4ImageProcessor.preprocess()):
      pixel_values        (1, N_patches=2520, 768) float32 — pre-patchified
      pixel_position_ids  (1, N_patches=2520, 2)  int64   — (x,y); -1 = padding

    Output:
      soft_tokens  (N_soft=256, text_hidden=1536) float32
    """

    def __init__(self, vision_model, embed_vision):
        super().__init__()
        self.vision_model = vision_model
        self.embed_vision = embed_vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Call HF VisionModel — handles patchification, transformer, pooling internally
        # Returns BaseModelOutputWithPast; .last_hidden_state is (N_soft, 768)
        enc_out = self.vision_model(pixel_values, pixel_position_ids)
        hidden_states = enc_out.last_hidden_state  # (N_soft, 768)

        # Project to text hidden space
        soft_tokens = self.embed_vision(hidden_states)  # (N_soft, 1536)
        return soft_tokens


class AudioEncoderExport(nn.Module):
    """Wraps Gemma4AudioModel + embed_audio projector for torch.export.

    Input  (from HF Gemma4AudioFeatureExtractor):
      input_features  (1, T, n_mels=128) float32 — time-major log-mel spectrogram

    Output:
      soft_tokens  (1, T_out, text_hidden=1536) float32
    """

    def __init__(self, audio_model, embed_audio):
        super().__init__()
        self.audio_model = audio_model
        self.embed_audio = embed_audio

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # Returns Gemma4AudioModelOutput; .last_hidden_state: (1, T_out, 1024)
        audio_out = self.audio_model(input_features).last_hidden_state
        soft_tokens = self.embed_audio(audio_out)  # (1, T_out, 1536)
        return soft_tokens


