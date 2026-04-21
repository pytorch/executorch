"""Gemma4 vision and audio encoder wrappers for ExecuTorch export.

Conforms to the standard MultimodalPrefiller ABI: each encoder takes a single
raw modality tensor and returns soft tokens shaped (1, N_soft, hidden) ready
to feed into text_decoder.

Patchification, position-id generation, and any other preprocessing that the
HF processor does is baked into the model graph here, so the C++ runner only
has to load and resize the image (or load PCM and compute mel for audio).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Vision constants matching HF Gemma4ImageProcessor (image_seq_length=280,
# patch_size=16, pooling_kernel_size=3 → 60x42 patch grid → 280 soft tokens).
PATCH_SIZE = 16
GRID_W = 60
GRID_H = 42
N_PATCHES = GRID_W * GRID_H        # 2520
IMG_W = GRID_W * PATCH_SIZE         # 960
IMG_H = GRID_H * PATCH_SIZE         # 672


class VisionEncoderExport(nn.Module):
    """Standard-ABI vision encoder: raw image → soft tokens.

    Input:  image (1, 3, IMG_H=672, IMG_W=960) float32 in [0, 1]
            (the C++ runner resizes any input image to this resolution).
    Output: soft_tokens (1, 280, 1536) float32

    Internally:
      1. Patchify image to (1, 2520, 768) in HWC pixel-major order
         (matches Gemma4ImageProcessor.preprocess output exactly).
      2. Look up the static 60×42 position grid (registered as a buffer).
      3. Run Gemma4VisionModel + Gemma4MultimodalEmbedder.
      4. Ensure batch dim is present in the output.
    """

    def __init__(self, vision_model, embed_vision):
        super().__init__()
        self.vision_model = vision_model
        self.embed_vision = embed_vision
        # Static 60x42 position grid for the kMaxPatches patches. Registered
        # as a non-persistent buffer so it ships in the .pte but isn't
        # restored from a checkpoint.
        pos_ids = torch.tensor(
            [[[x, y] for y in range(GRID_H) for x in range(GRID_W)]],
            dtype=torch.long,
        )  # (1, 2520, 2)
        self.register_buffer("position_ids", pos_ids, persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: (1, 3, IMG_H, IMG_W) in [0, 1]
        B, C, H, W = image.shape
        # Patchify with HWC pixel-major order inside each patch:
        #   (B, C, GRID_H, P, GRID_W, P)
        # → permute to (B, GRID_H, GRID_W, P, P, C)
        # → flatten last 3 dims to get (P*P*C) = 768 in pixel-major order
        x = image.view(B, C, GRID_H, PATCH_SIZE, GRID_W, PATCH_SIZE)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        pixel_values = x.view(B, N_PATCHES, PATCH_SIZE * PATCH_SIZE * C)

        # Position IDs are static — broadcast batch dim
        pos = self.position_ids.expand(B, -1, -1)

        enc_out = self.vision_model(pixel_values, pos)
        hidden_states = enc_out.last_hidden_state  # (N_soft, 768) or (B, N_soft, 768)
        soft_tokens = self.embed_vision(hidden_states)

        # Ensure batch dim is present for MultimodalPrefiller
        if soft_tokens.dim() == 2:
            soft_tokens = soft_tokens.unsqueeze(0)
        return soft_tokens


class AudioEncoderExport(nn.Module):
    """Standard-ABI audio encoder: mel spectrogram → soft tokens.

    Input:  mel (1, 128, T_mel) float32, channels-first (matches Voxtral).
            T_mel must be 200 (stride-48 conv constraint: T = 48k - 40 with k=5).
            The C++ runner pads/truncates mel from audio_preprocessor to T=200.
    Output: soft_tokens (1, 50, 1536) float32 (T_mel // 4 from internal downsampling).

    The audio_preprocessor method (separate) handles WAV→mel conversion;
    the C++ runner orchestrates: WAV → audio_preprocessor → transpose → audio_encoder.
    """

    def __init__(self, audio_model, embed_audio):
        super().__init__()
        self.audio_model = audio_model
        self.embed_audio = embed_audio

    def forward(self, mel_chw: torch.Tensor) -> torch.Tensor:
        # mel_chw: (1, 128, T_mel) channels-first → transpose to (1, T_mel, 128) for HF
        mel = mel_chw.transpose(1, 2)
        audio_out = self.audio_model(mel).last_hidden_state  # (1, T_out, 1024)
        soft_tokens = self.embed_audio(audio_out)             # (1, T_out, 1536)
        return soft_tokens
