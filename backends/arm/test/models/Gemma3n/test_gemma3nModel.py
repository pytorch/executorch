# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.models.Gemma3n.gemma3n_test_config import (
    get_gemma3n_audio_test_config,
    get_gemma3n_text_test_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nAudioAttention,
    Gemma3nAudioConformerAttention,
    Gemma3nAudioConformerBlock,
    Gemma3nAudioConformerFeedForward,
    Gemma3nAudioConformerLightConv1d,
    Gemma3nAudioCumulativeGroupNorm,
    Gemma3nAudioEncoder,
    Gemma3nAudioSubSampleConvProjection,
    Gemma3nRMSNorm,
    Gemma3nRotaryEmbedding,
    Gemma3nTextAltUp,
    Gemma3nTextAttention,
    Gemma3nTextDecoderLayer,
    Gemma3nTextLaurelBlock,
    Gemma3nTextMLP,
)

input_t = Tuple[torch.Tensor, ...]


class Gemma3NModule(torch.nn.Module):
    """Base class for Gemma3n modules in this test suite."""

    @classmethod
    def prepare_model_and_inputs(cls) -> Tuple[Gemma3NModule, input_t]:
        """Prepare the model and inputs for testing."""
        raise NotImplementedError("Subclasses must implement this method.")


def _make_position_ids(
    batch_size: int, seq_length: int, device: torch.device
) -> torch.Tensor:
    return torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)


def _make_rope_embeddings(
    config,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary = Gemma3nRotaryEmbedding(config)
    cos, sin = rotary(hidden_states, position_ids, layer_type=config.layer_types[0])
    return cos, sin


def _promote_gradient_clipping_buffers(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if "gradient_clipping" in submodule._buffers:
            buffer = submodule._buffers["gradient_clipping"]
            if buffer is not None:
                submodule._buffers.pop("gradient_clipping", None)
                if hasattr(submodule, "gradient_clipping"):
                    delattr(submodule, "gradient_clipping")
                submodule_any = cast(Any, submodule)
                submodule_any.gradient_clipping = float(buffer.detach().item())


class RMSNormModel(Gemma3NModule):
    """Gemma3n RMSNorm block wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 12, seq_length: int = 7, d_model: int = 768
    ) -> input_t:
        x = torch.randn(batch_size, seq_length, d_model)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_text_test_config()
        rmsnorm_model = cls(config)
        rmsnorm_model.eval()
        rmsnorm_model_inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (rmsnorm_model, rmsnorm_model_inputs)


class LaurelBlockModel(Gemma3NModule):
    """Gemma3n TextLaurelBlock wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        self.block = Gemma3nTextLaurelBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 12, seq_length: int = 7, d_model: int = 768
    ) -> input_t:
        x = torch.randn(batch_size, seq_length, d_model)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_text_test_config()
        laurel_model = cls(config)
        laurel_model.eval()
        laurel_model_inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (laurel_model, laurel_model_inputs)


class MLPModel(Gemma3NModule):
    """Gemma3n Text MLP wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        self.mlp = Gemma3nTextMLP(config, layer_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 12, seq_length: int = 7, d_model: int = 768
    ) -> input_t:
        x = torch.randn(batch_size, seq_length, d_model)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_text_test_config()
        config.hidden_size = 256
        config.intermediate_size = [512]
        config.activation_sparsity_pattern = [0.0]
        mlp_model = cls(config)
        mlp_model.eval()
        mlp_model_inputs = cls._prepare_inputs(
            batch_size=2,
            seq_length=4,
            d_model=config.hidden_size,
        )
        return (mlp_model, mlp_model_inputs)


class AltUpModel(Gemma3NModule):
    """Wrapper around Gemma3nTextAltUp."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.altup = Gemma3nTextAltUp(config)
        self.altup_active_idx = config.altup_active_idx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = hidden_states[self.altup_active_idx]
        if self.config.altup_correct_scale:
            return self.altup.scale_corrected_output(activated)
        return activated

    @staticmethod
    def _prepare_inputs(
        altup_num_inputs: int = 4,
        batch_size: int = 12,
        seq_length: int = 7,
        d_model: int = 768,
    ) -> input_t:
        x = torch.randn(altup_num_inputs, batch_size, seq_length, d_model)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_text_test_config()
        config.altup_num_inputs = 1
        config.altup_correct_scale = False
        config.hidden_size = 256
        altup_model = cls(config)
        altup_model.eval()
        altup_model_inputs = cls._prepare_inputs(
            altup_num_inputs=config.altup_num_inputs,
            batch_size=2,
            seq_length=4,
            d_model=config.hidden_size,
        )
        return (altup_model, altup_model_inputs)


class AttentionModel(Gemma3NModule):
    """Gemma3n TextAttention wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Gemma3nTextAttention(config, layer_idx=0)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        attn_output, _ = self.attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
        )
        return attn_output

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 12, seq_length: int = 7, d_model: int = 768
    ) -> input_t:
        config = get_gemma3n_text_test_config()
        hidden_states = torch.randn(batch_size, seq_length, d_model)
        position_ids = _make_position_ids(batch_size, seq_length, hidden_states.device)
        cos, sin = _make_rope_embeddings(config, hidden_states, position_ids)
        return (hidden_states, cos, sin)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_text_test_config()
        attn_model = cls(config)
        attn_model.eval()
        attn_model_inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (attn_model, attn_model_inputs)


class DecoderLayerModel(Gemma3NModule):
    """Gemma3n TextDecoderLayer wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        self.layer = Gemma3nTextDecoderLayer(config, layer_idx=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        del position_ids, cos, sin
        active = hidden_states[self.layer.config.altup_active_idx]
        gated = self.layer.per_layer_input_gate(active)
        gated = self.layer.act_fn(gated)
        gated = gated * per_layer_input
        projected = self.layer.per_layer_projection(gated)
        projected = self.layer.post_per_layer_input_norm(projected)
        output = hidden_states.clone()
        output[self.layer.config.altup_active_idx] = projected
        return output

    @staticmethod
    def _prepare_inputs(
        altup_num_inputs: int = 4,
        batch_size: int = 12,
        seq_length: int = 7,
        d_model: int = 768,
        per_layer_input_dim: int = 256,
    ) -> input_t:
        config = get_gemma3n_text_test_config()
        hidden_states = torch.randn(altup_num_inputs, batch_size, seq_length, d_model)
        active_prediction = hidden_states[config.altup_active_idx]
        position_ids = _make_position_ids(batch_size, seq_length, hidden_states.device)
        cos, sin = _make_rope_embeddings(config, active_prediction, position_ids)
        per_layer_input = torch.randn(
            batch_size, seq_length, per_layer_input_dim, device=hidden_states.device
        ).type_as(hidden_states)
        return (hidden_states, per_layer_input, position_ids, cos, sin)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_text_test_config()
        config.altup_num_inputs = 1
        config.altup_correct_scale = False
        config.hidden_size = 256
        config.hidden_size_per_layer_input = 64
        decoder_layer = cls(config)
        decoder_layer.eval()
        decoder_inputs = cls._prepare_inputs(
            altup_num_inputs=config.altup_num_inputs,
            batch_size=2,
            seq_length=4,
            d_model=config.hidden_size,
            per_layer_input_dim=config.hidden_size_per_layer_input,
        )
        return (decoder_layer, decoder_inputs)


class TestAudioAttentionModel(Gemma3NModule):
    """Wrap Gemma3nAudioAttention in a simple forward."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Gemma3nAudioAttention(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        b, t, _ = x.shape
        projected = self.attn.q_proj(x)
        return projected.reshape(b, t, self.attn.num_heads, self.attn.head_dim)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 2, num_frames: int = 128, d_model: int = 256
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, d_model)
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        return (x, mask)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        config._attn_implementation = "eager"
        model = cls(config)
        _promote_gradient_clipping_buffers(model)
        model.eval()
        inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (model, inputs)


class CumulativeGroupNormModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioCumulativeGroupNorm."""

    def __init__(self, num_channels: int = 512, eps: float = 1e-05) -> None:
        super().__init__()
        self.norm = Gemma3nAudioCumulativeGroupNorm(
            num_channels=num_channels,
            feature_dims=(),
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 4, num_frames: int = 256, num_channels: int = 512
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, num_channels)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        num_channels = 512
        cg_norm_model = cls(num_channels=num_channels, eps=1e-05)
        cg_norm_model.eval()
        cg_norm_inputs = cls._prepare_inputs(num_channels=num_channels)
        return (cg_norm_model, cg_norm_inputs)


class SSCPConvBlockModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioSSCPConvBlock (first block)."""

    def __init__(self, config) -> None:
        super().__init__()
        projection = Gemma3nAudioSubSampleConvProjection(config)
        self.block = projection.conv_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 4, num_frames: int = 256, input_feat_size: int = 128
    ) -> input_t:
        x = torch.randn(batch_size, 1, num_frames, input_feat_size)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        sscp_block = cls(config)
        sscp_block.block.norm = torch.nn.Identity()
        sscp_block.eval()
        sscp_inputs = cls._prepare_inputs(input_feat_size=config.input_feat_size)
        return (sscp_block, sscp_inputs)


class SSCPConvProjectionModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioSubSampleConvProjection."""

    def __init__(self, config) -> None:
        super().__init__()
        self.proj = Gemma3nAudioSubSampleConvProjection(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 4, num_frames: int = 256, input_feat_size: int = 128
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, input_feat_size)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        sscp_proj_model = cls(config)
        config.hidden_size = config.input_feat_size
        sscp_proj_model.proj.conv_0 = torch.nn.Identity()
        sscp_proj_model.proj.conv_1 = torch.nn.Identity()
        sscp_proj_model.proj.input_proj_linear = torch.nn.Identity()
        sscp_proj_model.eval()
        sscp_proj_inputs = cls._prepare_inputs(input_feat_size=config.input_feat_size)
        return (sscp_proj_model, sscp_proj_inputs)


class TestConformerAttentionModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioConformerAttention."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Gemma3nAudioConformerAttention(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.attn(x, mask)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 2, num_frames: int = 64, d_model: int = 256
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, d_model)
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        return (x, mask)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        config._attn_implementation = "eager"
        model = cls(config)
        _promote_gradient_clipping_buffers(model)
        model.eval()
        inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (model, inputs)


class TestConformerFFNModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioConformerFeedForward."""

    def __init__(self, config) -> None:
        super().__init__()
        self.ffn = Gemma3nAudioConformerFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 2, num_frames: int = 64, d_model: int = 256
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, d_model)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        config._attn_implementation = "eager"
        model = cls(config)
        _promote_gradient_clipping_buffers(model)
        model.eval()
        inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (model, inputs)


class ConformerLightConv1dModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioConformerLightConv1d."""

    def __init__(self, config) -> None:
        super().__init__()
        self.conv = Gemma3nAudioConformerLightConv1d(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 4, num_frames: int = 256, d_model: int = 256
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, d_model)
        return (x,)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        config.conf_conv_kernel_size = 1
        lightconv = cls(config)
        lightconv.conv.depthwise_conv1d = torch.nn.Identity()
        _promote_gradient_clipping_buffers(lightconv)
        lightconv.eval()
        lightconv_inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (lightconv, lightconv_inputs)


class TestConformerBlockModel(Gemma3NModule):
    """Wrap a single Gemma3nAudioConformerBlock."""

    def __init__(self, config) -> None:
        super().__init__()
        self.block = Gemma3nAudioConformerBlock(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x, mask)

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 2, num_frames: int = 64, d_model: int = 256
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, d_model)
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        return (x, mask)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        config._attn_implementation = "eager"
        config.conf_conv_kernel_size = 1
        model = cls(config)
        model.block.lconv1d = torch.nn.Identity()
        _promote_gradient_clipping_buffers(model)
        model.eval()
        inputs = cls._prepare_inputs(d_model=config.hidden_size)
        return (model, inputs)


class AudioEncoderModel(Gemma3NModule):
    """Wrapper for Gemma3nAudioEncoder."""

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = Gemma3nAudioEncoder(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encodings, _ = self.encoder(x, mask)
        return encodings

    @staticmethod
    def _prepare_inputs(
        batch_size: int = 4, num_frames: int = 256, input_feat_size: int = 128
    ) -> input_t:
        x = torch.randn(batch_size, num_frames, input_feat_size)
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        return (x, mask)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = get_gemma3n_audio_test_config()
        encoder = cls(config)
        encoder.encoder.subsample_conv_projection = torch.nn.Identity()
        encoder.encoder.conformer = torch.nn.ModuleList()
        _promote_gradient_clipping_buffers(encoder)
        encoder.eval()
        encoder_inputs = cls._prepare_inputs(input_feat_size=config.input_feat_size)
        return (encoder, encoder_inputs)


@dataclass(frozen=True)
class Gemma3nTestCase:
    model_cls: type[Gemma3NModule]
    call_delegates: int = 1
    atol: float = 1e-3
    rtol: float = 1e-3
    qtol: int = 1
    frobenius_threshold: float | None = 0.15
    cosine_threshold: float | None = 0.9
    run_on_vulkan_runtime: bool = True
    check_exir_quant_nodes: bool = True


TOSA_FP_TEST_CASES: dict[str, Gemma3nTestCase] = {
    "audio_attention": Gemma3nTestCase(
        model_cls=TestAudioAttentionModel,
    ),
    "conformer_attention": Gemma3nTestCase(
        model_cls=TestConformerAttentionModel,
        call_delegates=2,
    ),
    "conformer_block": Gemma3nTestCase(
        model_cls=TestConformerBlockModel,
        call_delegates=2,
    ),
    "conformer_ffn": Gemma3nTestCase(model_cls=TestConformerFFNModel),
    "conformer_light_conv1d": Gemma3nTestCase(model_cls=ConformerLightConv1dModel),
    "audio_encoder": Gemma3nTestCase(model_cls=AudioEncoderModel),
    "sscp_conv_block": Gemma3nTestCase(model_cls=SSCPConvBlockModel),
    "sscp_conv_projection": Gemma3nTestCase(model_cls=SSCPConvProjectionModel),
    "cumulative_group_norm": Gemma3nTestCase(model_cls=CumulativeGroupNormModel),
    "rms_norm": Gemma3nTestCase(model_cls=RMSNormModel),
    "altup": Gemma3nTestCase(model_cls=AltUpModel),
    "attention": Gemma3nTestCase(model_cls=AttentionModel),
    "decoder_layer": Gemma3nTestCase(model_cls=DecoderLayerModel),
    "laurel_block": Gemma3nTestCase(model_cls=LaurelBlockModel),
    "mlp": Gemma3nTestCase(model_cls=MLPModel),
}
##TODO (MLETORCH-1951): xfail/high atol/rtol

TOSA_INT_TEST_CASES: dict[str, Gemma3nTestCase] = {
    "audio_attention": Gemma3nTestCase(model_cls=TestAudioAttentionModel),
    "conformer_attention": Gemma3nTestCase(
        model_cls=TestConformerAttentionModel,
        call_delegates=3,
        atol=0.08,
        frobenius_threshold=0.2,
    ),
    "conformer_block": Gemma3nTestCase(
        model_cls=TestConformerBlockModel,
        call_delegates=3,
        atol=0.6,
        frobenius_threshold=0.2,
    ),
    "conformer_ffn": Gemma3nTestCase(model_cls=TestConformerFFNModel, atol=0.08),
    "conformer_light_conv1d": Gemma3nTestCase(
        model_cls=ConformerLightConv1dModel,
        atol=0.08,
    ),
    "audio_encoder": Gemma3nTestCase(
        model_cls=AudioEncoderModel,
        call_delegates=1,
        atol=0.08,
    ),
    "sscp_conv_block": Gemma3nTestCase(model_cls=SSCPConvBlockModel),
    "sscp_conv_projection": Gemma3nTestCase(model_cls=SSCPConvProjectionModel),
    "cumulative_group_norm": Gemma3nTestCase(
        model_cls=CumulativeGroupNormModel,
        atol=0.04,
    ),
    "rms_norm": Gemma3nTestCase(model_cls=RMSNormModel),
    "altup": Gemma3nTestCase(model_cls=AltUpModel),
    "attention": Gemma3nTestCase(model_cls=AttentionModel),
    "decoder_layer": Gemma3nTestCase(
        model_cls=DecoderLayerModel,
        call_delegates=0,
        frobenius_threshold=None,
        cosine_threshold=None,
    ),
    "laurel_block": Gemma3nTestCase(model_cls=LaurelBlockModel, qtol=2),
    "mlp": Gemma3nTestCase(model_cls=MLPModel),
}

VGF_NO_QUANT_TEST_CASES: dict[str, Gemma3nTestCase] = {
    "audio_attention": Gemma3nTestCase(model_cls=TestAudioAttentionModel),
    "conformer_attention": Gemma3nTestCase(model_cls=TestConformerAttentionModel),
    "conformer_block": Gemma3nTestCase(model_cls=TestConformerBlockModel),
    "conformer_ffn": Gemma3nTestCase(model_cls=TestConformerFFNModel),
    "conformer_light_conv1d": Gemma3nTestCase(model_cls=ConformerLightConv1dModel),
    "audio_encoder": Gemma3nTestCase(model_cls=AudioEncoderModel),
    "sscp_conv_block": Gemma3nTestCase(model_cls=SSCPConvBlockModel),
    "sscp_conv_projection": Gemma3nTestCase(
        model_cls=SSCPConvProjectionModel,
        run_on_vulkan_runtime=False,
    ),
    "cumulative_group_norm": Gemma3nTestCase(model_cls=CumulativeGroupNormModel),
    "rms_norm": Gemma3nTestCase(model_cls=RMSNormModel),
    "altup": Gemma3nTestCase(model_cls=AltUpModel),
    "attention": Gemma3nTestCase(model_cls=AttentionModel),
    "decoder_layer": Gemma3nTestCase(model_cls=DecoderLayerModel),
    "laurel_block": Gemma3nTestCase(model_cls=LaurelBlockModel),
    "mlp": Gemma3nTestCase(model_cls=MLPModel),
}

VGF_QUANT_TEST_CASES: dict[str, Gemma3nTestCase] = {
    "audio_attention": Gemma3nTestCase(model_cls=TestAudioAttentionModel),
    "conformer_attention": Gemma3nTestCase(
        model_cls=TestConformerAttentionModel, qtol=2
    ),
    "conformer_block": Gemma3nTestCase(
        model_cls=TestConformerBlockModel,
        atol=0.6,
    ),
    "conformer_ffn": Gemma3nTestCase(model_cls=TestConformerFFNModel),
    "conformer_light_conv1d": Gemma3nTestCase(model_cls=ConformerLightConv1dModel),
    "audio_encoder": Gemma3nTestCase(model_cls=AudioEncoderModel, call_delegates=1),
    "sscp_conv_block": Gemma3nTestCase(model_cls=SSCPConvBlockModel),
    "sscp_conv_projection": Gemma3nTestCase(model_cls=SSCPConvProjectionModel),
    "cumulative_group_norm": Gemma3nTestCase(
        model_cls=CumulativeGroupNormModel,
        atol=0.04,
    ),
    "rms_norm": Gemma3nTestCase(model_cls=RMSNormModel),
    "altup": Gemma3nTestCase(model_cls=AltUpModel),
    "attention": Gemma3nTestCase(model_cls=AttentionModel, atol=0.04),
    "decoder_layer": Gemma3nTestCase(
        model_cls=DecoderLayerModel,
        call_delegates=0,
        check_exir_quant_nodes=False,
    ),
    "laurel_block": Gemma3nTestCase(model_cls=LaurelBlockModel, qtol=2),
    "mlp": Gemma3nTestCase(model_cls=MLPModel),
}


@common.parametrize("test_case", TOSA_FP_TEST_CASES)
def test_gemma3n_tosa_FP(test_case: Gemma3nTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](model, inputs, aten_op=[], exir_op=[])
        pipeline.change_args(
            "check_count.exir",
            {
                "torch.ops.higher_order.executorch_call_delegate": test_case.call_delegates
            },
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize(
    "test_case",
    TOSA_INT_TEST_CASES,
    xfails={
        "decoder_layer": "No TOSA delegate generated for decoder_layer on INT path."
    },
)
def test_gemma3n_tosa_INT(test_case: Gemma3nTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            atol=test_case.atol,
            rtol=test_case.rtol,
            qtol=test_case.qtol,
            frobenius_threshold=test_case.frobenius_threshold,
            cosine_threshold=test_case.cosine_threshold,
        )
        pipeline.change_args(
            "check_count.exir",
            {
                "torch.ops.higher_order.executorch_call_delegate": test_case.call_delegates
            },
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_case", VGF_NO_QUANT_TEST_CASES)
def test_gemma3n_vgf_no_quant(test_case: Gemma3nTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=False,
            run_on_vulkan_runtime=test_case.run_on_vulkan_runtime,
            atol=test_case.atol,
            rtol=test_case.rtol,
            qtol=test_case.qtol,
        )
        if not test_case.check_exir_quant_nodes and pipeline.has_stage(
            "check_not.exir_quant_nodes"
        ):
            pipeline.pop_stage("check_not.exir_quant_nodes")
        pipeline.change_args(
            "check_count.exir",
            {
                "torch.ops.higher_order.executorch_call_delegate": test_case.call_delegates
            },
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_case", VGF_QUANT_TEST_CASES)
def test_gemma3n_vgf_quant(test_case: Gemma3nTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=True,
            run_on_vulkan_runtime=test_case.run_on_vulkan_runtime,
            atol=test_case.atol,
            rtol=test_case.rtol,
            qtol=test_case.qtol,
        )
        if not test_case.check_exir_quant_nodes and pipeline.has_stage(
            "check_not.exir_quant_nodes"
        ):
            pipeline.pop_stage("check_not.exir_quant_nodes")
        pipeline.change_args(
            "check_count.exir",
            {
                "torch.ops.higher_order.executorch_call_delegate": test_case.call_delegates
            },
        )
        pipeline.run()
