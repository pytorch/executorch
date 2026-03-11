# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Tuple

import pytest
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


class RMSNormModel(torch.nn.Module):
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


class LaurelBlockModel(torch.nn.Module):
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


class MLPModel(torch.nn.Module):
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


class AltUpModel(torch.nn.Module):
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


class AttentionModel(torch.nn.Module):
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


class DecoderLayerModel(torch.nn.Module):
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


class TestAudioAttentionModel(torch.nn.Module):
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


class CumulativeGroupNormModel(torch.nn.Module):
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


class SSCPConvBlockModel(torch.nn.Module):
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


class SSCPConvProjectionModel(torch.nn.Module):
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


class TestConformerAttentionModel(torch.nn.Module):
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


class TestConformerFFNModel(torch.nn.Module):
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


class ConformerLightConv1dModel(torch.nn.Module):
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


class TestConformerBlockModel(torch.nn.Module):
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


class AudioEncoderModel(torch.nn.Module):
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


PipelineConfig = tuple[str, bool | None, float | None, float | None, int | None]

PIPELINE_TEST_DATA: dict[str, PipelineConfig] = {
    "tosa_fp": ("tosa_fp", None, None, None, None),
    "tosa_int": ("tosa_int", None, None, None, None),
    "vgf_no_quant": ("vgf", False, None, None, None),
    "vgf_quant": ("vgf", True, None, None, None),
}

PIPELINE_TEST_DATA_NO_TOSA_FP: dict[str, PipelineConfig] = {
    "tosa_int": ("tosa_int", None, None, None, None),
    "vgf_no_quant": ("vgf", False, None, None, None),
    "vgf_quant": ("vgf", True, None, None, None),
}

CONFORMER_INT_ATOL = 0.08
PIPELINE_TEST_DATA_CONFORMER: dict[str, PipelineConfig] = dict(PIPELINE_TEST_DATA)
PIPELINE_TEST_DATA_CONFORMER["tosa_int"] = (
    "tosa_int",
    None,
    CONFORMER_INT_ATOL,
    None,
    None,
)

# Counts are the number of delegated subgraphs in the to_edge_transform_and_lower
# stage, i.e. the occurrences of executorch_call_delegate in the EXIR graph.
# To regenerate, run a pipeline and read
# executorch.devtools.backend_debug.get_delegation_info(...).num_delegated_subgraphs
# from the to_edge_transform_and_lower artifact.
EXPECTED_DELEGATE_COUNTS: dict[str, dict[str, int]] = {
    "AltUpModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "AttentionModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "AudioEncoderModel": {
        "tosa_fp": 1,
        "tosa_int": 2,
        "vgf_no_quant": 1,
        "vgf_quant": 2,
    },
    "ConformerLightConv1dModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "CumulativeGroupNormModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "DecoderLayerModel": {
        "tosa_fp": 1,
        "tosa_int": 0,
        "vgf_no_quant": 1,
        "vgf_quant": 0,
    },
    "LaurelBlockModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "MLPModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "RMSNormModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "SSCPConvBlockModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "SSCPConvProjectionModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "TestAudioAttentionModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "TestConformerAttentionModel": {
        "tosa_fp": 2,
        "tosa_int": 3,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "TestConformerBlockModel": {
        "tosa_fp": 2,
        "tosa_int": 3,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
    "TestConformerFFNModel": {
        "tosa_fp": 1,
        "tosa_int": 1,
        "vgf_no_quant": 1,
        "vgf_quant": 1,
    },
}


def _expected_delegate_count(
    model_cls, pipeline_kind: str, quantize: bool | None
) -> int:
    model_name = model_cls.__name__
    expected = EXPECTED_DELEGATE_COUNTS.get(model_name)
    if expected is None:
        raise KeyError(f"Missing delegate counts for {model_name}")
    if pipeline_kind == "vgf":
        if quantize is None:
            raise ValueError("quantize must be set for VGF pipeline")
        return expected["vgf_quant" if quantize else "vgf_no_quant"]
    return expected[pipeline_kind]


def _run_pipeline(
    model: torch.nn.Module,
    inputs: input_t,
    pipeline_kind: str,
    quantize: bool | None,
    compare_atol: float | None,
    compare_rtol: float | None,
    compare_qtol: int | None,
    expected_delegates: int,
) -> None:
    pipeline: Any
    if pipeline_kind == "tosa_fp":
        pipeline = TosaPipelineFP[input_t](model, inputs, aten_op=[], exir_op=[])
    elif pipeline_kind == "tosa_int":
        atol = compare_atol if compare_atol is not None else 1e-3
        rtol = compare_rtol if compare_rtol is not None else 1e-3
        qtol = compare_qtol if compare_qtol is not None else 1
        pipeline = TosaPipelineINT[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            atol=atol,
            rtol=rtol,
            qtol=qtol,
        )
    elif pipeline_kind == "vgf":
        if quantize is None:
            raise ValueError("quantize must be set for VGF pipeline")
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=quantize,
        )
    else:
        raise ValueError(f"Unsupported pipeline kind: {pipeline_kind}")
    pipeline.change_args(
        "check_count.exir",
        {"torch.ops.higher_order.executorch_call_delegate": expected_delegates},
    )
    pipeline.run()


def _run_pipeline_test(model_cls, pipeline_config: PipelineConfig) -> None:
    model, inputs = model_cls.prepare_model_and_inputs()
    pipeline_kind, quantize, compare_atol, compare_rtol, compare_qtol = pipeline_config
    expected_delegates = _expected_delegate_count(model_cls, pipeline_kind, quantize)
    if pipeline_kind == "vgf" and not common.arm_executor_runner_exists(
        "vkml_emulation_layer"
    ):
        pytest.xfail(
            "Did not find build executor_runner for VKML; run setup_testing_vkml.sh."
        )
    with torch.no_grad():
        _run_pipeline(
            model,
            inputs,
            pipeline_kind,
            quantize,
            compare_atol,
            compare_rtol,
            compare_qtol,
            expected_delegates,
        )


def _run_pipeline_test_by_key(
    model_cls, pipeline_data: dict[str, PipelineConfig], key: str
) -> None:
    _run_pipeline_test(model_cls, pipeline_data[key])


def test_gemma3n_tosa_FP_AudioAttentionModel():
    _run_pipeline_test_by_key(TestAudioAttentionModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_AudioAttentionModel():
    _run_pipeline_test_by_key(
        TestAudioAttentionModel, PIPELINE_TEST_DATA_NO_TOSA_FP, "tosa_int"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_AudioAttentionModel():
    _run_pipeline_test_by_key(
        TestAudioAttentionModel, PIPELINE_TEST_DATA_NO_TOSA_FP, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_AudioAttentionModel():
    _run_pipeline_test_by_key(
        TestAudioAttentionModel, PIPELINE_TEST_DATA_NO_TOSA_FP, "vgf_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_ConformerAttentionModel():
    _run_pipeline_test_by_key(
        TestConformerAttentionModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_fp"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_ConformerAttentionModel():
    _run_pipeline_test_by_key(
        TestConformerAttentionModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_int"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_ConformerAttentionModel():
    _run_pipeline_test_by_key(
        TestConformerAttentionModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_ConformerAttentionModel():
    _run_pipeline_test_by_key(
        TestConformerAttentionModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_ConformerBlockModel():
    _run_pipeline_test_by_key(
        TestConformerBlockModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_fp"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_ConformerBlockModel():
    _run_pipeline_test_by_key(
        TestConformerBlockModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_int"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_ConformerBlockModel():
    _run_pipeline_test_by_key(
        TestConformerBlockModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_ConformerBlockModel():
    _run_pipeline_test_by_key(
        TestConformerBlockModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_ConformerFFNModel():
    _run_pipeline_test_by_key(
        TestConformerFFNModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_fp"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_ConformerFFNModel():
    _run_pipeline_test_by_key(
        TestConformerFFNModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_int"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_ConformerFFNModel():
    _run_pipeline_test_by_key(
        TestConformerFFNModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_ConformerFFNModel():
    _run_pipeline_test_by_key(
        TestConformerFFNModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_ConformerLightConv1dModel():
    _run_pipeline_test_by_key(
        ConformerLightConv1dModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_fp"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_ConformerLightConv1dModel():
    _run_pipeline_test_by_key(
        ConformerLightConv1dModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_int"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_ConformerLightConv1dModel():
    _run_pipeline_test_by_key(
        ConformerLightConv1dModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_ConformerLightConv1dModel():
    _run_pipeline_test_by_key(
        ConformerLightConv1dModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_AudioEncoderModel():
    _run_pipeline_test_by_key(
        AudioEncoderModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_fp"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_AudioEncoderModel():
    _run_pipeline_test_by_key(
        AudioEncoderModel, PIPELINE_TEST_DATA_CONFORMER, "tosa_int"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_AudioEncoderModel():
    _run_pipeline_test_by_key(
        AudioEncoderModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_AudioEncoderModel():
    _run_pipeline_test_by_key(
        AudioEncoderModel, PIPELINE_TEST_DATA_CONFORMER, "vgf_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_SSCPConvBlockModel():
    _run_pipeline_test_by_key(SSCPConvBlockModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_SSCPConvBlockModel():
    _run_pipeline_test_by_key(SSCPConvBlockModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_SSCPConvBlockModel():
    _run_pipeline_test_by_key(SSCPConvBlockModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_SSCPConvBlockModel():
    _run_pipeline_test_by_key(SSCPConvBlockModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_SSCPConvProjectionModel():
    _run_pipeline_test_by_key(SSCPConvProjectionModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_SSCPConvProjectionModel():
    _run_pipeline_test_by_key(SSCPConvProjectionModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_SSCPConvProjectionModel():
    _run_pipeline_test_by_key(
        SSCPConvProjectionModel, PIPELINE_TEST_DATA, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_SSCPConvProjectionModel():
    _run_pipeline_test_by_key(SSCPConvProjectionModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_CumulativeGroupNormModel():
    _run_pipeline_test_by_key(CumulativeGroupNormModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_CumulativeGroupNormModel():
    _run_pipeline_test_by_key(CumulativeGroupNormModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_CumulativeGroupNormModel():
    _run_pipeline_test_by_key(
        CumulativeGroupNormModel, PIPELINE_TEST_DATA, "vgf_no_quant"
    )


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_CumulativeGroupNormModel():
    _run_pipeline_test_by_key(CumulativeGroupNormModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_RMSNormModel():
    _run_pipeline_test_by_key(RMSNormModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_RMSNormModel():
    _run_pipeline_test_by_key(RMSNormModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_RMSNormModel():
    _run_pipeline_test_by_key(RMSNormModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_RMSNormModel():
    _run_pipeline_test_by_key(RMSNormModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_AltUpModel():
    _run_pipeline_test_by_key(AltUpModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_AltUpModel():
    _run_pipeline_test_by_key(AltUpModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_AltUpModel():
    _run_pipeline_test_by_key(AltUpModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_AltUpModel():
    _run_pipeline_test_by_key(AltUpModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_AttentionModel():
    _run_pipeline_test_by_key(AttentionModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_AttentionModel():
    _run_pipeline_test_by_key(AttentionModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_AttentionModel():
    _run_pipeline_test_by_key(AttentionModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_AttentionModel():
    _run_pipeline_test_by_key(AttentionModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_DecoderLayerModel():
    _run_pipeline_test_by_key(DecoderLayerModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_DecoderLayerModel():
    _run_pipeline_test_by_key(DecoderLayerModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_DecoderLayerModel():
    _run_pipeline_test_by_key(DecoderLayerModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_DecoderLayerModel():
    _run_pipeline_test_by_key(DecoderLayerModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_LaurelBlockModel():
    _run_pipeline_test_by_key(LaurelBlockModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_LaurelBlockModel():
    _run_pipeline_test_by_key(LaurelBlockModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_LaurelBlockModel():
    _run_pipeline_test_by_key(LaurelBlockModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_LaurelBlockModel():
    _run_pipeline_test_by_key(LaurelBlockModel, PIPELINE_TEST_DATA, "vgf_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_FP_MLPModel():
    _run_pipeline_test_by_key(MLPModel, PIPELINE_TEST_DATA, "tosa_fp")


@common.SkipIfNoModelConverter
def test_gemma3n_tosa_INT_MLPModel():
    _run_pipeline_test_by_key(MLPModel, PIPELINE_TEST_DATA, "tosa_int")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_no_quant_MLPModel():
    _run_pipeline_test_by_key(MLPModel, PIPELINE_TEST_DATA, "vgf_no_quant")


@common.SkipIfNoModelConverter
def test_gemma3n_vgf_quant_MLPModel():
    _run_pipeline_test_by_key(MLPModel, PIPELINE_TEST_DATA, "vgf_quant")
