#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Gemma 4 E2B/E4B as a single .pte with 4 methods:
  speech_transform, audio_encoder, vision_encoder, text_decoder

Usage:
    # E2B (default):
    buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
        --checkpoint_path /tmp/gemma4-e2b-it \
        --output_path /tmp/gemma4.pte

    # E4B:
    buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
        --checkpoint_path /tmp/gemma4-e4b-it \
        --variant e4b \
        --output_path /tmp/gemma4_e4b.pte

"""

import argparse
import functools
import gc
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _export_speech_transform(checkpoint_path: str):
    """Export speech transform model. Returns ExportedProgram."""
    from executorch.examples.models.gemma4.speech_transform import (
        Gemma4SpeechTransformModel,
    )

    # Load audio config from HF checkpoint if available
    kwargs = {}
    try:
        import json
        import os

        # Audio feature extractor config is in processor_config.json (not config.json)
        processor_path = os.path.join(checkpoint_path, "processor_config.json")
        if os.path.exists(processor_path):
            with open(processor_path, "r") as f:
                processor_config = json.load(f)
            audio_fe = processor_config.get("feature_extractor", {})
            if audio_fe:
                field_map = {
                    "sampling_rate": "sample_rate",
                    "feature_size": "n_mels",
                    "min_frequency": "f_min",
                    "max_frequency": "f_max",
                    "mel_floor": "mel_floor",
                    "input_scale_factor": "input_scale_factor",
                    "frame_length": "frame_length",
                    "hop_length": "hop_length",
                }
                for hf_key, our_key in field_map.items():
                    if hf_key in audio_fe:
                        kwargs[our_key] = audio_fe[hf_key]
                logger.info(f"Speech transform config from checkpoint: {kwargs}")
    except Exception as e:
        logger.warning(
            f"Could not load audio config from checkpoint: {e}, using defaults"
        )

    model = Gemma4SpeechTransformModel(**kwargs)
    model.eval()

    sample_rate = kwargs.get("sample_rate", 16000)
    num_samples = sample_rate * 30  # 30s
    example = torch.randn(num_samples)
    min_samples = model.frame_length + model.hop_length + 1
    waveform_dim = torch.export.Dim("waveform_length", min=min_samples, max=960000)

    with torch.no_grad():
        ep = torch.export.export(
            model, (example,), dynamic_shapes={"waveform": {0: waveform_dim}}
        )

    logger.info("Speech transform exported")
    del model
    gc.collect()
    return ep


class _HFAudioEncoderWithProjection(torch.nn.Module):
    """Thin wrapper combining HF's Gemma4AudioModel + Gemma4MultimodalEmbedder.

    Loads audio_tower and embed_audio from HF's Gemma4ForConditionalGeneration,
    combines them into a single forward() that maps mel features to text-decoder-
    ready embeddings.
    """

    def __init__(self, audio_tower, embed_audio):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process mel spectrogram to LM-ready embeddings.

        Args:
            features: Mel spectrogram [batch, num_frames, n_mels]
            mask: Attention mask [batch, num_frames] (True=valid, False=padded)

        Returns:
            Tuple of (embeddings, output_mask):
            - embeddings: [batch, num_audio_tokens, hidden_size]
            - output_mask: [batch, num_audio_tokens] (True=valid)
        """
        encoder_output = self.audio_tower(features, attention_mask=mask)
        hidden = encoder_output.last_hidden_state
        output_mask = encoder_output.attention_mask
        return self.embed_audio(hidden), output_mask


def _patch_hf_audio_for_export():
    """Monkey-patch HF Gemma4 audio modules to fix torch.export blockers.

    Requires transformers >= 5.2.0 with Gemma4 model support.
    Build with: buck2 build ... --config ovr_config//third-party/transformers-stack/constraints:5.5.0=...
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        raise ImportError(
            "transformers.models.gemma4 not found. Requires transformers >= 5.2.0. "
            "If using buck2, add the constraint: "
            "--config ovr_config//third-party/transformers-stack/constraints:5.5.0="
            "ovr_config//third-party/transformers-stack/constraints:5.5.0"
        )

    # Patch 1: @torch.no_grad() on Gemma4AudioRelPositionalEncoding.forward
    # is graph-breaking for torch.export. Remove by unwrapping the decorator.
    # If HF removes the decorator in a future version, this is a safe no-op.
    cls = modeling_gemma4.Gemma4AudioRelPositionalEncoding
    if hasattr(cls.forward, "__wrapped__"):
        cls.forward = cls.forward.__wrapped__

    # Patch 2: @cached_property on Gemma4AudioCausalConv1d.left_pad creates a
    # lazy attribute incompatible with torch.export tracing. Convert to a regular
    # property (the value is a trivial integer computation).
    # If HF changes this in a future version, this is a safe no-op.
    if hasattr(modeling_gemma4, "Gemma4AudioCausalConv1d"):
        conv_cls = modeling_gemma4.Gemma4AudioCausalConv1d
        cached = conv_cls.__dict__.get("left_pad")
        if isinstance(cached, functools.cached_property):
            conv_cls.left_pad = property(cached.func)


def _load_hf_conditional_model(checkpoint_path: str):
    """Load HF Gemma4ForConditionalGeneration, handling quantization_config removal.

    Returns the loaded model. Caller is responsible for extracting components
    and deleting the model when done.
    """
    import json
    import os
    import tempfile

    from transformers import AutoConfig

    config_path = os.path.join(checkpoint_path, "config.json")
    patched_path = checkpoint_path
    temp_dir = None

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if "quantization_config" in config_dict:
            logger.info("  Patching config.json to remove quantization_config...")
            temp_dir = tempfile.mkdtemp(prefix="gemma4_export_")
            patched_path = temp_dir
            for item in os.listdir(checkpoint_path):
                src = os.path.join(checkpoint_path, item)
                dst = os.path.join(temp_dir, item)
                if item != "config.json":
                    os.symlink(src, dst)
            patched_config = {
                k: v for k, v in config_dict.items() if k != "quantization_config"
            }
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(patched_config, f, indent=2)

    try:
        config = AutoConfig.from_pretrained(patched_path, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            config.quantization_config = None

        try:
            from transformers import Gemma4ForConditionalGeneration
        except ImportError:
            raise ImportError(
                "Gemma4ForConditionalGeneration not found. Requires transformers >= 5.2.0."
            )

        logger.info("  Loading HF Gemma4ForConditionalGeneration...")
        with torch.device("cpu"):
            model = Gemma4ForConditionalGeneration.from_pretrained(
                patched_path,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        logger.info("  Model loaded")
        return model
    finally:
        if temp_dir is not None:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def _load_hf_audio_encoder(checkpoint_path: str):
    """Load HF Gemma4 audio encoder from checkpoint, returning the wrapper module."""
    model = _load_hf_conditional_model(checkpoint_path)

    audio_tower = model.model.audio_tower
    embed_audio = model.model.embed_audio
    logger.info(f"  Extracted audio_tower: {type(audio_tower).__name__}")
    logger.info(f"  Extracted embed_audio: {type(embed_audio).__name__}")

    wrapper = _HFAudioEncoderWithProjection(audio_tower, embed_audio)
    wrapper.eval()

    del model
    gc.collect()
    return wrapper


def _export_audio_encoder(checkpoint_path: str, quantize: str, group_size: int = 128):
    """Export audio encoder using HF's Gemma4AudioModel. Returns ExportedProgram."""
    # Patch HF modules for export compatibility
    _patch_hf_audio_for_export()

    # Load audio encoder from HF checkpoint
    model = _load_hf_audio_encoder(checkpoint_path)

    if quantize != "none":
        from executorch.examples.models.gemma4.quant_utils import (
            apply_linear_quantization,
        )

        model = apply_linear_quantization(model, quantize, group_size=group_size)
        model.eval()

    # Dynamic shapes: num_frames = 48*k - 25
    _num_frames = torch.export.Dim("_num_frames", min=2, max=63)
    num_frames_dim = 48 * _num_frames - 25
    valid_frames = 48 * 63 - 25  # 2999
    example_features = torch.randn(1, valid_frames, 128)
    example_mask = torch.ones(1, valid_frames, dtype=torch.bool)

    with torch.no_grad():
        ep = torch.export.export(
            model,
            (example_features, example_mask),
            dynamic_shapes={
                "features": {1: num_frames_dim},
                "mask": {1: num_frames_dim},
            },
        )

    logger.info("Audio encoder exported")
    del model
    gc.collect()
    return ep


class _HFVisionEncoderWithProjection(torch.nn.Module):
    """Thin wrapper combining HF's Gemma4VisionModel + Gemma4MultimodalEmbedder.

    Calls sub-components directly instead of vision_tower.forward() because
    the top-level forward does dynamic boolean indexing (hidden_states[mask])
    to strip padding, which is incompatible with torch.export.
    """

    def __init__(self, vision_tower, embed_vision):
        super().__init__()
        self.patch_embedder = vision_tower.patch_embedder
        self.encoder = vision_tower.encoder
        self.pooler = vision_tower.pooler
        self.embed_vision = embed_vision
        self.standardize = getattr(vision_tower.config, "standardize", False)
        if self.standardize:
            self.std_bias = vision_tower.std_bias
            self.std_scale = vision_tower.std_scale
        self.pooling_kernel_size = vision_tower.config.pooling_kernel_size

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process pre-patchified image to LM-ready embeddings.

        Args:
            pixel_values: [batch, num_patches, patch_dim] (pre-patchified, [0,1])
            pixel_position_ids: [batch, num_patches, 2] (x,y coords, -1=padding)

        Returns:
            Tuple of (embeddings, output_mask):
            - embeddings: [batch, output_length, text_hidden_size]
            - output_mask: [batch, output_length] (True=valid)
        """
        pks = self.pooling_kernel_size
        output_length = pixel_values.shape[1] // (pks * pks)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        encoder_output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoder_output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
            hidden_states = hidden_states.masked_fill(~pooler_mask.unsqueeze(-1), 0.0)

        embeddings = self.embed_vision(hidden_states)
        return embeddings, pooler_mask


def _load_hf_vision_encoder(checkpoint_path: str):
    """Load HF Gemma4 vision encoder from checkpoint, returning the wrapper module."""
    model = _load_hf_conditional_model(checkpoint_path)

    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision
    logger.info(f"  Extracted vision_tower: {type(vision_tower).__name__}")
    logger.info(f"  Extracted embed_vision: {type(embed_vision).__name__}")

    wrapper = _HFVisionEncoderWithProjection(vision_tower, embed_vision)
    wrapper.eval()

    del model
    gc.collect()
    return wrapper


def _quantize_position_embedding_table(model):
    """Quantize position embedding table from fp32 to int8 per-channel.

    The table is (2, 10240, 768) = 60 MB at fp32 -> 15 MB at int8 + scales.
    Quality impact is negligible (cosine sim > 0.999999 vs fp32 reference).

    Replaces the fp32 nn.Parameter with int8 data + fp32 scale, and patches
    the patch embedder's _position_embeddings() to dequantize before matmul.
    """
    pet = model.patch_embedder.position_embedding_table
    original_mb = pet.numel() * 4 / (1024 * 1024)

    # Per-channel quantization along last dim (768 channels)
    scale = pet.data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    quantized = torch.round(pet.data / scale).clamp(-128, 127).to(torch.int8)

    # Replace the parameter with int8 data + scale buffers
    del model.patch_embedder.position_embedding_table
    model.patch_embedder.register_buffer("_pet_int8", quantized)
    model.patch_embedder.register_buffer("_pet_scale", scale)

    new_mb = (quantized.numel() + scale.numel() * 4) / (1024 * 1024)
    logger.info(
        f"  Position embedding table: fp32 -> int8 per-channel ({original_mb:.0f} MB -> {new_mb:.0f} MB)"
    )

    # Patch _position_embeddings to dequantize on the fly
    import types

    def _position_embeddings_int8(self, pixel_position_ids, padding_positions):
        # Dequantize: int8 * scale -> fp32
        table = self._pet_int8.float() * self._pet_scale
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(clamped, num_classes=table.shape[1])
        one_hot = one_hot.permute(0, 2, 1, 3).float()
        pos_emb = one_hot @ table
        pos_emb = pos_emb.sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, pos_emb)

    model.patch_embedder._position_embeddings = types.MethodType(
        _position_embeddings_int8, model.patch_embedder
    )


def _export_vision_encoder(
    checkpoint_path: str, quantize: str, group_size: int = 128, max_patches: int = 1260
):
    """Export vision encoder using HF's Gemma4VisionModel. Returns ExportedProgram."""
    model = _load_hf_vision_encoder(checkpoint_path)

    # Quantize position embedding table (60 MB fp32 -> 15 MB int8)
    _quantize_position_embedding_table(model)

    if quantize != "none":
        from executorch.examples.models.gemma4.quant_utils import (
            apply_linear_quantization,
        )

        model = apply_linear_quantization(model, quantize, group_size=group_size)
        model.eval()

    # Dynamic shapes: num_patches must be divisible by pooling_kernel_size^2 (=9)
    _num_groups = torch.export.Dim("_num_groups", min=1, max=max_patches // 9)
    num_patches_dim = 9 * _num_groups

    example_pixels = torch.randn(1, max_patches, 3 * 16 * 16)
    example_positions = torch.zeros(1, max_patches, 2, dtype=torch.long)

    with torch.no_grad():
        ep = torch.export.export(
            model,
            (example_pixels, example_positions),
            dynamic_shapes={
                "pixel_values": {1: num_patches_dim},
                "pixel_position_ids": {1: num_patches_dim},
            },
        )

    logger.info("Vision encoder exported")
    del model
    gc.collect()
    return ep


def _export_text_decoder(
    checkpoint_path: str,
    quantize: str,
    max_seq_len: int,
    group_size: int = 128,
    tied_embedding: bool = False,
    variant: str = "e2b",
    quantize_kv_cache: bool = False,
    use_custom_sdpa: bool = True,
):
    """Export text decoder. Returns ExportedProgram."""
    from executorch.examples.models.gemma4.quant_utils import (
        apply_embedding_quantization,
        apply_linear_quantization,
        parse_quantize,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
        Gemma4Config,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_model import Gemma4Model

    config = Gemma4Config.from_config(variant)
    config.use_kv_cache = True
    config.max_seq_len = max_seq_len
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = use_custom_sdpa

    if use_custom_sdpa:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        logger.info("Custom SDPA enabled (tiled flash attention)")

    model_wrapper = Gemma4Model(
        config=config, checkpoint_path=checkpoint_path, dtype=torch.float32
    )
    model = model_wrapper.get_eager_model()
    model.eval()

    linear_quant, emb_quant = parse_quantize(quantize)
    if emb_quant and linear_quant and tied_embedding:
        # Share embed_tokens + lm_head via TiedEmbeddingQuantizer.
        # weight_dtype matches the embedding quantization (emb8 -> int8, emb4 -> int4).
        # lm_head is tied to embed_tokens — linear quantization skips it.
        from torchao.prototype.quantization.embedding.api import TiedEmbeddingQuantizer
        from torchao.quantization.granularity import PerAxis

        weight_dtype = torch.int4 if emb_quant == "emb4" else torch.int8
        logger.info(f"Applying tied embedding (embed_tokens + lm_head, {emb_quant})...")
        TiedEmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=PerAxis(0),
        ).quantize(
            model,
            embedding_to_unembedding={
                "model.self_decoder.embed_tokens": "model.lm_head",
            },
        )
        # Quantize embed_tokens_per_layer with llama's EmbeddingQuantHandler
        # (TorchAO's EmbeddingQuantizer overflows INT32 for Gemma 4's large per-layer embedding)
        model = apply_embedding_quantization(model, emb_quant)
        model.eval()
    else:
        if emb_quant:
            model = apply_embedding_quantization(model, emb_quant)
            model.eval()
    if quantize_kv_cache:
        from executorch.examples.models.gemma4.text_decoder.gemma4_attention import (
            replace_kv_cache_with_quantized_kv_cache,
        )

        logger.info("Replacing KV cache with INT8 quantized KV cache...")
        model = replace_kv_cache_with_quantized_kv_cache(
            model, use_custom_sdpa=use_custom_sdpa
        )
        model.eval()

    if linear_quant:
        model = apply_linear_quantization(model, linear_quant, group_size=group_size)
        model.eval()

    # Export with audio embeds + dynamic shapes
    example_inputs = model_wrapper.get_example_inputs_with_audio(seq_len=770)
    dynamic_shapes = model_wrapper.get_dynamic_shapes(with_audio_embeds=True)

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        with torch.no_grad():
            ep = torch.export.export(
                model,
                (example_inputs[0],),
                kwargs={
                    "input_pos": example_inputs[1],
                    "inputs_embeds": example_inputs[2],
                },
                dynamic_shapes=dynamic_shapes,
            )

    logger.info("Text decoder exported")
    del model, model_wrapper
    gc.collect()
    return ep


def _export_components(
    checkpoint_path: str,
    text_quantize: str,
    audio_quantize: str,
    vision_quantize: str,
    max_seq_len: int,
    max_patches: int,
    group_size: int,
    tied_embedding: bool,
    variant: str,
    quantize_kv_cache: bool,
    include_audio: bool,
    include_vision: bool,
    use_custom_sdpa: bool,
) -> dict:
    """Export each requested component to an ExportedProgram."""
    components = []
    if include_audio:
        components += ["speech_transform", "audio_encoder"]
    if include_vision:
        components += ["vision_encoder"]
    components += ["text_decoder"]
    num_components = len(components)
    logger.info(f"Exporting {num_components} methods: {', '.join(components)}")

    programs = {}
    step = 0

    if include_audio:
        step += 1
        logger.info(f"[{step}/{num_components}] Exporting speech transform...")
        programs["speech_transform"] = _export_speech_transform(checkpoint_path)

        step += 1
        logger.info(f"[{step}/{num_components}] Exporting audio encoder...")
        programs["audio_encoder"] = _export_audio_encoder(
            checkpoint_path, audio_quantize, group_size=group_size
        )

    if include_vision:
        step += 1
        logger.info(f"[{step}/{num_components}] Exporting vision encoder...")
        programs["vision_encoder"] = _export_vision_encoder(
            checkpoint_path,
            vision_quantize,
            group_size=group_size,
            max_patches=max_patches,
        )

    step += 1
    logger.info(f"[{step}/{num_components}] Exporting text decoder...")
    programs["text_decoder"] = _export_text_decoder(
        checkpoint_path,
        text_quantize,
        max_seq_len,
        group_size=group_size,
        tied_embedding=tied_embedding,
        variant=variant,
        quantize_kv_cache=quantize_kv_cache,
        use_custom_sdpa=use_custom_sdpa,
    )

    return programs


def _build_partitioners(
    include_audio: bool,
    include_vision: bool,
    audio_quantize: str,
    vision_quantize: str,
    text_quantize: str,
) -> dict:
    """Build per-method XNNPACK partitioner lists."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
    )

    xnnpack_quant = XnnpackDynamicallyQuantizedPartitioner()
    xnnpack = XnnpackPartitioner()

    def _for(quantize: str) -> list:
        return [xnnpack_quant, xnnpack] if quantize != "none" else [xnnpack]

    partitioners = {}
    if include_audio:
        partitioners["speech_transform"] = [xnnpack]
        partitioners["audio_encoder"] = _for(audio_quantize)
    if include_vision:
        partitioners["vision_encoder"] = _for(vision_quantize)
    partitioners["text_decoder"] = _for(text_quantize)
    return partitioners


def _build_transform_passes(include_audio: bool, include_vision: bool) -> dict:
    """Build per-method transform passes (text decoder gets bitwise lowering)."""
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.pass_base import ExportPass

    class _ReplaceBitwiseScalarPass(ExportPass):
        def __init__(self, ops_map):
            self.ops_map = ops_map
            super().__init__()

        def call_operator(self, op, args, kwargs, meta):
            if op not in self.ops_map:
                return super().call_operator(op, args, kwargs, meta)
            full_op = exir_ops.edge.aten.full.default
            scalar_as_tensor = super().call_operator(
                full_op,
                ((1,), float(args[1])),
                {
                    "dtype": args[0].to_tensor().dtype,
                    "device": args[0].to_tensor().device,
                },
                meta,
            )
            return super().call_operator(
                self.ops_map[op], (args[0], scalar_as_tensor, *args[2:]), kwargs, meta
            )

    bitwise_ops = {
        exir_ops.edge.aten.__rshift__.Scalar: exir_ops.edge.aten.bitwise_right_shift.Tensor,
        exir_ops.edge.aten.__lshift__.Scalar: exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.__and__.Scalar: exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.__or__.Scalar: exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.__xor__.Scalar: exir_ops.edge.aten.bitwise_xor.Tensor,
    }

    transform_passes = {}
    if include_audio:
        transform_passes["speech_transform"] = []
        transform_passes["audio_encoder"] = []
    if include_vision:
        transform_passes["vision_encoder"] = []
    transform_passes["text_decoder"] = [_ReplaceBitwiseScalarPass(bitwise_ops)]
    return transform_passes


def export_single_pte(
    checkpoint_path: str,
    output_path: str,
    text_quantize: str = "8da4w+emb8",
    audio_quantize: str = "8da4w",
    vision_quantize: str = "8da8w",
    max_seq_len: int = 1024,
    max_patches: int = 1260,
    group_size: int = 128,
    tied_embedding: bool = False,
    variant: str = "e2b",
    quantize_kv_cache: bool = False,
    include_audio: bool = True,
    include_vision: bool = True,
    use_custom_sdpa: bool = True,
) -> Path:
    """Export components into a single PTE.

    By default includes all 4 methods: speech_transform, audio_encoder,
    vision_encoder, text_decoder. Use include_audio=False or include_vision=False
    to exclude encoders and reduce PTE size.
    """
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import (
        ConstraintBasedSymShapeEvalPass,
    )

    programs = _export_components(
        checkpoint_path=checkpoint_path,
        text_quantize=text_quantize,
        audio_quantize=audio_quantize,
        vision_quantize=vision_quantize,
        max_seq_len=max_seq_len,
        max_patches=max_patches,
        group_size=group_size,
        tied_embedding=tied_embedding,
        variant=variant,
        quantize_kv_cache=quantize_kv_cache,
        include_audio=include_audio,
        include_vision=include_vision,
        use_custom_sdpa=use_custom_sdpa,
    )

    logger.info("Combining into single PTE...")
    for name in programs:
        programs[name] = programs[name].run_decompositions({})

    partitioners = _build_partitioners(
        include_audio, include_vision, audio_quantize, vision_quantize, text_quantize
    )
    transform_passes = _build_transform_passes(include_audio, include_vision)

    edge_manager = to_edge_transform_and_lower(
        programs,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=partitioners,
        transform_passes=transform_passes,
    )

    et_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        et_program.write_to_file(f)

    if et_program._tensor_data:
        tensor_data_dir = str(output_path.parent)
        et_program.write_tensor_data_to_file(tensor_data_dir)
        logger.info(f"Tensor data written to: {tensor_data_dir}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Single PTE exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Gemma 4 as a single PTE")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to HuggingFace checkpoint directory",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="e2b",
        choices=["e2b", "e4b"],
        help="Model variant (default: e2b)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/gemma4.pte",
        help="Output path for single .pte file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length for text decoder KV cache",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="8da4w+emb8",
        choices=["8da4w+emb8", "8da4w+emb4", "8da8w+emb8", "none"],
        help="Text decoder quantization (default: 8da4w+emb8)",
    )
    parser.add_argument(
        "--audio_quantize",
        type=str,
        default="8da4w",
        choices=["8da8w", "8da4w", "none"],
        help="Audio encoder quantization (default: 8da4w)",
    )
    parser.add_argument(
        "--vision_quantize",
        type=str,
        default="8da8w",
        choices=["8da8w", "8da4w", "none"],
        help="Vision encoder quantization (default: 8da8w)",
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=1260,
        help="Max patches for vision encoder (default: 2520 = 280 soft tokens)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for INT4 weight quantization (default: 128)",
    )
    parser.add_argument(
        "--tied_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tie embed_tokens + lm_head weights to reduce model size. "
        "Requires C++ runner with TorchAO shared embedding kernels.",
    )
    parser.add_argument(
        "--quantize_kv_cache",
        action="store_true",
        default=False,
        help="Use INT8 quantized KV cache to reduce memory for long sequences.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        default=False,
        help="Exclude speech_transform and audio_encoder methods to reduce PTE size.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        default=False,
        help="Exclude vision_encoder method to reduce PTE size.",
    )
    parser.add_argument(
        "--use_custom_sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route attention through llama::custom_sdpa (tiled flash attention). "
        "Pass --no-use_custom_sdpa to fall back to matmul attention.",
    )
    args = parser.parse_args()

    export_single_pte(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        text_quantize=args.quantize,
        audio_quantize=args.audio_quantize,
        vision_quantize=args.vision_quantize,
        max_seq_len=args.max_seq_len,
        max_patches=args.max_patches,
        group_size=args.group_size,
        tied_embedding=args.tied_embedding,
        variant=args.variant,
        quantize_kv_cache=args.quantize_kv_cache,
        include_audio=not args.no_audio,
        include_vision=not args.no_vision,
        use_custom_sdpa=args.use_custom_sdpa,
    )


if __name__ == "__main__":
    main()
