#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LLM model from HuggingFace to MLX backend.

By default, uses optimum-executorch's CausalLMExportableModule which provides
a proven export pipeline. Optional flags enable custom MLX-optimized components:

  --use-custom-sdpa   Register MLX attention (mlx::custom_sdpa) which handles
                      K/V slicing and causal masking internally.
  --use-custom-kv-cache  Replace HF's StaticCache with HFStaticCache that uses
                         mlx::kv_cache_update for optimized cache updates.

When neither flag is set, the script behaves identically to the original
optimum-executorch export pipeline.

Usage:
    # Baseline (optimum-executorch pipeline):
    python -m executorch.backends.mlx.examples.llm.export_llm_hf \\
        --model-id "unsloth/Llama-3.2-1B-Instruct" \\
        --output llama_hf.pte

    # With custom MLX components:
    python -m executorch.backends.mlx.examples.llm.export_llm_hf \\
        --model-id "unsloth/Llama-3.2-1B-Instruct" \\
        --output llama_hf_mlx.pte \\
        --use-custom-sdpa \\
        --use-custom-kv-cache

Requirements:
    pip install transformers torch optimum-executorch
"""

import argparse
import logging
import os
from typing import Optional

import torch

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def _export_with_optimum(
    model_id: str,
    output_path: str,
    max_seq_len: int,
    dtype: str,
    qlinear: Optional[str],
    qembedding: Optional[str],
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    multimodal_only: bool = False,
    nvfp4_per_tensor_scale: bool = False,
) -> None:
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(dtype, "bfloat16")

    logger.info(f"Loading model using optimum-executorch: {model_id}")
    exportable = load_causal_lm_model(
        model_id,
        dtype=dtype_str,
        max_seq_len=max_seq_len,
    )

    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        exportable.model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
        tie_word_embeddings=getattr(
            exportable.model.config, "tie_word_embeddings", False
        )
        and not no_tie_word_embeddings,
        skip_incompatible_shapes=True,  # Skip vision tower layers with odd shapes
        nvfp4_per_tensor_scale=nvfp4_per_tensor_scale,
    )

    logger.info("Exporting model with torch.export...")
    exported_progs = exportable.export()

    if len(exported_progs) == 1:
        exported_progs = {"forward": next(iter(exported_progs.values()))}

    # Skip forward if --multimodal-only is set
    if multimodal_only and "forward" in exported_progs:
        logger.info("Removing 'forward' export (--multimodal-only)")
        del exported_progs["forward"]

    # Add multimodal export methods (token_embedding and text_decoder)
    # for compatibility with MultimodalRunner
    logger.info("Adding multimodal export methods...")
    model = exportable.model
    max_cache_len = exportable.metadata.get("get_max_seq_len", max_seq_len)

    torch_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(dtype_str, torch.bfloat16)

    seq_length = 3
    example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
    example_cache_position = torch.arange(seq_length, dtype=torch.long)
    seq_len_dim = torch.export.Dim("seq_length_dim", max=max_cache_len - 1)

    with torch.no_grad():
        # Export token_embedding method
        logger.info("  Exporting 'token_embedding' method...")
        token_embedding_layer = model.get_input_embeddings()
        token_embedding_dynamic_shapes = ({1: seq_len_dim},)
        token_embedding_ep = torch.export.export(
            token_embedding_layer,
            args=(example_input_ids,),
            dynamic_shapes=token_embedding_dynamic_shapes,
            strict=True,
        )
        exported_progs["token_embedding"] = token_embedding_ep
        logger.info("    token_embedding export completed")

        # Export text_decoder method
        logger.info("  Exporting 'text_decoder' method...")
        # Handle nested configs (e.g., Gemma3 has text_config)
        if hasattr(model.config, "text_config"):
            hidden_size = model.config.text_config.hidden_size
        else:
            hidden_size = model.config.hidden_size
        example_inputs_embeds = torch.zeros(
            (1, seq_length, hidden_size), dtype=torch_dtype
        )
        text_decoder_dynamic_shapes = {
            "inputs_embeds": {1: seq_len_dim},
            "cache_position": {0: seq_len_dim},
        }

        class TextDecoderWrapper(torch.nn.Module):
            def __init__(self, exportable_module):
                super().__init__()
                self.exportable = exportable_module

            def forward(self, inputs_embeds, cache_position):
                if hasattr(self.exportable, "cache"):
                    cache = self.exportable.cache
                elif hasattr(self.exportable, "static_cache"):
                    cache = self.exportable.static_cache
                else:
                    cache = None

                outputs = self.exportable.model(
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                )
                return outputs.logits

        text_decoder_wrapper = TextDecoderWrapper(exportable)
        text_decoder_ep = torch.export.export(
            text_decoder_wrapper,
            args=(),
            kwargs={
                "inputs_embeds": example_inputs_embeds,
                "cache_position": example_cache_position,
            },
            dynamic_shapes=text_decoder_dynamic_shapes,
            strict=True,
        )
        exported_progs["text_decoder"] = text_decoder_ep
        logger.info("    text_decoder export completed")

        # Export vision_encoder method (for multimodal models with vision tower)
        if hasattr(model, "get_image_features") or hasattr(model, "vision_tower"):
            logger.info("  Exporting 'vision_encoder' method...")

            class VisionEncoderWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_features):
                    image_embeds = self.model.get_image_features(input_features)
                    if isinstance(image_embeds, list):
                        image_embeds = torch.stack(image_embeds)
                    return image_embeds

            vision_encoder = VisionEncoderWrapper(model)

            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(model_id)
                sample_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": "https://llava-vl.github.io/static/images/view.jpg",
                            },
                        ],
                    },
                ]
                processed_inputs = processor.apply_chat_template(
                    sample_conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if "pixel_values" in processed_inputs:
                    example_pixel_values = processed_inputs["pixel_values"].to(
                        dtype=torch_dtype
                    )
                    logger.info(
                        f"    Using pixel_values shape: {example_pixel_values.shape}"
                    )

                    vision_encoder_ep = torch.export.export(
                        vision_encoder,
                        args=(),
                        kwargs={"input_features": example_pixel_values},
                        dynamic_shapes=None,
                        strict=True,
                    )
                    exported_progs["vision_encoder"] = vision_encoder_ep
                    logger.info("    vision_encoder export completed")
                else:
                    logger.warning(
                        "    Skipping vision_encoder: processor didn't return pixel_values"
                    )
            except Exception as e:
                logger.warning(f"    Skipping vision_encoder export: {e}")
        else:
            logger.info("  Skipping vision_encoder: model has no vision tower")

    logger.info("Delegating to MLX backend...")
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    edge_program = exir.to_edge_transform_and_lower(
        exported_progs,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
        constant_methods=exportable.metadata,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    _save_program(executorch_program, output_path)


def _export_with_custom_components(
    model_id: str,
    output_path: str,
    max_seq_len: int,
    dtype: str,
    qlinear: Optional[str],
    qembedding: Optional[str],
    use_custom_sdpa: bool,
    use_custom_kv_cache: bool,
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    multimodal_only: bool = False,
    nvfp4_per_tensor_scale: bool = False,
) -> None:
    """
    Export using direct HF model with custom MLX components.

    Used when --use-custom-sdpa and/or --use-custom-kv-cache are set.
    """
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from transformers import AutoModelForCausalLM
    from transformers.integrations.executorch import (
        TorchExportableModuleWithStaticCache,
    )

    torch_dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    if use_custom_sdpa:
        from executorch.backends.mlx.llm.hf_attention import register_mlx_attention

        register_mlx_attention()
        logger.info("Registered MLX custom SDPA attention")

    attn_implementation = "mlx" if use_custom_sdpa else None

    # Detect sliding window models (e.g., gemma)
    sliding_window = None

    logger.info(f"Loading HuggingFace model: {model_id}")
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Check if model uses sliding window attention
    sliding_window = getattr(model.config, "sliding_window", None)
    if sliding_window is not None:
        logger.info(f"Model has sliding_window={sliding_window}")
        # Cap max_seq_len to sliding window size for cache allocation
        effective_cache_len = min(max_seq_len, sliding_window)
        logger.info(f"  Capping cache length to sliding window: {effective_cache_len}")
    else:
        effective_cache_len = max_seq_len

    model.generation_config.cache_implementation = "static"
    model.generation_config.cache_config = {
        "batch_size": 1,
        "max_cache_len": effective_cache_len,
    }
    model.eval()

    # Use HybridCache wrapper for sliding window models (stores cache as .cache),
    # StaticCache wrapper for non-sliding-window models (stores cache as .static_cache).
    # This matters because the sliding window SDPA closure looks up the cache via
    # exportable_module.cache, matching the optimum-executorch convention.
    if sliding_window is not None:
        from transformers.integrations.executorch import (
            TorchExportableModuleWithHybridCache,
        )

        logger.info("Creating TorchExportableModuleWithHybridCache wrapper...")
        exportable = TorchExportableModuleWithHybridCache(
            model=model,
            batch_size=1,
            max_cache_len=effective_cache_len,
        )
    else:
        logger.info("Creating TorchExportableModuleWithStaticCache wrapper...")
        exportable = TorchExportableModuleWithStaticCache(
            model=model,
            batch_size=1,
            max_cache_len=effective_cache_len,
        )

    if use_custom_kv_cache:
        if sliding_window is not None:
            # Use ring buffer cache for sliding window models
            from executorch.backends.mlx.llm.source_transformation import (
                replace_hf_cache_with_mlx_ring_buffer,
            )

            logger.info(
                f"Replacing StaticCache with RingBuffer KV cache "
                f"(window_size={effective_cache_len})..."
            )
            replace_hf_cache_with_mlx_ring_buffer(
                exportable,
                model.config,
                max_batch_size=1,
                window_size=effective_cache_len,
                dtype=torch_dtype,
            )

            if use_custom_sdpa:
                # Re-register attention with sliding window closure
                from executorch.backends.mlx.llm.hf_attention import (
                    register_mlx_sliding_window_attention,
                )

                register_mlx_sliding_window_attention(exportable)
                model.config._attn_implementation = "mlx_sliding_window"
                logger.info(
                    "  Registered sliding window attention (mlx_sliding_window)"
                )

            logger.info("  RingBuffer KV cache installed successfully")
        else:
            # Use standard linear cache for non-sliding-window models
            from executorch.backends.mlx.llm.source_transformation import (
                replace_hf_cache_with_mlx,
            )

            logger.info("Replacing HuggingFace StaticCache with HFStaticCache...")
            replace_hf_cache_with_mlx(
                exportable,
                model.config,
                max_batch_size=1,
                max_cache_len=effective_cache_len,
                dtype=torch_dtype,
            )
            logger.info("  HFStaticCache installed successfully")

    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        exportable.model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
        tie_word_embeddings=getattr(model.config, "tie_word_embeddings", False)
        and not no_tie_word_embeddings,
        skip_incompatible_shapes=True,  # Skip vision tower layers with odd shapes
        nvfp4_per_tensor_scale=nvfp4_per_tensor_scale,
    )

    logger.info("Exporting model with torch.export...")
    seq_length = 3
    example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
    example_cache_position = torch.arange(seq_length, dtype=torch.long)

    seq_len_dim = torch.export.Dim("seq_length_dim", max=effective_cache_len - 1)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "cache_position": {0: seq_len_dim},
    }

    exported_programs = {}

    with torch.no_grad():
        # 1. Export "forward" method (BC for TextLLMRunner - takes input_ids)
        # Skip if --multimodal-only is set (reduces model size ~2x)
        if not multimodal_only:
            logger.info("Exporting 'forward' method (input_ids -> logits)...")
            forward_ep = torch.export.export(
                exportable,
                args=(),
                kwargs={
                    "input_ids": example_input_ids,
                    "cache_position": example_cache_position,
                },
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
            exported_programs["forward"] = forward_ep
            logger.info("  forward export completed")
        else:
            logger.info("Skipping 'forward' export (--multimodal-only)")

        # 2. Export "token_embedding" method (for MultimodalRunner)
        logger.info("Exporting 'token_embedding' method (input_ids -> embeddings)...")
        token_embedding_layer = model.get_input_embeddings()
        token_embedding_dynamic_shapes = ({1: seq_len_dim},)
        token_embedding_ep = torch.export.export(
            token_embedding_layer,
            args=(example_input_ids,),
            dynamic_shapes=token_embedding_dynamic_shapes,
            strict=True,
        )
        exported_programs["token_embedding"] = token_embedding_ep
        logger.info("  token_embedding export completed")

        # 3. Export "text_decoder" method (for MultimodalRunner - takes inputs_embeds)
        logger.info("Exporting 'text_decoder' method (inputs_embeds -> logits)...")
        # Handle nested configs (e.g., Gemma3 has text_config)
        if hasattr(model.config, "text_config"):
            hidden_size = model.config.text_config.hidden_size
        else:
            hidden_size = model.config.hidden_size
        example_inputs_embeds = torch.zeros(
            (1, seq_length, hidden_size), dtype=torch_dtype
        )
        text_decoder_dynamic_shapes = {
            "inputs_embeds": {1: seq_len_dim},
            "cache_position": {0: seq_len_dim},
        }

        # Create a wrapper that takes inputs_embeds instead of input_ids
        class TextDecoderWrapper(torch.nn.Module):
            def __init__(self, exportable_module):
                super().__init__()
                self.exportable = exportable_module

            def forward(self, inputs_embeds, cache_position):
                # Get the cache from the exportable module
                if hasattr(self.exportable, "cache"):
                    cache = self.exportable.cache
                elif hasattr(self.exportable, "static_cache"):
                    cache = self.exportable.static_cache
                else:
                    cache = None

                # Call model with inputs_embeds instead of input_ids
                outputs = self.exportable.model(
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                )
                return outputs.logits

        text_decoder_wrapper = TextDecoderWrapper(exportable)
        text_decoder_ep = torch.export.export(
            text_decoder_wrapper,
            args=(),
            kwargs={
                "inputs_embeds": example_inputs_embeds,
                "cache_position": example_cache_position,
            },
            dynamic_shapes=text_decoder_dynamic_shapes,
            strict=True,
        )
        exported_programs["text_decoder"] = text_decoder_ep
        logger.info("  text_decoder export completed")

        # 4. Export "vision_encoder" method (for multimodal models with vision tower)
        if hasattr(model, "get_image_features") or hasattr(model, "vision_tower"):
            logger.info("Exporting 'vision_encoder' method (pixel_values -> image_embeds)...")

            class VisionEncoderWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_features):
                    image_embeds = self.model.get_image_features(input_features)
                    if isinstance(image_embeds, list):
                        image_embeds = torch.stack(image_embeds)
                    return image_embeds

            vision_encoder = VisionEncoderWrapper(model)

            # Get example input from processor
            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(model_id)
                sample_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": "https://llava-vl.github.io/static/images/view.jpg",
                            },
                        ],
                    },
                ]
                processed_inputs = processor.apply_chat_template(
                    sample_conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if "pixel_values" in processed_inputs:
                    example_pixel_values = processed_inputs["pixel_values"].to(
                        dtype=torch_dtype
                    )
                    logger.info(
                        f"    Using pixel_values shape: {example_pixel_values.shape}"
                    )

                    vision_encoder_ep = torch.export.export(
                        vision_encoder,
                        args=(),
                        kwargs={"input_features": example_pixel_values},
                        dynamic_shapes=None,  # No dynamic shapes for now
                        strict=True,
                    )
                    exported_programs["vision_encoder"] = vision_encoder_ep
                    logger.info("  vision_encoder export completed")
                else:
                    logger.warning(
                        "  Skipping vision_encoder: processor didn't return pixel_values"
                    )
            except Exception as e:
                logger.warning(f"  Skipping vision_encoder export: {e}")
        else:
            logger.info("  Skipping vision_encoder: model has no vision tower")

    logger.info("Export completed successfully")
    for name, ep in exported_programs.items():
        logger.info(f"  {name}: {len(ep.range_constraints)} range constraints")

    logger.info("Delegating to MLX backend...")
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    # Build metadata methods for the etLLM app
    metadata = {
        "get_max_seq_len": effective_cache_len,
        "get_max_context_len": effective_cache_len,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": use_custom_sdpa,
        "enable_dynamic_shape": True,
    }
    logger.info(f"Exporting with metadata: {metadata}")

    edge_program = exir.to_edge_transform_and_lower(
        exported_programs,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
        constant_methods=metadata,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )

    _save_program(executorch_program, output_path)


def _save_program(executorch_program, output_path: str) -> None:
    """Save the ExecuTorch program to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved model to: {output_path}")
    logger.info(f"Program size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")


def export_llama_hf(
    model_id: str,
    output_path: str,
    max_seq_len: int = 1024,
    dtype: str = "bf16",
    qlinear: Optional[str] = None,
    qembedding: Optional[str] = None,
    use_custom_sdpa: bool = False,
    use_custom_kv_cache: bool = False,
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    multimodal_only: bool = False,
    nvfp4_per_tensor_scale: bool = False,
) -> None:
    """
    Export a HuggingFace Llama model to ExecuTorch with MLX backend.

    Args:
        model_id: HuggingFace model ID
        output_path: Path to save the .pte file
        max_seq_len: Maximum sequence length for KV cache
        dtype: Model dtype ("fp32", "fp16", "bf16")
        qlinear: Quantization for linear layers ("4w", "8w", "nvfp4", or None)
        qembedding: Quantization for embeddings ("4w", "8w", "nvfp4", or None)
        use_custom_sdpa: Use MLX custom SDPA (mlx::custom_sdpa)
        use_custom_kv_cache: Use MLX custom KV cache (mlx::kv_cache_update)
    """
    if use_custom_sdpa or use_custom_kv_cache:
        logger.info(
            f"Using custom components: sdpa={use_custom_sdpa}, "
            f"kv_cache={use_custom_kv_cache}"
        )
        _export_with_custom_components(
            model_id=model_id,
            output_path=output_path,
            max_seq_len=max_seq_len,
            dtype=dtype,
            qlinear=qlinear,
            qembedding=qembedding,
            use_custom_sdpa=use_custom_sdpa,
            use_custom_kv_cache=use_custom_kv_cache,
            no_tie_word_embeddings=no_tie_word_embeddings,
            qlinear_group_size=qlinear_group_size,
            qembedding_group_size=qembedding_group_size,
            multimodal_only=multimodal_only,
            nvfp4_per_tensor_scale=nvfp4_per_tensor_scale,
        )
    else:
        logger.info("Using optimum-executorch pipeline (no custom components)")
        _export_with_optimum(
            model_id=model_id,
            output_path=output_path,
            max_seq_len=max_seq_len,
            dtype=dtype,
            qlinear=qlinear,
            qembedding=qembedding,
            no_tie_word_embeddings=no_tie_word_embeddings,
            qlinear_group_size=qlinear_group_size,
            qembedding_group_size=qembedding_group_size,
            multimodal_only=multimodal_only,
            nvfp4_per_tensor_scale=nvfp4_per_tensor_scale,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Llama model to MLX backend"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .pte file path",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for KV cache",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype",
    )
    from executorch.backends.mlx.llm.quantization import add_quantization_args

    add_quantization_args(parser)
    parser.add_argument(
        "--use-custom-sdpa",
        action="store_true",
        default=False,
        help="Use MLX custom SDPA (mlx::custom_sdpa) for attention",
    )
    parser.add_argument(
        "--use-custom-kv-cache",
        action="store_true",
        default=False,
        help="Use MLX custom KV cache (mlx::kv_cache_update)",
    )
    parser.add_argument(
        "--multimodal-only",
        action="store_true",
        default=False,
        help="Skip 'forward' export for multimodal models (reduces size ~2x)",
    )

    args = parser.parse_args()

    export_llama_hf(
        model_id=args.model_id,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        qlinear=args.qlinear,
        qembedding=args.qembedding,
        use_custom_sdpa=args.use_custom_sdpa,
        use_custom_kv_cache=args.use_custom_kv_cache,
        no_tie_word_embeddings=args.no_tie_word_embeddings,
        qlinear_group_size=args.qlinear_group_size,
        qembedding_group_size=args.qembedding_group_size,
        multimodal_only=args.multimodal_only,
        nvfp4_per_tensor_scale=getattr(args, "nvfp4_per_tensor_scale", False),
    )


if __name__ == "__main__":
    main()
