# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export script for static attention LLM models to CoreML via ExecuTorch.

Usage:
    python export_static_llm_coreml.py \
        --checkpoint /path/to/model.pth \
        --params /path/to/params.json \
        --output static_llm_coreml_model.pte \
        --max_context_len 1024 \
        --input_len 32 \
        --embedding_quantize 4,32 \
        --coreml_quantize c4w \
        --target_split_size 1048
"""

import argparse
import json

import coremltools as ct
import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from executorch.backends.apple.coreml.compiler.coreml_preprocess import (
    CoreMLBackend,
    MULTIMETHOD_WEIGHT_SHARING_STRATEGY,
)
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.examples.apple.coreml.llama.utils import (
    replace_linear_with_split_linear,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope
from executorch.examples.models.llama.static_attention import StaticAttentionIOManager
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.export_util.utils import save_pte_program
from torch.library import impl, Library
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

# Define custom graph break op
lib = Library("executorch_utils", "DEF")
lib.define("graph_break.Tensor(Tensor x) -> Tensor")


@impl(lib, "graph_break.Tensor", "CompositeExplicitAutograd")
def graph_break_impl(x):
    return x


class ExecutorchGraphBreakModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return tuple(
            (
                torch.ops.executorch_utils.graph_break.Tensor(a)
                if isinstance(a, torch.Tensor)
                else a
            )
            for a in args
        )


class BlockWithGraphBreak(nn.Module):
    def __init__(self, block: nn.Module, break_before: bool = True):
        super().__init__()
        self.graph_break = ExecutorchGraphBreakModule()
        self.block = block
        self.break_before = break_before

    def forward(self, *args, **kwargs):
        if self.break_before:
            new_args = self.graph_break(*args)
            out = self.block(*new_args, **kwargs)
            return out
        else:
            out = self.block(*args, **kwargs)
            out = self.graph_break(*out)
            return out


def remove_graph_break_(edge_manager):
    """Remove graph break ops from all methods in the edge manager."""
    from executorch.exir.dialects._ops import ops as exir_ops

    # Get all method names
    method_names = edge_manager.methods
    for method_name in method_names:
        ep = edge_manager.exported_program(method_name)
        for n in ep.graph_module.graph.nodes:
            if n.target == exir_ops.edge.executorch_utils.graph_break.Tensor:
                n.replace_all_uses_with(n.args[0])
        ep.graph_module.graph.eliminate_dead_code()


def load_model(
    checkpoint_path: str,
    params_path: str,
    max_context_len: int,
    generate_full_logits: bool = True,
    adapter_checkpoint: str = None,
    adapter_config: str = None,
):
    """Load the model from checkpoint with static_mha attention type.

    Args:
        checkpoint_path: Path to the model checkpoint (.pth)
        params_path: Path to params.json
        max_context_len: Maximum context length
        generate_full_logits: If True, output logits for all tokens (needed for
            lookahead decoding). If False, only output logits for the last token
            (more efficient for standard autoregressive generation).
        adapter_checkpoint: Path to LoRA adapter weights (.safetensors)
        adapter_config: Path to adapter_config.json
    """
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    args = ModelArgs(
        max_context_len=max_context_len,
        generate_full_logits=generate_full_logits,
        **params,
    )
    args.attention_type = "static_mha"
    args.attention_kwargs = {"decompose_sdpa_in_mha": True}

    if adapter_config is not None:
        with open(adapter_config, "r") as f:
            lora_config = json.loads(f.read())
        args.r = lora_config["r"]
        args.lora_alpha = lora_config["lora_alpha"]
        args.target_modules = lora_config["target_modules"]

    with torch.device("meta"):
        model = construct_transformer(args)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    # Rename attention weight keys for static attention:
    # wq.weight -> wqs.0.weight, wk.weight -> wks.0.weight, wv.weight -> wvs.0.weight
    # LoRALinear._load_from_state_dict remaps weight -> linear.weight automatically.
    for i in range(len(model.layers)):
        if f"layers.{i}.attention.wq.weight" in checkpoint:
            checkpoint[f"layers.{i}.attention.wqs.0.weight"] = checkpoint.pop(
                f"layers.{i}.attention.wq.weight"
            )
        if f"layers.{i}.attention.wk.weight" in checkpoint:
            checkpoint[f"layers.{i}.attention.wks.0.weight"] = checkpoint.pop(
                f"layers.{i}.attention.wk.weight"
            )
        if f"layers.{i}.attention.wv.weight" in checkpoint:
            checkpoint[f"layers.{i}.attention.wvs.0.weight"] = checkpoint.pop(
                f"layers.{i}.attention.wv.weight"
            )

    if adapter_checkpoint is not None:
        from executorch.examples.models.llama.convert_weights import (
            load_and_convert_unsloth_to_meta,
        )

        adapter_weights = load_and_convert_unsloth_to_meta(adapter_checkpoint)
        # Rename adapter keys: wq.lora_*.weight -> wqs.0.lora_*.weight
        for i in range(len(model.layers)):
            for old_proj, new_proj in [
                ("wq", "wqs.0"),
                ("wk", "wks.0"),
                ("wv", "wvs.0"),
            ]:
                for suffix in ["lora_a.weight", "lora_b.weight"]:
                    old_key = f"layers.{i}.attention.{old_proj}.{suffix}"
                    if old_key in adapter_weights:
                        new_key = f"layers.{i}.attention.{new_proj}.{suffix}"
                        adapter_weights[new_key] = adapter_weights.pop(old_key)

        checkpoint.update(adapter_weights)

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False,
        assign=True,
    )
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model, args


def _create_example_inputs(
    model_args, input_len, max_context_len, float_dtype, cache_len=None
):
    """
    Create example inputs for a given input length.

    Args:
        model_args: Model configuration arguments
        input_len: Sequence length for this forward pass
        max_context_len: Maximum context length
        float_dtype: Float dtype (torch.float16 or torch.float32)
        cache_len: Optional cache length override. If None, uses max_context_len - input_len.

    Returns:
        Tuple of (example_inputs, cache_len) where example_inputs is the tuple
        expected by the model's forward method.
    """
    if cache_len is None:
        cache_len = max_context_len - input_len

    mgr = StaticAttentionIOManager(
        model_args,
        input_len=input_len,
        cache_lens=cache_len,
        batch_size=1,
        dtype=float_dtype,
        style="smart_mask",
        mask_val=float("-inf"),
    )

    options = {
        "masks": mgr.masks,
        "freqs_cos_override": mgr.freqs_cos[:input_len],
        "freqs_sin_override": mgr.freqs_sin[:input_len],
        "in_cache_state": (mgr.k_caches, mgr.v_caches),
    }

    # When generate_full_logits=False, we need to pass last_valid_token_pos
    # to tell the model which position's logits to output.
    # This is the index of the last real token (before any padding).
    if not model_args.generate_full_logits:
        options["last_valid_token_pos"] = torch.tensor(
            [input_len - 1], dtype=torch.long
        )

    example_inputs = (
        torch.zeros(1, input_len, dtype=torch.int32),
        options,
    )

    return example_inputs, cache_len


def _get_metadata(model_args, example_inputs, input_len, cache_len, float_dtype):
    """
    Generate metadata methods for the C++ runner.

    The C++ runner needs these constant methods to understand the model structure:
    - vocab_size: Vocabulary size
    - head_dim: Head dimension
    - n_heads_per_cache: Number of KV heads
    - freqs_cos, freqs_sin: Pre-computed RoPE frequencies
    - freqs_cos_input_index, freqs_sin_input_index: Input indices for RoPE
    - kv_cache_specs: Tensor describing cache input/output indices and lengths
    - mask_specs: Tensor describing mask input indices
    - forward_input_len: Input length for forward method
    """
    # Pre-compute RoPE frequencies for the full context
    rope = Rope(model_args)
    freqs_cos, freqs_sin = rope.get_freqs(None, model_args.max_context_len)
    print(f"Pre-computed RoPE frequencies shape: {freqs_cos.shape}, {freqs_sin.shape}")

    # Flatten example inputs to get the pytree spec
    flat_inputs, in_spec = pytree.tree_flatten(example_inputs)

    # Reconstruct input indices from the pytree spec
    input_indices = pytree.tree_unflatten(
        list(range(in_spec.num_leaves)),
        in_spec,
    )

    # input_indices structure:
    # (token_idx, {
    #     "masks": {cache_len: mask_idx},
    #     "freqs_cos_override": freqs_cos_idx,
    #     "freqs_sin_override": freqs_sin_idx,
    #     "in_cache_state": ({k_cache_ids: k_cache_idx}, {v_cache_ids: v_cache_idx})
    # })

    # Get the options dict indices
    opts_indices = input_indices[1]

    # Build KV cache specs: [k_in_idx, k_out_idx, v_in_idx, v_out_idx, cache_len]
    # For static_mha, output cache indices follow the same order as inputs
    # Output structure: (logits, {"out_cache_state": ({k_ids: k_out}, {v_ids: v_out})})
    k_cache_in_indices = opts_indices["in_cache_state"][0]
    v_cache_in_indices = opts_indices["in_cache_state"][1]

    # Sort by layer to ensure consistent ordering
    sorted_k_cache_ids = sorted(k_cache_in_indices.keys())

    # Output indices are in the same order (after logits)
    # Logits is output 0, then k_caches, then v_caches
    kv_cache_specs = []
    for i, cache_id in enumerate(sorted_k_cache_ids):
        k_in_idx = k_cache_in_indices[cache_id]
        v_in_idx = v_cache_in_indices[cache_id]
        # Output indices: k_caches come after logits (idx 1 to n_layers),
        # v_caches come after k_caches (idx n_layers+1 to 2*n_layers)
        k_out_idx = 1 + i
        v_out_idx = 1 + len(sorted_k_cache_ids) + i
        kv_cache_specs.append([k_in_idx, k_out_idx, v_in_idx, v_out_idx, cache_len])

    print(f"KV cache specs (k_in, k_out, v_in, v_out, cache_len): {kv_cache_specs}")

    # Build mask specs: [mask_idx, cache_len]
    mask_specs = [
        [mask_idx, c_len] for c_len, mask_idx in opts_indices["masks"].items()
    ]
    print(f"Mask specs (mask_idx, cache_len): {mask_specs}")

    return {
        "vocab_size": model_args.vocab_size,
        "head_dim": model_args.head_dim,
        "n_heads_per_cache": model_args.n_kv_heads,
        "freqs_cos": freqs_cos.to(float_dtype),
        "freqs_sin": freqs_sin.to(float_dtype),
        "freqs_cos_input_index": torch.tensor(
            [opts_indices["freqs_cos_override"]], dtype=torch.int64
        ),
        "freqs_sin_input_index": torch.tensor(
            [opts_indices["freqs_sin_override"]], dtype=torch.int64
        ),
        "mask_specs": torch.tensor(mask_specs, dtype=torch.int64),
        "kv_cache_specs": torch.tensor(kv_cache_specs, dtype=torch.int64),
        "forward_input_len": input_len,
    }


def _transform_eager_model(model, args, float_dtype):
    """Apply splitting, quantization, and graph breaks to a model.

    This is shared across base and adapter models so the same transformations
    are applied consistently.
    """
    from executorch.examples.models.llama.lora import LoRALinear

    model = model.to(float_dtype).eval()

    if args.target_split_size is not None:
        print(f"\nSplitting linear layers with target size {args.target_split_size}...")
        replace_linear_with_split_linear(
            model,
            out_target_split_size=args.target_split_size,
            out_max_splits=args.max_splits,
            in_target_split_size=1,
            in_max_splits=1,
        )

    if args.embedding_quantize:
        bitwidth, group_size = args.embedding_quantize.split(",")
        bitwidth = int(bitwidth)
        group_size = int(group_size)
        assert bitwidth in [4, 8], "CoreML only supports 4-bit and 8-bit quantization"

        print(f"\nQuantizing embeddings: {bitwidth}-bit, group_size={group_size}...")
        if group_size == 0:
            granularity = PerAxis(0)
        else:
            granularity = PerGroup(group_size)
        weight_dtype = getattr(torch, f"int{bitwidth}")

        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=weight_dtype, granularity=granularity),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

    has_lora_modules = any(isinstance(m, LoRALinear) for m in model.modules())

    def _exclude_lora(m, fqn):
        if isinstance(m, LoRALinear):
            return False
        parts = fqn.split(".")
        if "lora_a" in parts or "lora_b" in parts:
            return False
        return isinstance(m, nn.Linear)

    linear_filter = _exclude_lora if has_lora_modules else None

    if args.linear_quantize == "b4w":
        print("\nQuantizing linear layers: 4-bit blockwise (group_size=32)...")
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(32),
            ),
            linear_filter,
        )
    elif args.linear_quantize == "c4w":
        print("\nQuantizing linear layers: 4-bit channelwise...")
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerAxis(0),
            ),
            linear_filter,
        )

    if not args.no_graph_breaks:
        print("\nAdding graph breaks between before/after the transformer blocks...")
        n_layers = len(model.layers)
        model.layers[0] = BlockWithGraphBreak(model.layers[0], break_before=True)
        model.layers[n_layers - 1] = BlockWithGraphBreak(
            model.layers[n_layers - 1], break_before=False
        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export static attention Llama model to CoreML"
    )

    # Model paths
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to params.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="model.pte",
        help="Output filename for the .pte model",
    )

    # Model configuration
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=1024,
        help="Maximum context length",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=32,
        help="Input sequence length per forward pass",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model dtype.  The ANE requires fp16.",
    )

    # Quantization options
    parser.add_argument(
        "-E",
        "--embedding_quantize",
        default="8,0",
        type=str,
        help="Embedding quantization: '<bitwidth>,<groupsize>', e.g., '4,32' or '8,0' for per-channel",
    )
    parser.add_argument(
        "--linear_quantize",
        default="c4w",
        choices=["b4w", "c4w"],
        help="CoreML linear quantization: b4w (blockwise 4-bit) or c4w (channelwise 4-bit).  The ANE requires channelwise.",
    )

    # Linear splitting options
    parser.add_argument(
        "--target_split_size",
        type=int,
        default=1024,
        help="Split linear layers into chunks of this size (helps with ANE performance)",
    )
    parser.add_argument(
        "--max_splits",
        type=int,
        default=8,
        help="Maximum number of splits for linear layers",
    )

    # Graph break options
    parser.add_argument(
        "--no_graph_breaks",
        action="store_true",
        help="Disable graph breaks between transformer blocks",
    )

    # Export mode options
    parser.add_argument(
        "--multifunction",
        action="store_true",
        help="Export as multifunction model with separate prefill (seqlen=input_len) "
        "and decode (seqlen=1) methods. Weight sharing is enabled across methods. "
        "When disabled, exports a single-method model with fixed seqlen=input_len "
        "and generate_full_logits=True for lookahead decoding support.",
    )

    # LoRA adapter options
    parser.add_argument(
        "--adapter",
        nargs=3,
        action="append",
        metavar=("NAME", "CHECKPOINT", "CONFIG"),
        help="LoRA adapter: method name, path to adapter.safetensors, "
        "path to adapter_config.json. Can be repeated for multiple adapters.",
    )

    args = parser.parse_args()

    has_adapters = args.adapter is not None

    print("Export mode:")
    if args.multifunction:
        print(
            "\tMultifunction: separate prefill/decode graphs, generate_full_logits=False"
        )
    else:
        print("\tSingle method: fixed seqlen, generate_full_logits=True (lookahead)")
    if has_adapters:
        print(f"\tAdapters: {[a[0] for a in args.adapter]}")

    print("\nQuantization and datatype:")
    print(f"\tEmbedding quantize: {args.embedding_quantize}")
    print(f"\tLinear quantize: {args.linear_quantize}")
    print(f"\tDtype: {args.dtype}")

    # Compute cache length
    cache_len = args.max_context_len - args.input_len
    print("\nGeneration configuration:")
    print(f"\tMax context length: {args.max_context_len}")
    print(f"\tInput length: {args.input_len}")
    print(f"\tCache length: {cache_len}")

    print("\nLinear splitting:")
    print(f"\tTarget split size: {args.target_split_size}")
    print(f"\tMax splits: {args.max_splits}")

    # Load base model
    # For multifunction: generate_full_logits=False (efficient, only last token)
    # For single method: generate_full_logits=True (needed for lookahead decoding)
    generate_full_logits = not args.multifunction
    print(f"\nLoading model from {args.checkpoint}...")
    model, model_args = load_model(
        args.checkpoint,
        args.params,
        args.max_context_len,
        generate_full_logits=generate_full_logits,
    )
    print(f"Model loaded: {model_args.n_layers} layers, {model_args.dim} dim")

    float_dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = _transform_eager_model(model, args, float_dtype)

    # Load adapter models
    lora_models = {}
    if has_adapters:
        for name, adapter_ckpt, adapter_cfg in args.adapter:
            print(f"\nLoading adapter '{name}' from {adapter_ckpt}...")
            lora_model, _ = load_model(
                args.checkpoint,
                args.params,
                args.max_context_len,
                generate_full_logits=generate_full_logits,
                adapter_checkpoint=adapter_ckpt,
                adapter_config=adapter_cfg,
            )
            lora_model = _transform_eager_model(lora_model, args, float_dtype)
            lora_models[name] = lora_model

    def _export_model(m, inputs, label="model"):
        print(f"\nTesting eager execution ({label})...")
        with torch.no_grad():
            m(*inputs)
        print(f"Eager execution successful ({label})!")

        print(f"\nExporting {label}...")
        ep = torch.export.export(m, inputs)
        print(f"Export successful ({label})!")
        print(ep)
        return ep

    if args.multifunction:
        # Multifunction mode: separate prefill and decode graphs with weight sharing
        # Both methods use the same cache_len (decode's cache size) so they can share
        # the same cache buffer at runtime without any copying.
        decode_input_len = 1
        prefill_input_len = args.input_len  # default 32
        shared_cache_len = (
            args.max_context_len - decode_input_len
        )  # Use decode's cache size for both

        print(f"\nShared cache length for prefill/decode: {shared_cache_len}")

        print(f"\nCreating example inputs for decode (seqlen={decode_input_len})...")
        decode_inputs, decode_cache_len = _create_example_inputs(
            model_args,
            decode_input_len,
            args.max_context_len,
            float_dtype,
            cache_len=shared_cache_len,
        )

        print(f"Creating example inputs for prefill (seqlen={prefill_input_len})...")
        prefill_inputs, prefill_cache_len = _create_example_inputs(
            model_args,
            prefill_input_len,
            args.max_context_len,
            float_dtype,
            cache_len=shared_cache_len,
        )

        # Export base model
        methods = {
            "forward": _export_model(model, decode_inputs, "base decode"),
            "prefill": _export_model(model, prefill_inputs, "base prefill"),
        }

        # Export adapter models
        for name, lora_model in lora_models.items():
            methods[f"{name}_forward"] = _export_model(
                lora_model, decode_inputs, f"{name} decode"
            )
            methods[f"{name}_prefill"] = _export_model(
                lora_model, prefill_inputs, f"{name} prefill"
            )

        # Generate metadata
        print("\nGenerating metadata for C++ runner...")
        decode_metadata = _get_metadata(
            model_args, decode_inputs, decode_input_len, decode_cache_len, float_dtype
        )
        prefill_metadata = _get_metadata(
            model_args,
            prefill_inputs,
            prefill_input_len,
            prefill_cache_len,
            float_dtype,
        )

        # Combine metadata - shared values go without prefix,
        # method-specific values get prefixed.
        constant_methods = {
            # Shared metadata (same for both methods)
            "vocab_size": decode_metadata["vocab_size"],
            "head_dim": decode_metadata["head_dim"],
            "n_heads_per_cache": decode_metadata["n_heads_per_cache"],
            "freqs_cos": decode_metadata["freqs_cos"],
            "freqs_sin": decode_metadata["freqs_sin"],
            # Decode-specific metadata (forward method)
            "decode_input_len": decode_metadata["forward_input_len"],
            "decode_freqs_cos_input_index": decode_metadata["freqs_cos_input_index"],
            "decode_freqs_sin_input_index": decode_metadata["freqs_sin_input_index"],
            "decode_mask_specs": decode_metadata["mask_specs"],
            "decode_kv_cache_specs": decode_metadata["kv_cache_specs"],
            # Prefill-specific metadata
            "prefill_input_len": prefill_metadata["forward_input_len"],
            "prefill_freqs_cos_input_index": prefill_metadata["freqs_cos_input_index"],
            "prefill_freqs_sin_input_index": prefill_metadata["freqs_sin_input_index"],
            "prefill_mask_specs": prefill_metadata["mask_specs"],
            "prefill_kv_cache_specs": prefill_metadata["kv_cache_specs"],
        }
        if has_adapters:
            constant_methods["has_lora"] = True
    else:
        # Fixed seqlen mode: base + optional adapter methods
        print(f"\nCreating example inputs (seqlen={args.input_len})...")
        example_inputs, example_cache_len = _create_example_inputs(
            model_args, args.input_len, args.max_context_len, float_dtype
        )

        methods = {
            "forward": _export_model(model, example_inputs, "base"),
        }
        for name, lora_model in lora_models.items():
            methods[name] = _export_model(lora_model, example_inputs, name)

        print("\nGenerating metadata for C++ runner...")
        constant_methods = _get_metadata(
            model_args, example_inputs, args.input_len, example_cache_len, float_dtype
        )
        if has_adapters:
            constant_methods["has_lora"] = True

    # Setup CoreML partitioner
    print("\nSetting up CoreML partitioner...")
    compile_specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision={
            torch.float16: ct.precision.FLOAT16,
            torch.float32: ct.precision.FLOAT32,
        }[float_dtype],
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,
    )
    if len(methods) > 1:
        compile_specs.append(
            CoreMLBackend.generate_multimethod_weight_sharing_strategy_compile_spec(
                MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL
            )
        )
    partitioner = CoreMLPartitioner(
        compile_specs=compile_specs,
        take_over_mutable_buffer=False,
        skip_ops_for_coreml_delegation=[],
    )

    # Lower to edge
    print("\nLowering to edge...")
    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_manager = to_edge_transform_and_lower(
        methods,
        partitioner=[partitioner],
        constant_methods=constant_methods,
        compile_config=edge_compile_config,
    )
    for method_name in methods:
        print(f"\nDelegated program ({method_name}):")
        print(
            format_delegated_graph(
                edge_manager.exported_program(method_name).graph_module
            )
        )

    # Convert to ExecuTorch
    print("\nConverting to ExecuTorch...")
    remove_graph_break_(edge_manager)
    executorch_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    # Save the program
    filename = save_pte_program(executorch_program, args.output)
    print(f"\nSaved ExecuTorch program to {filename}")


if __name__ == "__main__":
    main()
