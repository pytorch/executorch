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

from typing import Optional

from executorch.backends.apple.coreml.compiler import CoreMLBackend
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


def remove_graph_break_(edge_manager, method_names=None):
    from executorch.exir.dialects._ops import ops as exir_ops

    if method_names is None:
        method_names = [None]  # Default behavior for single method

    for method_name in method_names:
        if method_name is None:
            ep = edge_manager.exported_program()
        else:
            ep = edge_manager.exported_program(method_name)

        for n in ep.graph_module.graph.nodes:
            if n.target == exir_ops.edge.executorch_utils.graph_break.Tensor:
                n.replace_all_uses_with(n.args[0])
        ep.graph_module.graph.eliminate_dead_code()


def load_model(
    checkpoint_path: str,
    params_path: str,
    max_context_len: int,
    adapter_checkpoint_path: Optional[str] = None,
    adapter_config_path: Optional[str] = None,
):
    """Load the model from checkpoint with static_mha attention type.

    Args:
        checkpoint_path: Path to model checkpoint (.pth)
        params_path: Path to params.json
        max_context_len: Maximum context length
        adapter_checkpoint_path: Optional path to LoRA adapter weights (adapter_model.safetensors)
        adapter_config_path: Optional path to adapter config (adapter_config.json)
    """
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    assert (adapter_config_path is None and adapter_checkpoint_path is None) or (adapter_config_path is not None and adapter_checkpoint_path is not None)

    # Load adapter config if provided
    adapter_config = None
    if adapter_config_path is not None:
        with open(adapter_config_path, "r") as f:
            adapter_config = json.loads(f.read())
        print(f"Loaded adapter config: rank={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
        print(f"Target modules: {adapter_config.get('target_modules')}")

        # Merge adapter config into params
        params["r"] = adapter_config.get("r")
        params["lora_alpha"] = adapter_config.get("lora_alpha")
        params["target_modules"] = adapter_config.get("target_modules")

    # TODO: to support lookahead decoding, the static model outputs
    # full logits, but if we are not using lookahead decoding, we can have a
    # more efficient model by setting generate_full_logits=False and supplying the last
    # valid token
    args = ModelArgs(
        max_context_len=max_context_len,
        generate_full_logits=True,
        **params,
    )
    args.attention_type = "static_mha"
    args.attention_kwargs = {"decompose_sdpa_in_mha": True}

    with torch.device("meta"):
        model = construct_transformer(args)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    # Load and merge adapter weights if provided
    if adapter_checkpoint_path is not None:
        from safetensors.torch import load_file
        from executorch.examples.models.llama.convert_weights import unsloth_to_meta

        adapter_weights = load_file(adapter_checkpoint_path)
        # Convert adapter weight keys to Meta format
        adapter_weights = unsloth_to_meta(adapter_weights)
        print(f"Loaded {len(adapter_weights)} adapter weights")

        # Merge adapter weights into checkpoint
        checkpoint.update(adapter_weights)

    # Rename attention weight keys for static attention
    # This handles both base weights and LoRA weights
    for i in range(len(model.layers)):
        # Base weights
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

        # LoRA weights (lora_a and lora_b)
        for lora_suffix in ["lora_a.weight", "lora_b.weight"]:
            if f"layers.{i}.attention.wq.{lora_suffix}" in checkpoint:
                checkpoint[f"layers.{i}.attention.wqs.0.{lora_suffix}"] = checkpoint.pop(
                    f"layers.{i}.attention.wq.{lora_suffix}"
                )
            if f"layers.{i}.attention.wk.{lora_suffix}" in checkpoint:
                checkpoint[f"layers.{i}.attention.wks.0.{lora_suffix}"] = checkpoint.pop(
                    f"layers.{i}.attention.wk.{lora_suffix}"
                )
            if f"layers.{i}.attention.wv.{lora_suffix}" in checkpoint:
                checkpoint[f"layers.{i}.attention.wvs.0.{lora_suffix}"] = checkpoint.pop(
                    f"layers.{i}.attention.wv.{lora_suffix}"
                )

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


def prepare_model(
    model: nn.Module,
    model_args: ModelArgs,
    float_dtype: torch.dtype,
    target_split_size: int,
    max_splits: int,
    embedding_quantize: str,
    linear_quantize: str,
    no_graph_breaks: bool,
):
    """Apply dtype, splitting, quantization, and graph breaks to a model.

    Args:
        model: The model to prepare
        model_args: Model arguments
        float_dtype: Target dtype (torch.float16 or torch.float32)
        target_split_size: Target size for linear layer splitting
        max_splits: Maximum number of splits for linear layers
        embedding_quantize: Embedding quantization string (e.g., "8,0")
        linear_quantize: Linear quantization type ("b4w" or "c4w")
        no_graph_breaks: If True, skip adding graph breaks

    Returns:
        The prepared model
    """
    # Set dtype
    model = model.to(float_dtype).eval()

    # Apply linear splitting (before quantization)
    if target_split_size is not None:
        replace_linear_with_split_linear(
            model,
            out_target_split_size=target_split_size,
            out_max_splits=max_splits,
            in_target_split_size=1,
            in_max_splits=1,
        )

    def make_linear_filter_fn(group_size=0):
        """Create a filter function for linear quantization.

        Args:
            group_size: Group size for quantization. 0 means per-axis (no constraint).
        """
        def filter_fn(m, fqn):
            # Check if it's a regular nn.Linear
            is_linear = isinstance(m, nn.Linear)

            # Check if it's a LoRALinear (which has a base weight parameter to quantize)
            is_lora_linear = False
            try:
                from executorch.examples.models.llama.lora import LoRALinear
                is_lora_linear = isinstance(m, LoRALinear)
            except ImportError:
                pass

            if not (is_linear or is_lora_linear):
                return False

            # For per-axis (group_size=0), no shape constraint
            if group_size == 0:
                return True

            # Check if the weight shape is compatible with group size
            return m.weight.shape[1] % group_size == 0

        return filter_fn

    # Apply embedding quantization
    if embedding_quantize:
        bitwidth, group_size = embedding_quantize.split(",")
        bitwidth = int(bitwidth)
        group_size = int(group_size)
        assert bitwidth in [4, 8], "CoreML only supports 4-bit and 8-bit quantization"

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

    # Apply linear quantization
    if linear_quantize == "b4w":
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(32),
            ),
            filter_fn=make_linear_filter_fn(group_size=32),
        )
    elif linear_quantize == "c4w":
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerAxis(0),
            ),
            filter_fn=make_linear_filter_fn(group_size=0),
        )

    # Add graph breaks between transformer blocks
    if not no_graph_breaks:
        n_layers = len(model.layers)
        model.layers[0] = BlockWithGraphBreak(model.layers[0], break_before=True)
        model.layers[n_layers - 1] = BlockWithGraphBreak(
            model.layers[n_layers - 1], break_before=False
        )

    return model


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

    # LoRA adapter options
    parser.add_argument(
        "--adapter_checkpoint",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (adapter_model.safetensors)",
    )
    parser.add_argument(
        "--adapter_config",
        type=str,
        default=None,
        help="Path to adapter config (adapter_config.json)",
    )
    parser.add_argument(
        "--multimethod",
        action="store_true",
        help="Export both base and LoRA models as separate methods ('base' and 'lora') in one PTE file",
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

    args = parser.parse_args()

    # Compute cache length

    print("Quantization and datatype:")
    print(f"\tEmbedding quantize: {args.embedding_quantize}")
    print(f"\tLinear quantize: {args.linear_quantize}")
    print(f"\tDtype: {args.dtype}")

    cache_len = args.max_context_len - args.input_len
    print("\nGeneration configuration:")
    print(f"\tMax context length: {args.max_context_len}")
    print(f"\tInput length: {args.input_len}")
    print(f"\tCache length: {cache_len}")

    print("\nLinear splitting:")
    print(f"\tTarget split size: {args.target_split_size}")
    print(f"\tMax splits: {args.max_splits}")

    # Load model
    float_dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.multimethod:
        # Multimethod export: create both base and LoRA models
        if not args.adapter_checkpoint or not args.adapter_config:
            raise ValueError("--multimethod requires --adapter_checkpoint and --adapter_config")

        print(f"\n[Multimethod Export] Loading base model from {args.checkpoint}...")
        base_model, model_args = load_model(
            args.checkpoint,
            args.params,
            args.max_context_len,
        )
        print(f"Base model loaded: {model_args.n_layers} layers, {model_args.dim} dim")

        print(f"\n[Multimethod Export] Loading LoRA model from {args.checkpoint}...")
        print(f"  with adapter from {args.adapter_checkpoint}")
        lora_model, _ = load_model(
            args.checkpoint,
            args.params,
            args.max_context_len,
            args.adapter_checkpoint,
            args.adapter_config,
        )
        print("LoRA model loaded")

        # Prepare both models
        print("\n[Multimethod Export] Preparing base model...")
        base_model = prepare_model(
            base_model,
            model_args,
            float_dtype,
            args.target_split_size,
            args.max_splits,
            args.embedding_quantize,
            args.linear_quantize,
            args.no_graph_breaks,
        )

        print("\n[Multimethod Export] Preparing LoRA model...")
        lora_model = prepare_model(
            lora_model,
            model_args,
            float_dtype,
            args.target_split_size,
            args.max_splits,
            args.embedding_quantize,
            args.linear_quantize,
            args.no_graph_breaks,
        )

        # Create IO manager and example inputs (shared for both models)
        mgr = StaticAttentionIOManager(
            model_args,
            input_len=args.input_len,
            cache_lens=cache_len,
            batch_size=1,
            dtype=float_dtype,
            style="smart_mask",
            mask_val=float("-inf"),
        )
        example_inputs = (
            torch.zeros(1, args.input_len, dtype=torch.int32),
            {
                "masks": mgr.masks,
                "freqs_cos_override": mgr.freqs_cos[: args.input_len],
                "freqs_sin_override": mgr.freqs_sin[: args.input_len],
                "in_cache_state": (mgr.k_caches, mgr.v_caches),
            },
        )

        # Test eager execution for both models
        print("\n[Multimethod Export] Testing eager execution...")
        with torch.no_grad():
            base_model(*example_inputs)
            lora_model(*example_inputs)
        print("Eager execution successful for both models!")

        # Export both models
        print("\n[Multimethod Export] Exporting base model...")
        base_ep = torch.export.export(base_model, example_inputs, strict=False)
        print("Base model export successful!")

        print("\n[Multimethod Export] Exporting LoRA model...")
        lora_ep = torch.export.export(lora_model, example_inputs, strict=False)
        print("LoRA model export successful!")

        # Use dictionary of exported programs for multimethod
        exported_programs = {
            "base": base_ep,
            "lora": lora_ep,
        }
    else:
        # Single method export (original behavior)
        print(f"\nLoading model from {args.checkpoint}...")
        if args.adapter_checkpoint:
            print(f"Loading LoRA adapter from {args.adapter_checkpoint}...")
        model, model_args = load_model(
            args.checkpoint,
            args.params,
            args.max_context_len,
            args.adapter_checkpoint,
            args.adapter_config,
        )
        print(f"Model loaded: {model_args.n_layers} layers, {model_args.dim} dim")

        # Prepare model
        print("\nPreparing model...")
        model = prepare_model(
            model,
            model_args,
            float_dtype,
            args.target_split_size,
            args.max_splits,
            args.embedding_quantize,
            args.linear_quantize,
            args.no_graph_breaks,
        )

        # Create IO manager and example inputs
        mgr = StaticAttentionIOManager(
            model_args,
            input_len=args.input_len,
            cache_lens=cache_len,
            batch_size=1,
            dtype=float_dtype,
            style="smart_mask",
            mask_val=float("-inf"),
        )
        example_inputs = (
            torch.zeros(1, args.input_len, dtype=torch.int32),
            {
                "masks": mgr.masks,
                "freqs_cos_override": mgr.freqs_cos[: args.input_len],
                "freqs_sin_override": mgr.freqs_sin[: args.input_len],
                "in_cache_state": (mgr.k_caches, mgr.v_caches),
            },
        )

        # Test eager execution
        print("\nTesting eager execution...")
        with torch.no_grad():
            model(*example_inputs)
        print("Eager execution successful!")

        # Export the model
        print("\nExporting model...")
        ep = torch.export.export(model, example_inputs, strict=False)
        print("Export successful!")
        print(ep)

        # Use single exported program
        exported_programs = ep

    # Generate metadata for C++ runner
    print("\nGenerating metadata for C++ runner...")
    constant_methods = _get_metadata(
        model_args, example_inputs, args.input_len, cache_len, float_dtype
    )

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
    partitioner = CoreMLPartitioner(
        compile_specs=compile_specs,
        take_over_mutable_buffer=False,
        skip_ops_for_coreml_delegation=[],
    )

    # Lower to edge with constant methods for C++ runner
    print("\nLowering to edge...")
    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_manager = to_edge_transform_and_lower(
        exported_programs,
        partitioner=[partitioner],
        constant_methods=constant_methods,
        compile_config=edge_compile_config,
    )

    if args.multimethod:
        print("\nDelegated programs:")
        for method_name in ["base", "lora"]:
            print(f"\n--- {method_name} ---")
            print(format_delegated_graph(edge_manager.exported_program(method_name).graph_module))
    else:
        print("\nDelegated program:")
        print(format_delegated_graph(edge_manager.exported_program().graph_module))

    # Convert to ExecuTorch
    print("\nConverting to ExecuTorch...")
    if args.multimethod:
        remove_graph_break_(edge_manager, method_names=["base", "lora"])
    else:
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
    if args.multimethod:
        print("Methods available: 'base', 'lora'")


if __name__ == "__main__":
    main()
