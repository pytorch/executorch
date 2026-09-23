# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import math
from pathlib import Path

import torch
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import MAX_BRANCH, MAX_STATE, SPECIAL
from model import Backbone, Prefill, Score
from torch.export import Dim


def export_model(backbone, head, limits, metadata):
    max_prefix, max_context, max_questions, max_options = limits
    sample_prefix = min(3, max_prefix)
    first_linear = next(
        layer.linear_attn
        for layer in backbone.layers
        if layer.block_type == "linear_attention"
    )
    first_full = next(
        layer.self_attn
        for layer in backbone.layers
        if layer.block_type == "full_attention"
    )
    n_linear = sum(layer.block_type == "linear_attention" for layer in backbone.layers)
    dtype = backbone.embed_tokens.weight.dtype
    conv = torch.zeros(
        n_linear, 1, first_linear.conv_dim, first_linear.conv_kernel_size, dtype=dtype
    )
    recurrent = torch.zeros(
        n_linear,
        1,
        first_linear.num_v_heads,
        first_linear.head_k_dim,
        first_linear.head_v_dim,
    )
    kv = torch.zeros(
        len(backbone.layers) - n_linear,
        2,
        1,
        first_full.config.num_key_value_heads,
        sample_prefix,
        first_full.head_dim,
        dtype=dtype,
    )
    prefix_dim = Dim("prefix", min=1, max=max_prefix) if max_prefix > 1 else Dim.STATIC
    question_dim = Dim("questions", min=1, max=max_questions)
    option_dim = Dim("options", min=1, max=max_options)
    branch_dim = Dim("branch", min=2, max=max_context - 1)
    with torch.no_grad():
        programs = {
            "prefill": torch.export.export(
                Prefill(backbone).eval(),
                (torch.zeros(1, sample_prefix, dtype=torch.long),),
                dynamic_shapes=({1: prefix_dim},),
                strict=True,
            ),
            "score": torch.export.export(
                Score(backbone, head).eval(),
                (
                    torch.zeros(2, 5, dtype=torch.long),
                    torch.full((2,), 4, dtype=torch.long),
                    torch.tensor([[2, 3], [2, 3]]),
                    conv,
                    recurrent,
                    kv,
                ),
                dynamic_shapes=(
                    {0: question_dim, 1: branch_dim},
                    {0: question_dim},
                    {0: question_dim, 1: option_dim},
                    None,
                    None,
                    {4: prefix_dim},
                ),
                strict=True,
            ),
        }
        if backbone.backend == "mlx":
            from executorch.backends.mlx import MLXPartitioner
            from executorch.backends.mlx.passes import get_default_passes

            partitioner = MLXPartitioner()
            passes = get_default_passes()
        else:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )

            partitioner = XnnpackPartitioner(enable_bf16=dtype == torch.bfloat16)
            passes = []
        edge = to_edge_transform_and_lower(
            programs,
            partitioner=[partitioner],
            transform_passes=passes,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=backbone.backend == "mlx"
            ),
            constant_methods=metadata,
        )
        return edge.to_executorch(
            ExecutorchBackendConfig(
                extract_delegate_segments=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export Kev prefill and pointer scoring"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=("xnnpack", "mlx"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--max-prefix",
        type=int,
        default=MAX_STATE,
        help="Maximum state tokens, including the state delimiter (default: %(default)s)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=MAX_BRANCH,
        help="Maximum tokens for the prefix plus one question (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_prefix < args.max_context <= 2**31 - 1 or args.max_context < 6:
        parser.error(
            "Require 1 <= max-prefix < max-context <= 2147483647 and max-context >= 6"
        )
    checkpoint = Checkpoint(args.checkpoint)
    if (
        checkpoint.meta.option_isolation
        or checkpoint.meta.special_embeddings
        or checkpoint.meta.weights_dtype != "fp32"
        or checkpoint.adapter_config().get("trainable_token_indices")
    ):
        parser.error("Expected a dense Qwen3.5 Kev checkpoint with mergeable FP32 LoRA")
    if (
        not math.isfinite(checkpoint.meta.temperature)
        or checkpoint.meta.temperature <= 0
    ):
        parser.error("Checkpoint temperature must be finite and positive")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    tokenizer, model = checkpoint.load("cpu", LoadOptions(dtype=dtype, attn="sdpa"))
    backbone = Backbone(model.lm, args.backend).eval()
    limits = (args.max_prefix, args.max_context, 8, 255)
    metadata = dict(
        zip(
            (
                "get_max_prefix",
                "get_max_context",
                "get_max_questions",
                "get_max_options",
            ),
            limits,
        )
    )
    metadata.update(
        get_kev_version=1,
        get_pad_id=model.pad_id,
        get_temperature=float(model.head.temperature),
        get_checkpoint_id="sha256:"
        + hashlib.sha256(checkpoint.file("head.pt").read_bytes()).hexdigest(),
    )
    for index, token in enumerate(SPECIAL):
        metadata[f"get_special_{index}"] = tokenizer.convert_tokens_to_ids(token)
    program = export_model(backbone, model.head, limits, metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "model.pte").open("wb") as f:
        program.write_to_file(f)
    tokenizer.save_pretrained(args.output)
    print(args.output / "model.pte")


if __name__ == "__main__":
    main()
