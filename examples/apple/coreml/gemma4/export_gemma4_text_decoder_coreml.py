# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Gemma 4 text decoder to a CoreML-delegated ExecuTorch program.

Gemma 4's hybrid sliding/full attention is structurally compatible with
CoreML's MLProgram backend: the existing Gemma4TextModel implementation
in ``examples/models/gemma4/text_decoder/`` lowers cleanly through
``torch.export`` and ``CoreMLPartitioner``.  This script wraps that
pipeline with the CoreML-specific defaults (iOS18+ for stateful KV
caches, fp16, MQA-friendly mutable-buffer handling) so users do not
have to reassemble it themselves.

Usage::

    # From a HuggingFace checkpoint directory:
    python export_gemma4_text_decoder_coreml.py \\
        --checkpoint_path /path/to/gemma4-e2b-it \\
        --output gemma4_text_decoder.pte

    # From a JSON config alone (random weights, smoke-test mode):
    python export_gemma4_text_decoder_coreml.py \\
        --config_json /path/to/config.json --random_weights \\
        --max_seq_len 1024 --output gemma4_synthetic.pte

The audio / vision encoders shipped with Gemma 4 are not part of this
export — for those the existing ``examples/models/gemma4`` ATen pipeline
is more appropriate.
"""

import argparse
import json
import logging
import os
from typing import Optional, Tuple

import coremltools as ct
import torch

import executorch.exir
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.models.gemma4.text_decoder.gemma4_transformer import (
    Gemma4TextModel,
)
from executorch.exir import EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_config(
    checkpoint_path: Optional[str],
    config_json: Optional[str],
    max_seq_len: int,
    sliding_window: Optional[int],
    sliding_window_pattern: Optional[int],
) -> Gemma4Config:
    """Build a Gemma4Config from a checkpoint dir, a JSON file, or defaults."""
    if checkpoint_path is not None:
        config = Gemma4Config.from_json(os.path.join(checkpoint_path, "config.json"))
    elif config_json is not None:
        config = Gemma4Config.from_json(config_json)
    else:
        config = Gemma4Config()

    config.max_seq_len = max_seq_len
    config.max_context_len = max_seq_len
    if sliding_window is not None:
        config.sliding_window = sliding_window
    if sliding_window_pattern is not None:
        config.sliding_window_pattern = sliding_window_pattern
    return config


def _load_weights(
    model: Gemma4TextModel,
    config: Gemma4Config,
    checkpoint_path: str,
    dtype: torch.dtype,
) -> None:
    """Load Gemma 4 text-decoder weights from a HuggingFace checkpoint dir.

    Reuses the same convert_weights flow that examples/models/gemma4 uses
    so the loaded model exactly matches what ``examples/models/gemma4``
    would produce on the ATen path.
    """
    from executorch.examples.models.gemma4.text_decoder.convert_weights import (
        convert_hf_to_custom,
    )

    state_dict = convert_hf_to_custom(checkpoint_path, config, dtype=dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(
            "Missing %d keys when loading weights (first 5: %s)",
            len(missing),
            missing[:5],
        )
    if unexpected:
        logger.warning(
            "Unexpected %d keys (first 5: %s)", len(unexpected), unexpected[:5]
        )


def build_model(
    config: Gemma4Config,
    checkpoint_path: Optional[str],
    dtype: torch.dtype,
) -> Gemma4TextModel:
    model = Gemma4TextModel(config).eval()
    if checkpoint_path is not None:
        _load_weights(model, config, checkpoint_path, dtype)
    return model.to(dtype)


def _example_inputs(input_len: int) -> Tuple[torch.Tensor, ...]:
    """Inputs for prefill: a single batch with `input_len` placeholder tokens."""
    return (torch.zeros(1, input_len, dtype=torch.long),)


def export(
    model: Gemma4TextModel,
    input_len: int,
    minimum_deployment_target: ct.target,
    compute_precision: ct.precision,
    output_path: str,
) -> None:
    """Run the Gemma 4 text-decoder model through to_edge_transform_and_lower."""
    example_inputs = _example_inputs(input_len)

    logger.info("Eager smoke-test (input_len=%d)...", input_len)
    with torch.no_grad():
        model(*example_inputs)

    logger.info("torch.export...")
    ep = torch.export.export(model, example_inputs, strict=False)
    logger.info(
        "  exported program: %d nodes",
        sum(1 for _ in ep.graph_module.graph.nodes),
    )

    compile_specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=compute_precision,
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,
    )
    partitioner = CoreMLPartitioner(
        compile_specs=compile_specs,
        # Gemma 4's text decoder owns its KV caches as torch buffers; let
        # CoreML take them over as iOS18+ stateful tensors.
        take_over_mutable_buffer=True,
    )

    logger.info("to_edge_transform_and_lower with CoreMLPartitioner...")
    edge = executorch.exir.to_edge_transform_and_lower(
        ep,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    fully_delegated = all(
        node.op != "call_function"
        or node.target.__name__ in ("executorch_call_delegate", "getitem")
        for node in edge.exported_program().graph.nodes
    )
    if fully_delegated:
        logger.info("  fully delegated: every call_function is a CoreML call.")
    else:
        leftover = sorted(
            {
                node.target.__name__
                for node in edge.exported_program().graph.nodes
                if node.op == "call_function"
                and node.target.__name__
                not in ("executorch_call_delegate", "getitem")
            }
        )
        logger.warning(
            "  %d op type(s) fell back to portable: %s",
            len(leftover),
            leftover,
        )

    logger.info("to_executorch...")
    program = edge.to_executorch(
        ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    save_pte_program(program, output_path)
    logger.info("Saved %s (%.2f MB)", output_path, os.path.getsize(output_path) / 1e6)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a HuggingFace Gemma 4 checkpoint directory.",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Path to a Gemma 4 config.json (used if --checkpoint_path is omitted).",
    )
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Skip checkpoint loading; use random weights (smoke-test only).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gemma4_text_decoder.pte",
        help="Output .pte path.",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--input_len",
        type=int,
        default=64,
        help="Prefill sequence length used to build example inputs for export.",
    )
    parser.add_argument(
        "--sliding_window",
        type=int,
        default=None,
        help="Override the model's sliding window (default: from config).",
    )
    parser.add_argument(
        "--sliding_window_pattern",
        type=int,
        default=None,
        help="Override the sliding/full attention pattern (default: from config).",
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--minimum_deployment_target",
        type=str,
        default="iOS18",
        choices=["iOS17", "iOS18", "iOS26"],
        help="Minimum CoreML deployment target.  Stateful KV caches require iOS18+.",
    )
    args = parser.parse_args()

    if args.random_weights and (args.checkpoint_path or args.config_json):
        # Allow --random_weights with --config_json (for synthetic export); the
        # combination with --checkpoint_path would be confusing because the
        # checkpoint's config would be loaded but its weights ignored.
        if args.checkpoint_path:
            parser.error("--random_weights conflicts with --checkpoint_path")
    if not args.random_weights and not args.checkpoint_path:
        parser.error("either --checkpoint_path or --random_weights is required")

    config = _load_config(
        checkpoint_path=args.checkpoint_path if not args.random_weights else None,
        config_json=args.config_json,
        max_seq_len=args.max_seq_len,
        sliding_window=args.sliding_window,
        sliding_window_pattern=args.sliding_window_pattern,
    )

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    target = {
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
        "iOS26": ct.target.iOS26,
    }[args.minimum_deployment_target]
    precision = {torch.float16: ct.precision.FLOAT16, torch.float32: ct.precision.FLOAT32}[dtype]

    logger.info("Gemma 4 text decoder export -> CoreML")
    logger.info("  dtype=%s  target=%s", args.dtype, args.minimum_deployment_target)
    logger.info(
        "  layers=%d  hidden=%d  kv_heads=%d  sliding_window=%d  pattern=%d",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_key_value_heads,
        config.sliding_window,
        config.sliding_window_pattern,
    )

    model = build_model(
        config,
        checkpoint_path=args.checkpoint_path if not args.random_weights else None,
        dtype=dtype,
    )

    export(
        model,
        input_len=args.input_len,
        minimum_deployment_target=target,
        compute_precision=precision,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
