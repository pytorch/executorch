# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import torch
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
    gen_samsung_backend_compile_weight_spec,
    PerformanceMode,
    WeightSharingFlag,
)
from executorch.backends.samsung.utils.export_utils import get_edge_compile_config
from executorch.examples.samsung.scripts.llm.models.modeling_gemma3 import (
    Gemma3ForCausalLM_ENN,
)
from executorch.examples.samsung.scripts.llm.models.modeling_llama import (
    LlamaForCausalLM_ENN,
)
from executorch.examples.samsung.scripts.llm.util.util import (
    replace_rms_norm_with_native_rms_norm,
)
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.export_util.utils import save_pte_program
from executorch.extension.llm.export.export_passes import RemoveRedundantTransposes
from executorch.extension.llm.export.partitioner_lib import get_xnnpack_partitioner
from transformers import AutoTokenizer


# Supported model names
SUPPORTED_MODEL_NAMES = ["gemma3_1", "llama3.2_1"]

# Graph names for prefill and decode models
PREFILL_FORWARD = "prefill_forward"
DECODE_FORWARD = "decode_forward"
GRAPH_NAMES = [PREFILL_FORWARD, DECODE_FORWARD]


def get_enn_partitioner(
    chipset: str,
    graph_name: Optional[str] = None,
) -> EnnPartitioner:
    weight_sharing_flag = {
        PREFILL_FORWARD: WeightSharingFlag.WEIGHT_SHARING_GEN,
        DECODE_FORWARD: WeightSharingFlag.WEIGHT_SHARING_USE,
    }.get(graph_name, WeightSharingFlag.WEIGHT_SHARING_NONE)

    compile_specs = [
        gen_samsung_backend_compile_spec(
            chipset, PerformanceMode.HIGH_PERFORMANCE, weight_sharing_flag
        )
    ]
    if weight_sharing_flag != WeightSharingFlag.WEIGHT_SHARING_NONE:
        compile_specs.append(gen_samsung_backend_compile_weight_spec())
    return EnnPartitioner(compile_specs)


def update_metadata_with_genai_mode(
    metadata: Dict[str, Any], chipset: str, genai_mode: bool
) -> None:
    """Update metadata with get_genai_mode based on chipset and genai_mode flag.

    Args:
        metadata: Metadata dictionary to update
        chipset: Samsung chipset (e.g., "E9965", "E9955")
        genai_mode: Whether genai_mode is enabled
    """
    metadata["get_genai_mode"] = 0
    if genai_mode:
        mode_per_chipset = {"E9965": 1211, "E9955": 1215}
        chipset = chipset.upper()
        if chipset not in mode_per_chipset:
            raise ValueError(f"genai_mode is not supported for chipset {chipset}.")
        metadata["get_genai_mode"] = mode_per_chipset[chipset]


def load_llm_model(
    model_name: str,
    max_context_len: int,
    max_seq_len: int,
    ar_len: int,
) -> Tuple[torch.nn.Module, Any, Any, Dict[str, Any]]:
    """Load LLM model based on model name.

    Args:
        model_name: Name of the model (e.g., "gemma3_1")
        max_context_len: Maximum context length
        max_seq_len: Maximum sequence length
        ar_len: Auto-regression length

    Returns:
        Tuple of (model, tokenizer, example_inputs, metadata)
    """
    if model_name == "gemma3_1":
        model_id = "google/gemma-3-1b-it"
        model = Gemma3ForCausalLM_ENN.from_pretrained(
            model_id,
            dtype=torch.float32,
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
            ar_len=ar_len,
        )
    elif model_name == "llama3.2_1":
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        model = LlamaForCausalLM_ENN.from_pretrained(
            model_id,
            dtype=torch.float32,
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
            ar_len=ar_len,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.model.eval()
    example_inputs = model.get_example_inputs(num_token=ar_len)
    metadata = model.get_metadata()
    return model.model, tokenizer, example_inputs, metadata


def export_model(
    model_name: str,
    max_context_len: int,
    max_seq_len: int,
    ar_len: int,
) -> Tuple[torch.export.ExportedProgram, Dict[str, Any], Any, Any]:
    """Export LLM model.

    Args:
        model_name: Name of the model (e.g., "gemma3_1")
        max_context_len: Maximum context length
        max_seq_len: Maximum sequence length
        ar_len: Auto-regression length

    Returns:
        Tuple of (exported_model, metadata, example_inputs, example_outputs)
    """
    model, _, example_inputs, metadata = load_llm_model(
        model_name=model_name,
        max_context_len=max_context_len,
        max_seq_len=max_seq_len,
        ar_len=ar_len,
    )

    exportable_model = replace_rms_norm_with_native_rms_norm(model)
    with torch.no_grad():
        example_outputs = exportable_model.forward(*example_inputs)
    exported_model = torch.export.export(exportable_model, example_inputs, strict=True)

    remove_redundant_transposes = RemoveRedundantTransposes()
    exported_model_graph_module = remove_redundant_transposes(
        exported_model.module()
    ).graph_module

    # ExportedProgram need to be re-constructed using new transformed graph module.
    exported_model = torch.export.export(
        exported_model_graph_module, example_inputs, strict=True
    )

    return exported_model, metadata, example_inputs, example_outputs


def export_llm(args: argparse.Namespace) -> None:
    """Main export function for LLM models.

    Args:
        args: Command line arguments
    """
    # Validate arguments
    if args.model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model name: {args.model_name}. "
            f"Valid model names are: {SUPPORTED_MODEL_NAMES}"
        )
    assert args.max_context_len >= args.prefill_ar_len, (
        f"max_context_len ({args.max_context_len}) must be >= prefill_ar_len "
        f"({args.prefill_ar_len})"
    )
    assert args.max_context_len >= args.max_seq_len, (
        f"max_context_len ({args.max_context_len}) must be >= max_seq_len "
        f"({args.max_seq_len})"
    )

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Export both prefill and decode models
    exported_model = {}
    example_inputs = {}
    example_outputs = {}
    metadata = None
    for graph_name in GRAPH_NAMES:
        (
            exported_model[graph_name],
            current_metadata,
            example_inputs[graph_name],
            example_outputs[graph_name],
        ) = export_model(
            model_name=args.model_name,
            max_context_len=args.max_context_len,
            max_seq_len=args.max_seq_len,
            ar_len=args.prefill_ar_len if graph_name == PREFILL_FORWARD else 1,
        )
        if graph_name == PREFILL_FORWARD:
            metadata = current_metadata

    # Update metadata with get_genai_mode based on chipset and genai_mode flag
    update_metadata_with_genai_mode(metadata, args.chipset, args.genai_mode)

    # Create partitioners
    partitioners = {
        graph_name: [
            get_enn_partitioner(
                args.chipset, None if args.disable_weight_sharing else graph_name
            ),
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=False),
        ]
        for graph_name in GRAPH_NAMES
    }

    # Transform and lower
    edge_manager = to_edge_transform_and_lower(
        exported_model,
        transform_passes=[],
        partitioner=partitioners,
        compile_config=get_edge_compile_config(),
        constant_methods=metadata,
        generate_etrecord=False,
    )

    # Export to Executorch
    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[],
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    # Save the pte
    save_pte_program(export_program, args.model_name, args.output_dir)

    # Save example_inputs and example_outputs to separate directories
    if args.dump:
        for graph_name in GRAPH_NAMES:
            artifact_dir = os.path.join(args.output_dir, graph_name + "_in_out")
            os.makedirs(artifact_dir, exist_ok=True)
            save_tensors(example_inputs[graph_name], "float_in", artifact_dir)
            save_tensors(example_outputs[graph_name], "float_out", artifact_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LLM models for Samsung backend"
    )
    parser.add_argument(
        "-c",
        "--chipset",
        required=True,
        help="Samsung chipset, i.e. E9955, etc",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Model name. Valid ones: {SUPPORTED_MODEL_NAMES}",
    )
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")
    parser.add_argument(
        "--max_context_len",
        help="Maximum context length for model to remember, including prompt tokens and generated tokens.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--max_seq_len",
        help="Maximum sequence length generated by the model.",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--prefill_ar_len",
        help="The auto-regression (AR) length refers to the number of tokens that can be processed in a single step during the prefill phase.",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--disable_weight_sharing",
        action="store_true",
        help="Embed the weights in every method's program instead of sharing one "
        "copy through an external .ptd. Each of the prefill and decode methods "
        "then carries its own copy, so the artifacts are roughly twice as large.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Whether to dump all outputs. If not set, we only dump pte.",
    )
    parser.add_argument(
        "--genai_mode",
        action="store_true",
        help="Enable genai mode. If set, get_genai_mode will be set based on chipset. This can improve performance.",
    )

    args = parser.parse_args()

    if args.max_context_len is None:
        args.max_context_len = args.max_seq_len

    export_llm(args)


if __name__ == "__main__":
    main()
