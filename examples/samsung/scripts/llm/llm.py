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
from executorch.backends.samsung.quantizer import EnnQuantizer, Precision
from executorch.backends.samsung.quantizer.custom_annotation import (
    custom_annotate_gemma3_a16w4,
    custom_annotate_llama_a16w4,
)
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
    build_calib_dataset,
    fuse_layer_norms,
    get_model_with_r1_r2,
    removing_gpu_node_in_graph,
    replace_rms_norm_with_native_rms_norm,
)

from executorch.examples.samsung.utils import save_tensors
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.export_util.utils import save_pte_program
from executorch.extension.llm.export.export_passes import RemoveRedundantTransposes
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoTokenizer


# Supported model names
SUPPORTED_MODEL_NAMES = ["gemma3_1", "llama3.2_1"]

# Graph names for prefill and decode models
PREFILL_FORWARD = "prefill_forward"
DECODE_FORWARD = "decode_forward"
GRAPH_NAMES = [PREFILL_FORWARD, DECODE_FORWARD]


def _to_cpu(value: Any) -> Any:
    """Recursively rewrite Tensors in nested tuple/list/dict to CPU copies.

    Keeps non-tensor values unchanged. Used to run export on CPU even when the
    reference forward was computed on GPU.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def get_device(device: Optional[str] = None, precision: bool = False) -> torch.device:
    """Resolve the device to run the model on, checking GPU availability.

    Args:
        device: Requested device, e.g. "cpu", "cuda", or "cuda:6". If None,
            auto-detect using CUDA availability.

    Returns:
        The resolved torch.device. Falls back to CPU if the requested CUDA
        device is not available on the host.

    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not precision:
        device = "cpu"
    return torch.device(device)


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
    device: Optional[str] = None,
    precision: bool = False,
    model_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any, Any, Dict[str, Any]]:
    """Load LLM model based on model name.

    Args:
        model_name: Name of the model (e.g., "gemma3_1")
        max_context_len: Maximum context length
        max_seq_len: Maximum sequence length
        ar_len: Auto-regression length
        device: Compute device to load the model on (e.g., "cpu" or "cuda"). If
            None, auto-detect using GPU availability.
        model_path: Optional local model directory (or hub id) to load from
            instead of the built-in default model for ``model_name``.

    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    target_device = get_device(device, precision)
    if model_name == "gemma3_1":
        if model_path is not None:
            model_id = model_path
        elif precision:
            model_id = "google/gemma-3-1b-it-qat-int4-unquantized"
        else:
            model_id = "google/gemma-3-1b-it"
        model = Gemma3ForCausalLM_ENN.from_pretrained(
            model_id,
            dtype=torch.float32,
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
            ar_len=ar_len,
            device_map=target_device,
        )
    elif model_name == "llama3.2_1":
        model_id = (
            model_path if model_path is not None else "meta-llama/Llama-3.2-1B-Instruct"
        )
        model = LlamaForCausalLM_ENN.from_pretrained(
            model_id,
            dtype=torch.float32,
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
            ar_len=ar_len,
            device_map=target_device,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"

    model.model.eval()
    metadata = model.get_metadata()
    return model, tokenizer, metadata


def load_and_prepare_model(
    model_name: str,
    max_context_len: int,
    max_seq_len: int,
    ar_len: int,
    device: Optional[str] = None,
    precision: bool = False,
    sample_number: int = 1,
    optimized_rotation_path: str = None,
    model_path: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Load the LLM model and return a single ready-to-export target model.

    Based on ``precision`` this is either the FP model (with native RMSNorm) or
    the quantized (A16W4) module. Export always runs on CPU, so the target model
    is moved to CPU here, up front, instead of inside the per-graph export loop.
    This also makes the expensive steps (downloading/reading the weights and the
    quantization calibration/generation) happen only a single time rather than
    once per graph.

    Args:
        model_name: Name of the model (e.g., "gemma3_1")
        max_context_len: Maximum context length
        max_seq_len: Maximum sequence length
        ar_len: Auto-regression length used to build the KV cache
        device: Compute device to load the model on (e.g., "cpu" or "cuda"). If
            None, auto-detect using GPU availability.
        precision: Whether to quantize the model (A16W4).
        sample_number: Number of calibration samples to use during quantization.
        optimized_rotation_path: Ratation martix for llama

    Returns:
        Tuple of (causal_model, target_model, metadata). ``causal_model`` is
        used by the export loop to rebind the KV cache and build example inputs.
    """
    causal_model, tokenizer, metadata = load_llm_model(
        model_name=model_name,
        max_context_len=max_context_len,
        max_seq_len=max_seq_len,
        ar_len=ar_len,
        device=device,
        precision=precision,
        model_path=model_path,
    )

    exportable_model = replace_rms_norm_with_native_rms_norm(causal_model.model)
    target_device = get_device(device, precision)

    if optimized_rotation_path is not None:
        if not precision:
            raise ValueError("--optimized_rotation_path requires --precision.")
        if "gemma3" in model_name:
            raise ValueError("--optimized_rotation_path only applies to llama models.")
        if not os.path.isfile(optimized_rotation_path):
            raise FileNotFoundError(optimized_rotation_path)

    if precision:
        quant_dtype = Precision.A16W4
        is_per_channel = True
        is_qat = False
        enable_spin = False

        if "gemma3" in model_name:
            custom_annotations = (custom_annotate_gemma3_a16w4,)
            act_observer_cls = HistogramObserver
        else:
            custom_annotations = (custom_annotate_llama_a16w4,)
            act_observer_cls = MinMaxObserver
            if optimized_rotation_path is not None:
                if id(causal_model.model.lm_head.weight) == id(
                    causal_model.model.embed_tokens.weight
                ):
                    causal_model.model.lm_head.weight = torch.nn.Parameter(
                        causal_model.model.embed_tokens.weight.data.detach().clone(),
                        requires_grad=False,
                    )
                    causal_model.config.tie_word_embeddings = False
                exportable_model = fuse_layer_norms(exportable_model)
                exportable_model = get_model_with_r1_r2(optimized_rotation_path)(
                    exportable_model
                )
                causal_model.model = exportable_model
                enable_spin = True

        example_inputs, dynamic_shapes = causal_model.get_example_inputs(
            num_token=ar_len
        )
        with torch.inference_mode():
            exported_model = torch.export.export(
                exportable_model, example_inputs, dynamic_shapes=dynamic_shapes
            )

        causal_model.model = exported_model.module()

        quantizer = EnnQuantizer()
        quantizer.setup_quant_params(
            quant_dtype,
            is_per_channel,
            is_qat,
            act_observer_cls=act_observer_cls,
            enable_spin=enable_spin,
        )
        quantizer.add_custom_quant_annotations(custom_annotations)

        m = prepare_pt2e(exported_model.module(), quantizer)
        m = m.to(target_device)
        causal_model.model = m

        dataset = build_calib_dataset(
            tokenizer,
            sample_number=sample_number,
            max_length=max_seq_len,
        )

        # Generate from each calibration sample so the observers of the prepared
        # module see activations for the prompt and for decoding.
        for d in dataset:
            max_new_tokens_gen = max_seq_len - d["input_ids"].shape[1]
            causal_model.generate(
                d["input_ids"].to(target_device),
                max_new_tokens=max_new_tokens_gen,
                disable_compile=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # convert_pt2e returns the quantized GraphModule directly.
        target_model = convert_pt2e(causal_model.model)

        causal_model.model = target_model

        with torch.inference_mode():
            exported_model = torch.export.export(
                target_model, example_inputs, dynamic_shapes=dynamic_shapes
            )

        causal_model.model = exported_model.module()

        if target_device.type != "cpu":
            target_model = target_model.to("cpu")
            target_model = removing_gpu_node_in_graph(target_model)
            causal_model.target_device = torch.device("cpu")
    else:
        target_model = exportable_model

    # Export always runs on CPU; move the selected model there up front.
    target_model = target_model.to("cpu")

    return causal_model, target_model, metadata


def export_prepared_model(
    causal_model: Any,
    target_model: Any,
    ar_len: int,
) -> Tuple[torch.export.ExportedProgram, Any, Any]:
    """Export a single graph (prefill/decode) from the shared prepared model.

    The model itself is loaded/quantized and moved to CPU once by
    ``load_and_prepare_model``; here we only rebind the KV cache for the graph's
    ``ar_len`` and export the reused target model with graph-specific inputs.

    Args:
        causal_model: Loaded (and possibly quantized) causal LM wrapper.
        target_model: Ready-to-export module (FP or quantized), already on CPU.
        ar_len: The graph's auto-regression length.

    Returns:
        Tuple of (exported_model, example_inputs, example_outputs)
    """
    causal_model.ar_len = ar_len
    causal_model.setup_kv_cache()

    example_inputs = causal_model.get_example_inputs(num_token=ar_len)[0]
    example_inputs = _to_cpu(example_inputs)
    with torch.no_grad():
        example_outputs = target_model.forward(*example_inputs)

    with torch.inference_mode():
        exported_model = torch.export.export(target_model, example_inputs, strict=True)

    remove_redundant_transposes = RemoveRedundantTransposes()
    exported_model_graph_module = remove_redundant_transposes(
        exported_model.module()
    ).graph_module

    # ExportedProgram need to be re-constructed using new transformed graph module.
    with torch.inference_mode():
        exported_model = torch.export.export(
            exported_model_graph_module, example_inputs, strict=True
        )

    return exported_model, example_inputs, example_outputs


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

    # Load and (optionally) quantize the model once, before exporting any graph.
    # Fail before the slow load and calibration if the chipset cannot run genai_mode.
    update_metadata_with_genai_mode({}, args.chipset, args.genai_mode)
    causal_model, target_model, metadata = load_and_prepare_model(
        model_name=args.model_name,
        max_context_len=args.max_context_len,
        max_seq_len=args.max_seq_len,
        ar_len=args.prefill_ar_len,
        device=args.device,
        precision=args.precision,
        sample_number=args.sample_number,
        optimized_rotation_path=args.optimized_rotation_path,
        model_path=args.model_path,
    )

    # Export both prefill and decode models, reusing the prepared target model.
    exported_model = {}
    example_inputs = {}
    example_outputs = {}
    for graph_name in GRAPH_NAMES:
        (
            exported_model[graph_name],
            example_inputs[graph_name],
            example_outputs[graph_name],
        ) = export_prepared_model(
            causal_model=causal_model,
            target_model=target_model,
            ar_len=args.prefill_ar_len if graph_name == PREFILL_FORWARD else 1,
        )

    # Update metadata with get_genai_mode based on chipset and genai_mode flag
    update_metadata_with_genai_mode(metadata, args.chipset, args.genai_mode)

    # Create partitioners
    partitioners = {
        graph_name: [
            get_enn_partitioner(
                args.chipset, None if args.disable_weight_sharing else graph_name
            ),
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
        default="E9955",
        help="Samsung chipset, i.e. E9955, etc",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        default="gemma3_1",
        help=f"Model name. Valid ones: {SUPPORTED_MODEL_NAMES}",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Compute device to run the model on, e.g. 'cpu', 'cuda', or 'cuda:6'. "
            "A GPU is only used when quantizing (--precision); otherwise the model "
            "runs on CPU."
        ),
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
    parser.add_argument(
        "-p",
        "--precision",
        action="store_true",
        default=False,
        help=("Quantization precision. If set, the model will be quantized to A16W4."),
    )
    parser.add_argument(
        "--sample_number",
        help="Number of calibration samples to use for quantization.",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--optimized_rotation_path",
        default=None,
        required=False,
        help="Optimized rotation checkpoint path. Just apply R1/R2 for llama here."
        "You can train the rotation matrices by https://github.com/facebookresearch/SpinQuant/tree/main",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        required=False,
        help=(
            "Optional local model directory (or hub id) to load the model and "
            "tokenizer from, instead of the built-in default for --model_name. "
            "Useful for offline / pre-downloaded checkpoints."
        ),
    )
    args = parser.parse_args()

    if args.max_context_len is None:
        args.max_context_len = args.max_seq_len

    export_llm(args)


if __name__ == "__main__":
    main()
