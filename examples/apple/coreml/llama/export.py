# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import argparse
import json

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend  # pyre-ignore
from executorch.backends.apple.coreml.partition import CoreMLPartitioner  # pyre-ignore
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
)

from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.export_util.utils import export_to_edge, save_pte_program

import sys
sys.path.insert(0, "..")
from llama.llama_transformer import (
    ModelArgs,
    Transformer,
)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--output_name",
        default="model.pte",
        help="Override the output filename of the saved pte model file.",
    )
    parser.add_argument(
        "-p",
        "--params",
        help="config.json",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="checkpoint path",
    )
    parser.add_argument(
        "--static_seq_length",
        type=int,
        default=1,  # set to 1 for decode
        help="length sequence to evaluate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )
    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )
    parser.add_argument(
        "--coreml-quantize",
        default="c4w",
        choices=["b4w", "c4w"],
        help="This option is only for coreml: Use coreml quantization, e.g. b4w (for blockwise 4 bit weight), c4w (for channelwise 4 bit weight)",
    )

    export_args = parser.parse_args()
    params_path = export_args.params
    checkpoint_path = export_args.checkpoint

    # Load model args
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    args = ModelArgs(
        max_seq_len=export_args.max_seq_length,
        generate_full_logits=False,
        **params,
    )

    with torch.device("meta"):
        model = Transformer(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False,
        assign=True,
    )
    print("Missing keys: ", missing)
    print("Unexpected keys: ", unexpected)

    float_dtype = torch.float16  # dtype for model/inputs

    assert export_args.static_seq_length < args.max_seq_len

    cache_shape = (
        args.n_layers,
        args.max_batch_size,
        args.n_kv_heads,
        args.max_seq_len - export_args.static_seq_length,
        args.head_dim,
    )
    attn_mask_shape = (export_args.static_seq_length, args.max_seq_len)

    example_inputs = (
        torch.tensor(
            [0 for _ in range(export_args.static_seq_length)], dtype=torch.long
        ).reshape(1, -1),  # tokens
        torch.tensor([0], dtype=torch.long),  # input_pos
        torch.zeros(cache_shape, dtype=float_dtype),  # k_cache
        torch.zeros(cache_shape, dtype=float_dtype),  # v_cache
        torch.zeros(attn_mask_shape, dtype=float_dtype),  # attn_mask
    )
    model.eval()
    model.to(float_dtype)

    if export_args.embedding_quantize:
        bitwidth, group_size = export_args.embedding_quantize.split(",")
        if group_size == "none" or group_size == "None" or group_size == "0":
            group_size = None
        else:
            group_size = int(group_size)
        bitwidth = int(bitwidth)
        model = EmbeddingQuantHandler(
            model,
            bitwidth=bitwidth,
            group_size=group_size,
            packed=(bitwidth in [2, 4]),
        ).quantized_model()

    if export_args.coreml_quantize == "b4w":
        op_linear_quantizer_config = {
            "mode": "linear_symmetric",
            "dtype": "int4",
            "granularity": "per_block",
            "block_size": 32,
            "weight_threshold": 512,
        }
    elif export_args.coreml_quantize == "c4w":
        op_linear_quantizer_config = {
            "mode": "linear_symmetric",
            "dtype": "int4",
            "granularity": "per_channel",
        }
    else:
        raise ValueError("Invalid coreml_quantize arg")

    compile_specs = CoreMLBackend.generate_compile_specs(  # pyre-fixme[16]
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision(ct.precision.FLOAT16.value),
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,  # pyre-fixme[16]
        op_linear_quantizer_config=op_linear_quantizer_config,
    )
    partitioner = CoreMLPartitioner(  # pyre-fixme[16]
        compile_specs=compile_specs,
        take_over_mutable_buffer=False,
        skip_ops_for_coreml_delegation=[
            "quantized_decomposed.embedding_4bit.dtype",
            "aten.embedding.default",
        ],
    )

    edge_manager = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_type_promotion=(float_dtype == torch.float16),
            _skip_dim_order=True,
        ),
    )
    print("Edge program")
    print(edge_manager.exported_program())

    edge_manager = edge_manager.to_backend(partitioner)

    print("Delegated program")

    print(format_delegated_graph(edge_manager.exported_program().graph_module))

    executorch_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[
                QuantFusionPass(),
            ],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    filename = save_pte_program(executorch_program, export_args.output_name)
    print(f"Saved Executorch program to local {filename}")
    
if __name__ == "__main__":
    main()  # pragma: no cover
