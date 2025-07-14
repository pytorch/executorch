# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend  # pyre-ignore
from executorch.backends.apple.coreml.partition import CoreMLPartitioner  # pyre-ignore

from executorch.examples.apple.coreml.llama.llama_transformer import (
    InputManager,
    load_model,
)
from executorch.examples.apple.coreml.llama.utils import (
    replace_linear_with_split_linear,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
)

from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.program._program import to_edge_with_preserved_ops
from executorch.extension.export_util.utils import save_pte_program


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
        "--seq_length",
        type=int,
        default=1,
        help="length sequence to evaluate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=None,
        help="Cache size.  Old items are evicted from cache",
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
        default=None,
        choices=["b4w", "c4w"],
        help="This option is only for coreml: Use coreml quantization, e.g. b4w (for blockwise 4 bit weight), c4w (for channelwise 4 bit weight)",
    )
    parser.add_argument(
        "--use_cache_list",
        action="store_true",
        help="Use cache list to speed up model computation (does not work in pybindings)",
    )
    parser.add_argument(
        "--target_split_size",
        type=int,
        default=None,
        help="Split linear layers into smaller chunks of target_split_size.",
    )
    parser.add_argument(
        "--max_splits",
        type=int,
        default=8,
        help="Maximum number of splits to divide linear layers",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
    )

    export_args = parser.parse_args()
    model = load_model(
        export_args.checkpoint,
        export_args.params,
        max_seq_length=export_args.max_seq_length,
        use_cache_list=export_args.use_cache_list,
    )

    float_dtype = {"fp16": torch.float16, "fp32": torch.float32}[
        export_args.dtype
    ]  # dtype for model/inputs

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

    if export_args.target_split_size is not None:
        replace_linear_with_split_linear(
            model,
            out_target_split_size=export_args.target_split_size,
            out_max_splits=export_args.max_splits,
            # I have not found splitting on in_features to be beneficial,
            # and it often leads to OOM so I set in_max_splits to 1
            in_target_split_size=1,
            in_max_splits=1,
        )

    model.eval()
    model.to(float_dtype)

    op_linear_quantizer_config = None
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

    compile_specs = CoreMLBackend.generate_compile_specs(  # pyre-fixme[16]
        minimum_deployment_target=ct.target.iOS18,
        compute_precision={
            torch.float16: ct.precision.FLOAT16,
            torch.float32: ct.precision.FLOAT32,
        }[float_dtype],
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

    input_manager = InputManager(
        n_layers=model.params.n_layers,
        max_batch_size=model.params.max_batch_size,
        n_kv_heads=model.params.n_kv_heads,
        max_seq_length=model.params.max_seq_len,
        head_dim=model.params.head_dim,
        use_cache_list=export_args.use_cache_list,
        seq_length=export_args.seq_length,
        dtype=float_dtype,
        minus_infinity=-30000,
        cache_size=export_args.cache_size,
    )
    example_inputs = input_manager.get_inputs(tokens=[0])

    ep = torch.export.export(model, example_inputs, strict=True)
    print("Exported program")
    print(ep)

    edge_manager = to_edge_with_preserved_ops(
        ep,
        preserve_ops=[
            torch.ops.aten.scaled_dot_product_attention.default,
            # preserve norm op for numerical stability
            torch.ops.aten.linalg_vector_norm.default,
            torch.ops.aten.reciprocal.default,
        ],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    print("Edge program")
    print(edge_manager.exported_program())

    for node in edge_manager.exported_program().graph_module.graph.nodes:
        print(node.name, node.target, node.args, node.kwargs)

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
