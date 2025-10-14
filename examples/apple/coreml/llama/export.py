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

from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.export_util.utils import save_pte_program

from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
from torchao.utils import unwrap_tensor_subclass


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

    model.eval()
    model.to(float_dtype)

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

    # Quantization
    if export_args.embedding_quantize:
        bitwidth, group_size = export_args.embedding_quantize.split(",")
        bitwidth = int(bitwidth)
        assert bitwidth in [4, 8], "CoreML only supports 4-bit and 8-bit quantization"
        group_size = int(group_size)
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

    if export_args.coreml_quantize == "b4w":
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(32),
            ),
        )
    elif export_args.coreml_quantize == "c4w":
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerAxis(0),
            ),
        )

    compile_specs = CoreMLBackend.generate_compile_specs(  # pyre-fixme[16]
        minimum_deployment_target=ct.target.iOS18,
        compute_precision={
            torch.float16: ct.precision.FLOAT16,
            torch.float32: ct.precision.FLOAT32,
        }[float_dtype],
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,  # pyre-fixme[16]
    )
    partitioner = CoreMLPartitioner(  # pyre-fixme[16]
        compile_specs=compile_specs,
        take_over_mutable_buffer=False,
        skip_ops_for_coreml_delegation=[],
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

    model = unwrap_tensor_subclass(model)

    ep = torch.export.export(model, example_inputs, strict=True)
    print("Exported program")
    print(ep)

    edge_manager = to_edge_transform_and_lower(
        ep,
        partitioner=[partitioner],
    )

    print("Delegated program")
    print(format_delegated_graph(edge_manager.exported_program().graph_module))

    executorch_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    filename = save_pte_program(executorch_program, export_args.output_name)
    print(f"Saved Executorch program to local {filename}")


if __name__ == "__main__":
    main()  # pragma: no cover
