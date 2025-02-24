# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import argparse
import json

import sys

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

sys.path.insert(0, ".")
from llama_transformer import InputManager, ModelArgs, Transformer


class SplitLinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, target_split_size, max_splits):
        super(SplitLinearModule, self).__init__()
        num_splits = max(out_features // target_split_size, 1)
        if num_splits > max_splits:
            num_splits = max_splits

        self.split_size = out_features // num_splits
        self.split_remainder = out_features % num_splits
        self.splits = torch.nn.ModuleList(
            [torch.nn.Linear(in_features, self.split_size) for _ in range(num_splits)]
        )
        print(
            f"Splitting out_features={out_features} into {num_splits} of size {self.split_size}"
        )
        if self.split_remainder > 0:
            print(
                f"Warning: remainder {self.split_remainder} after splitting out_features={out_features} into {num_splits} of size {self.split_size}"
            )
            self.splits.append(torch.nn.Linear(in_features, self.split_remainder))

    def split_sizes(self):
        return [split.out_features for split in self.splits]

    def forward(self, x):
        return torch.cat([split(x) for split in self.splits], dim=-1)


def replace_linear_with_split_linear(model, target_split_size, max_splits):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            new_module = SplitLinearModule(
                module.in_features, module.out_features, target_split_size, max_splits
            )
            split_sizes = new_module.split_sizes()
            if module.bias is not None:
                split_bias = module.bias.split(split_sizes)
            split_weights = module.weight.split(split_sizes, dim=0)
            for i, split in enumerate(new_module.splits):
                split.weight = torch.nn.Parameter(split_weights[i])
                if module.bias is not None:
                    split.bias = torch.nn.Parameter(split_bias[i])
                else:
                    split.bias = None
            setattr(model, name, new_module)
        else:
            replace_linear_with_split_linear(module, target_split_size, max_splits)


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

    export_args = parser.parse_args()
    params_path = export_args.params
    checkpoint_path = export_args.checkpoint

    # Load model args
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    args = ModelArgs(
        max_seq_len=export_args.max_seq_length,
        generate_full_logits=False,
        use_cache_list=export_args.use_cache_list,
        **params,
    )

    with torch.device("meta"):
        model = Transformer(args)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
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

    if export_args.target_split_size is not None:
        replace_linear_with_split_linear(
            model, export_args.target_split_size, export_args.max_splits
        )

    model = model.to(float_dtype)

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

    input_manager = InputManager(
        n_layers=args.n_layers,
        max_batch_size=args.max_batch_size,
        n_kv_heads=args.n_kv_heads,
        max_seq_length=args.max_seq_len,
        head_dim=args.head_dim,
        use_cache_list=export_args.use_cache_list,
        seq_length=export_args.seq_length,
        dtype=float_dtype,
        minus_infinity=-30000,
        cache_size=export_args.cache_size,
    )
    example_inputs = input_manager.get_inputs(tokens=[0])

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
