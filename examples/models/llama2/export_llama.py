# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import argparse
import json
import logging
from json import JSONDecodeError
from pathlib import Path

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from ...portable.utils import export_to_edge, save_pte_program

from ..model_factory import EagerModelFactory

from .quantize import WeightOnlyInt8QuantHandler

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def quantize(model) -> torch.nn.Module:
    """
    Quantizes a model by converting all weights to int8.
    Args:
        model: A model to quantize.
    Returns:
        A quantized model.
    """
    model_int8 = WeightOnlyInt8QuantHandler(model)
    model_int8_state_dict = model_int8.create_quantized_state_dict()
    model_int8 = model_int8.convert_for_runtime()
    model_int8.load_state_dict(model_int8_state_dict)
    return model_int8


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    modelname = "llama2"
    ckpt_dir = Path(__file__).absolute().parent / "params"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")
    parser.add_argument(
        "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
    )
    parser.add_argument("-Q", "--quantize", default=None, action="store_true")

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=ckpt_dir / "demo_rand_params.pth",
        help="checkpoint.pth",
    )
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to epxort a model using kv cache",
    )
    parser.add_argument(
        "-p", "--params", default=ckpt_dir / "demo_config.json", help="config.json"
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        help='metadata string in json format. Example {"get_bos_id": 3, "get_eos_id": 3, "get_n_bos": 1, "get_n_eos": 2}',
    )

    parser.add_argument("-2", "--fairseq2", action="store_true")
    parser.add_argument("-H", "--half", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-X", "--xnnpack", action="store_true")

    args = parser.parse_args()

    model, example_inputs, _ = EagerModelFactory.create_model(
        "llama2",
        "Llama2Model",
        checkpoint=args.checkpoint,
        params=args.params,
        use_kv_cache=args.use_kv_cache,
        fairseq2=args.fairseq2,
    )

    if args.use_kv_cache:
        # seq length is fixed to 1 with current kv cache impl
        dynamic_shapes = None
    else:
        dim = torch.export.Dim("token_dim", max=model.params.max_seq_len - 1)
        dynamic_shapes = {"tokens": {1: dim}}

    if args.quantized_ckpt or args.quantize:
        modelname = f"{modelname}_q"
        model = quantize(model)

        if args.verbose:
            print(f"{modelname}:")
            print(f"{model}")

    if args.half:
        # only converts floating point dtypes to half
        # input and output are torch.long, so signature unchanged
        model.to(dtype=torch.half)
        modelname = f"{modelname}_h"
    else:
        # int8 quantization code has some bf16,
        # switch all to FP32
        model.to(dtype=torch.float)

    # metadata that we want to serialize into .pte file
    metadata = {
        "get_vocab_size": model.params.vocab_size,
        "get_max_seq_len": model.params.max_seq_len,
    }
    if args.metadata:
        try:
            extra = json.loads(args.metadata)
            for k, v in extra.items():
                metadata[k] = v
        except JSONDecodeError:
            logging.error("Invalid metadata, should be a valid JSON string")

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        edge_manager = export_to_edge(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            edge_constant_methods=metadata,
            edge_compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
    if args.xnnpack:
        edge_manager = edge_manager.to_backend(XnnpackPartitioner())
        modelname = f"xnnpack_{modelname}"

    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_constant_segment=True,
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
    print(
        "Required memory for activation in bytes: ",
        export_program._emitter_output.program.execution_plan[0].non_const_buffer_sizes,
    )
    save_pte_program(export_program.buffer, modelname, args.output_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
