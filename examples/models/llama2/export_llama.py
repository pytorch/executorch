# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import argparse
import logging
from pathlib import Path

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from ...portable.utils import export_to_edge, save_pte_program

from ..model_factory import EagerModelFactory

from .quantize import WeightOnlyInt8QuantHandler

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


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
    )

    if args.use_kv_cache:
        # seq length is fixed to 1 with current kv cache impl
        dynamic_shapes = None
    else:
        dim = torch.export.Dim("token_dim", max=model.params.max_seq_len - 1)
        dynamic_shapes = {"tokens": {1: dim}}

    if args.half:
        # only converts floating point dtypes to half
        # input and output are torch.long, so signature unchanged
        model.to(dtype=torch.half)
        modelname = f"{modelname}_h"

    if args.quantized_ckpt or args.quantize:
        modelname = f"{modelname}_q"
        model_int8 = WeightOnlyInt8QuantHandler(model)
        model_int8_state_dict = model_int8.create_quantized_state_dict()

        if args.quantized_ckpt:
            torch.save(model_int8_state_dict, args.quantized_ckpt)

        if args.verbose:
            print("*******quantized checkpoint********")
            for key, data in model_int8_state_dict.items():
                print(f"{key}")

        model_int8 = model_int8.convert_for_runtime()
        model_int8.load_state_dict(model_int8_state_dict)
        model = model_int8

        if args.verbose:
            print(f"{modelname}:")
            print(f"{model}")

    if args.half:
        # only converts floating point dtypes to half
        # input and output are torch.long, so signature unchanged
        model.to(dtype=torch.half)
    else:
        # int8 quantization code has some bf16,
        # switch all to FP32
        model.to(dtype=torch.float)

    dim = torch.export.Dim("token_dim", max=model.params.max_seq_len - 1)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        edge_manager = export_to_edge(
            model,
            example_inputs,
            dynamic_shapes={"tokens": {1: dim}},
            #            edge_constant_methods=params,
            edge_compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )

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

    if args.xnnpack:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            xnn_model = export_to_edge(
                model,
                example_inputs,
                dynamic_shapes={"tokens": {1: dim}},
                edge_compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                ),
            )
            xnn_partitioned = xnn_model.to_backend(XnnpackPartitioner())
            exec_prog = xnn_partitioned.to_executorch()

        with open(f"./xnnpack_{modelname}.pte", "wb") as file:
            file.write(exec_prog.buffer)


if __name__ == "__main__":
    main()  # pragma: no cover
