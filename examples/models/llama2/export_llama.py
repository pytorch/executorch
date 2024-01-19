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
    parser.add_argument("-h", "--half", action="store_true")

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

    if args.quantized_ckpt or args.quantize:
        model_int8 = WeightOnlyInt8QuantHandler(model)
        model_int8_state_dict = model_int8.create_quantized_state_dict()

        if args.quantized_ckpt:
            torch.save(model_int8_state_dict, args.quantized_ckpt)

        model_int8 = model_int8.convert_for_runtime()
        model_int8.load_state_dict(model_int8_state_dict)
        model = model_int8

    dim = torch.export.Dim("token_dim", max=model.params.max_seq_len - 1)

    edge_manager = export_to_edge(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
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
    save_pte_program(export_program.buffer, "llama2", args.output_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
