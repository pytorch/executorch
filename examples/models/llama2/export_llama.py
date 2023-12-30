# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

from memory_profiler import profile
import argparse
import logging
from pathlib import Path

import torch
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig

from ...portable.utils import export_to_edge, save_pte_program

from ..model_factory import EagerModelFactory


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    ckpt_dir = Path(__file__).absolute().parent / "params"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=ckpt_dir / "demo_rand_params.pth",
        help="checkpoint.pth",
    )
    parser.add_argument(
        "-p", "--params", default=ckpt_dir / "demo_config.json", help="config.json"
    )

    args = parser.parse_args()

    model, example_inputs = EagerModelFactory.create_model(
        "llama2", "Llama2Model", checkpoint=args.checkpoint, params=args.params
    )

    dim = torch.export.Dim("token_dim", max=model.params.max_seq_len - 1)

    print("export_llama: export_to_edge")
    edge_manager = export_to_edge(
        model,
        example_inputs,
        dynamic_shapes={"tokens": {1: dim}},
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    print("export_llama: to_executorch")
    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(extract_constant_segment=False)
    )
    print("export_llama: save_pte_program")
    save_pte_program(export_program.buffer, "llama2", args.output_dir)
    # model.forward(input)


if __name__ == "__main__":
    main()  # pragma: no cover
