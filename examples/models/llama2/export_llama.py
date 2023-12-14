# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import argparse
import logging

from executorch.exir.capture._config import ExecutorchBackendConfig

from ...portable.utils import export_to_exec_prog, save_pte_program

from ..model_factory import EagerModelFactory


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")
    parser.add_argument(
        "-c", "--checkpoint", default="demo_rand_params.pth", help="checkpoint.pth"
    )
    parser.add_argument(
        "-p", "--params", default="demo_config.json", help="config.json"
    )

    args = parser.parse_args()

    model, example_inputs = EagerModelFactory.create_model(
        "llama2", "Llama2Model", checkpoint=args.checkpoint, params=args.params
    )

    prog = export_to_exec_prog(
        model,
        example_inputs,
        backend_config=ExecutorchBackendConfig(extract_constant_segment=True),
    )

    save_pte_program(prog.buffer, "llama2", args.output_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
