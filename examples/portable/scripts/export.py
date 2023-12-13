# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

from executorch.exir.capture._config import ExecutorchBackendConfig

from ...models import MODEL_NAME_TO_MODEL
from ...models.model_factory import EagerModelFactory
from ..utils import export_to_exec_prog, save_pte_program


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")
    parser.add_argument(
        "-c",
        "--constant_segment",
        default=True,
        help="whether or not to store constants in a separate segment")
    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    config = ExecutorchBackendConfig(extract_constant_segment=args.constant_segment)
    prog = export_to_exec_prog(model, example_inputs, backend_config=config)
    save_pte_program(prog.buffer, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
