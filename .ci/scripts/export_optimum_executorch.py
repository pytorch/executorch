# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export an Optimum ExecuTorch model without eager MPS execution.

The Metal recipe lowers a CPU export to Metal and does not require the eager
model to reside on MPS. Keeping the eager model on CPU also avoids exercising
the experimental torchao FPA4 MPS operator during export.
"""

import argparse

from optimum.commands.export.executorch import parse_args_executorch
from optimum.exporters.executorch import main_export


def main() -> None:
    parser = argparse.ArgumentParser("Hugging Face Optimum ExecuTorch exporter")
    parse_args_executorch(parser)
    args = parser.parse_args()

    base_arguments = {"model", "output_dir", "task", "recipe"}
    export_arguments = {
        name: value
        for name, value in vars(args).items()
        if name not in base_arguments and value is not None
    }
    if export_arguments.get("device") == "mps":
        export_arguments["device"] = "cpu"

    main_export(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        task=args.task,
        recipe=args.recipe,
        **export_arguments,
    )


if __name__ == "__main__":
    main()
