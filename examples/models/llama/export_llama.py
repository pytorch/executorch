# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# force=True to ensure logging while in debugger. Set up logger before any
# other imports.
import logging

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, force=True)

import argparse
import runpy
import sys

import torch

sys.setrecursionlimit(4096)


def parse_hydra_arg():
    """First parse out the arg for whether to use Hydra or the old CLI."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--hydra", action="store_true")
    args, remaining = parser.parse_known_args()
    return args.hydra, remaining


def main() -> None:
    seed = 42
    torch.manual_seed(seed)

    use_hydra, remaining_args = parse_hydra_arg()
    if use_hydra:
        # The import runs the main function of export_llama_hydra with the remaining args
        # under the Hydra framework.
        sys.argv = [arg for arg in sys.argv if arg != "--hydra"]
        print(f"running with {sys.argv}")
        runpy.run_module(
            "executorch.examples.models.llama.export_llama_hydra", run_name="__main__"
        )
    else:
        # Use the legacy version of the export_llama script which uses argsparse.
        from executorch.examples.models.llama.export_llama_args import (
            main as export_llama_args_main,
        )

        export_llama_args_main(remaining_args)


if __name__ == "__main__":
    main()  # pragma: no cover
