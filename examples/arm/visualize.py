# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import model_explorer

from executorch.devtools.visualization.visualization_utils import (
    visualize_model_explorer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a model using model explorer."
    )
    parser.add_argument("model_path", type=str, help="Path to the model file.")
    args = parser.parse_args()

    config = model_explorer.config()
    (config.add_model_from_path(args.model_path))

    visualize_model_explorer(
        config=config,
        extensions=["tosa_adapter_model_explorer"],
    )


if __name__ == "__main__":
    main()
