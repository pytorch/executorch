# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 - 2026 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from executorch.devtools.visualization.visualization_utils import (
    visualize_with_clusters,
)
from executorch.exir import ExecutorchProgramManager


def save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> str:
    if model_name.endswith(".pte"):
        filename = model_name
        visualize_file_name = f"{model_name}.json"
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")
        visualize_file_name = os.path.join(output_dir, f"{model_name}.json")
    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")

    visualize_with_clusters(prog.exported_program(), visualize_file_name, False)
    return filename
