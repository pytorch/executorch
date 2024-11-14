# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import zipfile
from collections import defaultdict
from typing import Optional, Tuple

import torch


def flatten_args(args) -> tuple | list:
    flattened_args: list = []
    if isinstance(args, torch.Tensor):
        return [args]

    for arg in args:
        if isinstance(arg, (tuple, list)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    return tuple(flattened_args)


class GenericModelEvaluator:
    def __init__(
        self,
        model_name: str,
        fp32_model: torch.nn.Module,
        int8_model: torch.nn.Module,
        example_input: Tuple[torch.Tensor],
        tosa_output_path: Optional[str],
    ) -> None:
        self.model_name = model_name

        self.fp32_model = fp32_model
        self.int8_model = int8_model
        self.example_input = example_input

        if tosa_output_path:
            self.tosa_output_path = tosa_output_path
        else:
            self.tosa_output_path = None

    def get_model_error(self) -> defaultdict:
        """
        Returns a dict containing the following metrics between the outputs of the FP32 and INT8 model:
        - Maximum error
        - Maximum absolute error
        - Maximum percentage error
        - Mean absolute error
        """
        fp32_outputs = flatten_args(self.fp32_model(*self.example_input))
        int8_outputs = flatten_args(self.int8_model(*self.example_input))

        model_error_dict = defaultdict(list)

        for fp32_output, int8_output in zip(fp32_outputs, int8_outputs):
            difference = fp32_output - int8_output
            percentage_error = torch.div(difference, fp32_output) * 100
            model_error_dict["max_error"].append(torch.max(difference).item())
            model_error_dict["max_absolute_error"].append(
                torch.max(torch.abs(difference)).item()
            )
            model_error_dict["max_percentage_error"].append(
                torch.max(percentage_error).item()
            )
            model_error_dict["mean_absolute_error"].append(
                torch.mean(torch.abs(difference).float()).item()
            )

        return model_error_dict

    def get_compression_ratio(self) -> float:
        """Compute the compression ratio of the outputted TOSA flatbuffer."""
        with tempfile.NamedTemporaryFile(delete=True, suffix=".zip") as temp_zip:
            with zipfile.ZipFile(
                temp_zip.name, "w", compression=zipfile.ZIP_DEFLATED
            ) as f:
                f.write(self.tosa_output_path)

            compression_ratio = os.path.getsize(
                self.tosa_output_path
            ) / os.path.getsize(temp_zip.name)

        return compression_ratio

    def evaluate(self) -> dict[any]:
        model_error_dict = self.get_model_error()

        output_metrics = {"name": self.model_name, "metrics": dict(model_error_dict)}

        if self.tosa_output_path:
            # We know output_metrics["metrics"] is list since we just defined it, safe to ignore.
            # pyre-ignore[16]
            output_metrics["metrics"][
                "compression_ratio"
            ] = self.get_compression_ratio()

        return output_metrics
