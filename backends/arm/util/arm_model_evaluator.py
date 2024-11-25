# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import tempfile
import zipfile

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Logger for outputting progress for longer running evaluation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    REQUIRES_CONFIG = False

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

    def evaluate(self) -> dict[Any]:
        model_error_dict = self.get_model_error()

        output_metrics = {"name": self.model_name, "metrics": dict(model_error_dict)}

        if self.tosa_output_path:
            # We know output_metrics["metrics"] is list since we just defined it, safe to ignore.
            # pyre-ignore[16]
            output_metrics["metrics"][
                "compression_ratio"
            ] = self.get_compression_ratio()

        return output_metrics


class MobileNetV2Evaluator(GenericModelEvaluator):
    REQUIRES_CONFIG = True

    def __init__(
        self,
        model_name: str,
        fp32_model: Module,
        int8_model: Module,
        example_input: Tuple[torch.Tensor],
        tosa_output_path: str | None,
        batch_size: int,
        validation_dataset_path: str,
    ) -> None:
        super().__init__(
            model_name, fp32_model, int8_model, example_input, tosa_output_path
        )

        self.__batch_size = batch_size
        self.__validation_set_path = validation_dataset_path

    @staticmethod
    def __load_dataset(directory: str) -> datasets.ImageFolder:
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory: {directory} does not exist.")

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.484, 0.454, 0.403], std=[0.225, 0.220, 0.220]
                ),
            ]
        )
        return datasets.ImageFolder(directory_path, transform=transform)

    @staticmethod
    def get_calibrator(training_dataset_path: str) -> DataLoader:
        dataset = MobileNetV2Evaluator.__load_dataset(training_dataset_path)
        rand_indices = random.sample(range(len(dataset)), k=1000)

        # Return a subset of the dataset to be used for calibration
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, rand_indices),
            batch_size=1,
            shuffle=False,
        )

    def __evaluate_mobilenet(self) -> Tuple[float, float]:
        dataset = MobileNetV2Evaluator.__load_dataset(self.__validation_set_path)
        loaded_dataset = DataLoader(
            dataset,
            batch_size=self.__batch_size,
            shuffle=False,
        )

        top1_correct = 0
        top5_correct = 0

        for i, (image, target) in enumerate(loaded_dataset):
            prediction = self.int8_model(image)
            top1_prediction = torch.topk(prediction, k=1, dim=1).indices
            top5_prediction = torch.topk(prediction, k=5, dim=1).indices

            top1_correct += (top1_prediction == target.view(-1, 1)).sum().item()
            top5_correct += (top5_prediction == target.view(-1, 1)).sum().item()

            logger.info("Iteration: {}".format((i + 1) * self.__batch_size))
            logger.info(
                "Top 1: {}".format(top1_correct / ((i + 1) * self.__batch_size))
            )
            logger.info(
                "Top 5: {}".format(top5_correct / ((i + 1) * self.__batch_size))
            )

        top1_accuracy = top1_correct / len(dataset)
        top5_accuracy = top5_correct / len(dataset)

        return top1_accuracy, top5_accuracy

    def evaluate(self) -> dict[str, Any]:
        top1_correct, top5_correct = self.__evaluate_mobilenet()
        output = super().evaluate()

        output["metrics"]["accuracy"] = {"top-1": top1_correct, "top-5": top5_correct}
        return output
