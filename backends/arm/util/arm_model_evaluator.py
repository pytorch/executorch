# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

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
from torch.utils._pytree import tree_flatten
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import-untyped]


# Logger for outputting progress for longer running evaluation
logger = logging.getLogger(__name__)
# Explicitly set logging level: MLETORCH-893
logger.setLevel(logging.INFO)


# ImageNet 224x224 transforms (Resize->CenterCrop->ToTensor->Normalize)
# If future models require different preprocessing, extend this helper accordingly.
def _get_imagenet_224_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.484, 0.454, 0.403], std=[0.225, 0.220, 0.220]),
        ]
    )


def _build_calibration_loader(
    dataset: datasets.ImageFolder, max_items: int
) -> DataLoader:
    """Return a DataLoader over a deterministic shuffled subset of size <= max_items.

    Shuffles with seed: ARM_EVAL_CALIB_SEED (int) or default 1337; then selects first k and
    sorts indices to keep enumeration order stable while content depends on seed.
    """
    k = min(max_items, len(dataset))
    seed_env = os.getenv("ARM_EVAL_CALIB_SEED")
    default_seed = 1337
    if seed_env is not None:
        try:
            seed = int(seed_env)
        except ValueError:
            logger.warning(
                "ARM_EVAL_CALIB_SEED is not an int (%s); using default seed %d",
                seed_env,
                default_seed,
            )
            seed = default_seed
    else:
        seed = default_seed
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = sorted(indices[:k])
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected), batch_size=1, shuffle=False
    )


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
            self.tosa_output_path = ""

    def get_model_error(self) -> defaultdict:
        """
        Returns a dict containing the following metrics between the outputs of the FP32 and INT8 model:
        - Maximum error
        - Maximum absolute error
        - Maximum percentage error
        - Mean absolute error
        """
        fp32_outputs, _ = tree_flatten(self.fp32_model(*self.example_input))
        int8_outputs, _ = tree_flatten(self.int8_model(*self.example_input))

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

    def evaluate(self) -> dict[str, Any]:
        model_error_dict = self.get_model_error()

        output_metrics = {"name": self.model_name, "metrics": dict(model_error_dict)}

        if self.tosa_output_path:
            # We know output_metrics["metrics"] is list since we just defined it, safe to ignore.
            # pyre-ignore[16]
            output_metrics["metrics"][  # type: ignore[index]
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
        transform = _get_imagenet_224_transforms()
        return datasets.ImageFolder(directory_path, transform=transform)

    @staticmethod
    def get_calibrator(training_dataset_path: str) -> DataLoader:
        dataset = MobileNetV2Evaluator.__load_dataset(training_dataset_path)
        return _build_calibration_loader(dataset, 1000)

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


class DeiTTinyEvaluator(GenericModelEvaluator):
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
        transform = _get_imagenet_224_transforms()
        return datasets.ImageFolder(directory_path, transform=transform)

    @staticmethod
    def get_calibrator(training_dataset_path: str) -> DataLoader:
        dataset = DeiTTinyEvaluator.__load_dataset(training_dataset_path)
        return _build_calibration_loader(dataset, 1000)

    def __evaluate_classifier(self) -> Tuple[float, float]:
        dataset = DeiTTinyEvaluator.__load_dataset(self.__validation_set_path)
        loaded_dataset = DataLoader(
            dataset,
            batch_size=self.__batch_size,
            shuffle=False,
        )
        top1_correct = 0
        top5_correct = 0
        for i, (image, target) in enumerate(loaded_dataset):
            prediction = self.int8_model(image)
            topk = torch.topk(prediction, k=5, dim=1).indices
            top1_prediction = topk[:, :1]
            top5_prediction = topk
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
        top1, top5 = self.__evaluate_classifier()
        output = super().evaluate()
        output["metrics"]["accuracy"] = {"top-1": top1, "top-5": top5}
        return output
