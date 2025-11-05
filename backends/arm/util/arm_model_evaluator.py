# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import random
import tempfile
import zipfile

from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Optional, Tuple

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
    """Return standard ImageNet 224x224 preprocessing transforms."""
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
    """Return a DataLoader over a deterministic, shuffled subset of size <= max_items.

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


def _load_imagenet_folder(directory: str) -> datasets.ImageFolder:
    """Shared helper to load an ImageNet-layout folder.

    Raises FileNotFoundError for a missing directory early to aid debugging.
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory: {directory} does not exist.")
    transform = _get_imagenet_224_transforms()
    return datasets.ImageFolder(directory_path, transform=transform)


class GenericModelEvaluator:
    """Base evaluator computing quantization error metrics and optional compression ratio.

    Subclasses can extend: provide calibration (get_calibrator) and override evaluate()
    to add domain specific metrics (e.g. top-1 / top-5 accuracy).
    """

    @staticmethod
    def evaluate_topk(
        model: Module,
        dataset: datasets.ImageFolder,
        batch_size: int,
        topk: int = 5,
        log_every: int = 50,
    ) -> Tuple[float, float]:
        """Evaluate model top-1 / top-k accuracy.

        Args:
            model: Torch module (should be in eval() mode prior to call).
            dataset: ImageFolder style dataset.
            batch_size: Batch size for evaluation.
            topk: Maximum k for accuracy (default 5).
            log_every: Log running accuracy every N batches.
        Returns:
            (top1_accuracy, topk_accuracy)
        """
        # Some exported / quantized models (torchao PT2E) disallow direct eval()/train().
        # Try to switch to eval mode, but degrade gracefully if unsupported.
        try:
            model.eval()
        except NotImplementedError:
            # Attempt to enable train/eval overrides if torchao helper is present.
            try:
                from torchao.quantization.pt2e.utils import (  # type: ignore
                    allow_exported_model_train_eval,
                )

                allow_exported_model_train_eval(model)
                try:
                    model.eval()
                except Exception:
                    logger.debug(
                        "Model eval still not supported after allow_exported_model_train_eval; proceeding without explicit eval()."
                    )
            except Exception:
                logger.debug(
                    "Model eval() unsupported and torchao allow_exported_model_train_eval not available; proceeding."
                )
        loaded_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        top1_correct = 0
        topk_correct = 0
        total = 0
        with torch.inference_mode():  # disable autograd + some backend optimizations
            for i, (image, target) in enumerate(loaded_dataset):
                prediction = model(image)
                topk_indices = torch.topk(prediction, k=topk, dim=1).indices
                # target reshaped for broadcasting
                target_view = target.view(-1, 1)
                top1_correct += (topk_indices[:, :1] == target_view).sum().item()
                topk_correct += (topk_indices == target_view).sum().item()
                batch_sz = image.size(0)
                total += batch_sz
                if (i + 1) % log_every == 0 or total == len(dataset):
                    logger.info(
                        "Eval progress: %d / %d  top1=%.4f top%d=%.4f",
                        total,
                        len(dataset),
                        top1_correct / total,
                        topk,
                        topk_correct / total,
                    )
        top1_accuracy = top1_correct / len(dataset)
        topk_accuracy = topk_correct / len(dataset)
        return top1_accuracy, topk_accuracy

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
        """Return per-output quantization error statistics.

        Metrics (lists per output tensor):
            max_error
            max_absolute_error
            max_percentage_error (safe-divided; zero fp32 elements -> 0%)
            mean_absolute_error
        """
        fp32_outputs, _ = tree_flatten(self.fp32_model(*self.example_input))
        int8_outputs, _ = tree_flatten(self.int8_model(*self.example_input))

        model_error_dict = defaultdict(list)

        for fp32_output, int8_output in zip(fp32_outputs, int8_outputs):
            difference = fp32_output - int8_output
            # Avoid divide by zero: elements where fp32 == 0 produce 0% contribution
            percentage_error = torch.where(
                fp32_output != 0,
                difference / fp32_output * 100,
                torch.zeros_like(difference),
            )
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
        return _load_imagenet_folder(directory)

    @staticmethod
    def get_calibrator(training_dataset_path: str) -> DataLoader:
        dataset = MobileNetV2Evaluator.__load_dataset(training_dataset_path)
        return _build_calibration_loader(dataset, 1000)

    @classmethod
    def from_config(
        cls,
        model_name: str,
        fp32_model: Module,
        int8_model: Module,
        example_input: Tuple[torch.Tensor],
        tosa_output_path: str | None,
        config: dict[str, Any],
    ) -> "MobileNetV2Evaluator":
        """Factory constructing evaluator from a config dict.

        Expected keys: batch_size, validation_dataset_path
        """
        return cls(
            model_name,
            fp32_model,
            int8_model,
            example_input,
            tosa_output_path,
            batch_size=config["batch_size"],
            validation_dataset_path=config["validation_dataset_path"],
        )

    def evaluate(self) -> dict[str, Any]:
        # Load dataset and compute top-1 / top-5
        dataset = MobileNetV2Evaluator.__load_dataset(self.__validation_set_path)
        top1_correct, top5_correct = GenericModelEvaluator.evaluate_topk(
            self.int8_model, dataset, self.__batch_size, topk=5
        )
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
        return _load_imagenet_folder(directory)

    @staticmethod
    def get_calibrator(training_dataset_path: str) -> DataLoader:
        dataset = DeiTTinyEvaluator.__load_dataset(training_dataset_path)
        return _build_calibration_loader(dataset, 1000)

    @classmethod
    def from_config(
        cls,
        model_name: str,
        fp32_model: Module,
        int8_model: Module,
        example_input: Tuple[torch.Tensor],
        tosa_output_path: str | None,
        config: dict[str, Any],
    ) -> "DeiTTinyEvaluator":
        """Factory constructing evaluator from a config dict.

        Expected keys: batch_size, validation_dataset_path
        """
        return cls(
            model_name,
            fp32_model,
            int8_model,
            example_input,
            tosa_output_path,
            batch_size=config["batch_size"],
            validation_dataset_path=config["validation_dataset_path"],
        )

    def evaluate(self) -> dict[str, Any]:
        # Load dataset and compute top-1 / top-5
        dataset = DeiTTinyEvaluator.__load_dataset(self.__validation_set_path)
        top1, top5 = GenericModelEvaluator.evaluate_topk(
            self.int8_model, dataset, self.__batch_size, topk=5
        )
        output = super().evaluate()
        output["metrics"]["accuracy"] = {"top-1": top1, "top-5": top5}
        return output


class ResNet18Evaluator(GenericModelEvaluator):
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
        return _load_imagenet_folder(directory)

    @staticmethod
    def get_calibrator(training_dataset_path: str) -> DataLoader:
        dataset = ResNet18Evaluator.__load_dataset(training_dataset_path)
        return _build_calibration_loader(dataset, 1000)

    @classmethod
    def from_config(
        cls,
        model_name: str,
        fp32_model: Module,
        int8_model: Module,
        example_input: Tuple[torch.Tensor],
        tosa_output_path: str | None,
        config: dict[str, Any],
    ) -> "ResNet18Evaluator":
        return cls(
            model_name,
            fp32_model,
            int8_model,
            example_input,
            tosa_output_path,
            batch_size=config["batch_size"],
            validation_dataset_path=config["validation_dataset_path"],
        )

    def evaluate(self) -> dict[str, Any]:
        dataset = ResNet18Evaluator.__load_dataset(self.__validation_set_path)
        top1, top5 = GenericModelEvaluator.evaluate_topk(
            self.int8_model, dataset, self.__batch_size, topk=5
        )
        output = super().evaluate()
        output["metrics"]["accuracy"] = {"top-1": top1, "top-5": top5}
        return output


evaluators: dict[str, type[GenericModelEvaluator]] = {
    "generic": GenericModelEvaluator,
    "mv2": MobileNetV2Evaluator,
    "deit_tiny": DeiTTinyEvaluator,
    "resnet18": ResNet18Evaluator,
}


def evaluator_calibration_data(
    evaluator_name: str,
    evaluator_config: str | None,
):
    evaluator = evaluators[evaluator_name]

    if hasattr(evaluator, "get_calibrator"):
        assert evaluator_config is not None

        config_path = Path(evaluator_config)
        with config_path.open() as f:
            config = json.load(f)

        # All current evaluators exposing calibration implement a uniform
        # static method signature: get_calibrator(training_dataset_path: str)
        # so we can call it generically without enumerating classes.
        return evaluator.get_calibrator(
            training_dataset_path=config["training_dataset_path"]
        )


def evaluate_model(
    model_name: str,
    intermediates: str,
    model_fp32: torch.nn.Module,
    model_int8: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str,
    evaluator_config: str | None,
) -> None:
    evaluator = evaluators[evaluator_name]

    intermediates_path = Path(intermediates)
    tosa_paths = list(intermediates_path.glob("*.tosa"))

    if evaluator.REQUIRES_CONFIG:
        assert evaluator_config is not None
        config_path = Path(evaluator_config)
        with config_path.open() as f:
            config = json.load(f)

        # Prefer a subclass provided from_config if available.
        if hasattr(evaluator, "from_config"):
            factory = cast(Any, evaluator.from_config)  # type: ignore[attr-defined]
            init_evaluator = factory(
                model_name,
                model_fp32,
                model_int8,
                example_inputs,
                str(tosa_paths[0]),
                config,
            )
        else:
            raise RuntimeError(
                f"Evaluator {evaluator_name} requires config but does not implement from_config()"
            )
    else:
        init_evaluator = evaluator(
            model_name, model_fp32, model_int8, example_inputs, str(tosa_paths[0])
        )

    quant_metrics = init_evaluator.evaluate()
    output_json_path = intermediates_path / "quant_metrics.json"

    with output_json_path.open("w") as json_file:
        json.dump(quant_metrics, json_file)
