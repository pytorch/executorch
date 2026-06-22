# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import tempfile
import zipfile

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


class Evaluator:
    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError


class NumericalModelEvaluator(Evaluator):
    """Evaluator computing numerical error metrics."""

    def __init__(
        self,
        model_name: str,
        ref_model: torch.nn.Module,
        eval_model: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        eval_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._model_name = model_name
        self._ref_model = ref_model
        self._eval_model = eval_model
        self._example_inputs = example_inputs
        self._eval_dtype = eval_dtype

    def evaluate(self) -> dict[str, Any]:
        """Return per-output error statistics.

        Metrics (lists per output tensor):
            * max_error
            * max_absolute_error
            * max_percentage_error (safe-divided; zero ref elements -> 0%)
            * mean_absolute_error

        """
        if self._eval_dtype is not None:
            eval_inputs = tuple(
                inp.to(self._eval_dtype) for inp in self._example_inputs
            )
        else:
            eval_inputs = self._example_inputs

        ref_outputs, _ = tree_flatten(self._ref_model(*self._example_inputs))
        eval_outputs, _ = tree_flatten(self._eval_model(*eval_inputs))

        metrics = self._get_model_error(ref_outputs, eval_outputs)

        return metrics

    @staticmethod
    def _get_model_error(ref_outputs, eval_outputs) -> dict[str, Any]:
        metrics = {}

        for ref_output, eval_output in zip(ref_outputs, eval_outputs):
            difference = ref_output - eval_output
            # Avoid divide by zero: elements where ref_output == 0 produce 0% contribution
            percentage_error = torch.where(
                ref_output != 0,
                difference / ref_output * 100,
                torch.zeros_like(difference),
            )

            metrics["max_error"] = torch.max(difference).item()
            metrics["max_absolute_error"] = torch.max(torch.abs(difference)).item()
            metrics["max_percentage_error"] = torch.max(percentage_error).item()
            metrics["mean_absolute_error"] = torch.mean(
                torch.abs(difference).float()
            ).item()

        return metrics


class ImageNetEvaluator(Evaluator):
    """Evaluator computing accuracy for ImageNet-style classifiers.

    Provides dataset loading and a standard `evaluate` that computes
    top-1/top-5 accuracy.

    """

    def __init__(
        self,
        model_name: str,
        eval_model: Module,
        batch_size: int,
        validation_dataset_path: str,
        eval_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._model_name = model_name
        self._eval_model = eval_model
        self._batch_size = batch_size
        self._validation_set_path = validation_dataset_path
        self._eval_dtype = eval_dtype

    def evaluate(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        dataset = ImageNetEvaluator.load_imagenet_folder(self._validation_set_path)
        logger.debug(
            "Starting ImageNet evaluation for model '%s' on dataset '%s' with %d samples.",
            self._model_name,
            self._validation_set_path,
            len(dataset),
        )

        top1, top5 = self._evaluate_topk(dataset, topk=5)
        metrics["accuracy"] = {"top-1": top1, "top-5": top5}

        return metrics

    def _evaluate_topk(
        self,
        dataset: datasets.ImageFolder,
        topk: int = 5,
        log_every: int = 50,
    ) -> Tuple[float, float]:
        """Evaluate model top-1 / top-k accuracy.

        Args:
            dataset: ImageFolder style dataset.
            topk: Maximum k for accuracy (default 5).
            log_every: Log running accuracy every N batches.
        Returns:
            (top1_accuracy, topk_accuracy)

        """
        # Some exported / quantized models (torchao PT2E) disallow direct eval()/train().
        # Try to switch to eval mode, but degrade gracefully if unsupported.
        try:
            self._eval_model.eval()
        except NotImplementedError:
            # Attempt to enable train/eval overrides if torchao helper is present.
            try:
                from torchao.quantization.pt2e.utils import (  # type: ignore
                    allow_exported_model_train_eval,
                )

                allow_exported_model_train_eval(self._eval_model)
                try:
                    self._eval_model.eval()
                except Exception:
                    logger.debug(
                        "Model eval still not supported after allow_exported_model_train_eval; proceeding without explicit eval()."
                    )
            except Exception:
                logger.debug(
                    "Model eval() unsupported and torchao allow_exported_model_train_eval not available; proceeding."
                )
        loaded_dataset = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
        top1_correct = 0
        topk_correct = 0
        total = 0
        with torch.inference_mode():  # disable autograd + some backend optimizations
            for i, (image, target) in enumerate(loaded_dataset):
                if self._eval_dtype is not None:
                    image = image.to(self._eval_dtype)

                prediction = self._eval_model(image)
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

    @staticmethod
    def load_imagenet_folder(directory: str) -> datasets.ImageFolder:
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory: {directory} does not exist.")
        transform = ImageNetEvaluator._get_imagenet_224_transforms()
        return datasets.ImageFolder(directory_path, transform=transform)

    # ImageNet 224x224 transforms (Resize->CenterCrop->ToTensor->Normalize)
    # If future models require different preprocessing, extend this helper accordingly.
    @staticmethod
    def _get_imagenet_224_transforms():
        """Return standard ImageNet 224x224 preprocessing transforms."""
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.484, 0.454, 0.403], std=[0.225, 0.220, 0.220]
                ),
            ]
        )


class FileCompressionEvaluator(Evaluator):
    """Evaluator computing compression ratio for a TOSA flatbuffer."""

    def __init__(self, model_name: str, tosa_output_path: str) -> None:
        self._model_name = model_name
        self._tosa_output_path = tosa_output_path

    def evaluate(self) -> dict[str, Any]:
        return {
            "compression_ratio": self._get_compression_ratio(self._tosa_output_path)
        }

    @staticmethod
    def _get_compression_ratio(tosa_output_path: str) -> float:
        """Compute the compression ratio of the outputted TOSA flatbuffer."""
        with tempfile.NamedTemporaryFile(delete=True, suffix=".zip") as temp_zip:
            with zipfile.ZipFile(
                temp_zip.name, "w", compression=zipfile.ZIP_DEFLATED
            ) as f:
                f.write(tosa_output_path)

            compression_ratio = os.path.getsize(tosa_output_path) / os.path.getsize(
                temp_zip.name
            )

        return compression_ratio
