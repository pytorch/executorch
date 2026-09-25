# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np

import torch

from executorch.backends.nxp.tests.calibration_dataset import (
    CalibrationDataset,
    RandomCalibrationDataset,
)
from executorch.examples.models.mlperf_tiny import DeepAutoEncoderModel
from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel
from torch.utils.data import Dataset
from torchao.quantization.pt2e import disable_observer
from tqdm import tqdm

log = logging.getLogger(__name__)


class MLPerfTinyAnomalyDetection(MLPerfTinyModel):
    """MLPerf Tiny Anomaly Detection model (DeepAutoEncoder).

    The input shape is set to (98, 640) as the reference internal model was trained with this shape.
    The dataset is generated for this shape and thus for calibration the data needs to be flattened/unbatched
    first. The model is not a classification model so a different loss function than the other MLPerf Tiny models
    is used. For class interpretation in output comparison a get_class_from_reconstruction_error() post-processing
    function is used.
    """

    INPUT_SHAPE = (98, 640)
    IDX_TO_LABEL = {0: "normal", 1: "anomaly"}
    CLASS_THRESHOLD = 17.0  # Empirically chosen
    _batch_size = INPUT_SHAPE[0]

    # DeepAutoEncoder specific QAT training hyperparameters.
    TRAIN_HYPERPARAMETERS = {
        "num_epochs": 15,
        "batch_size": _batch_size,
        "lr": 5e-6,
        "eps": 1e-7,
    }

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        use_random_dataset: bool = False,
        num_samples: int | None = None,
        num_workers: int = 4,
    ):
        self._dataset_flattened = False
        super().__init__(
            dataset_path=dataset_path,
            use_random_dataset=use_random_dataset,
            num_samples=num_samples,
            num_workers=num_workers,
        )

    @property
    def input_shape(self):
        return self.INPUT_SHAPE

    @property
    def labels(self):
        return self.IDX_TO_LABEL

    def _flatten_dataset(self):
        flat_xs = []
        flat_ys = []

        for sample in self.dataset.examples:
            input_batch, single_label = sample
            batch_size = input_batch.shape[0]

            if batch_size != self._batch_size:
                logging.warning(
                    f"The dataset was exported with `batch_size={batch_size}` "
                    f"which is different than `self.batch_size={self._batch_size}`. "
                    "This may produce unexpected behavior. "
                    "Note: The `self._batch_size` should match "
                    "the one from `prepare_calibration_data.py` in MLPerfTiny."
                )

            flat_xs.extend(list(input_batch))
            flat_ys.extend([single_label] * batch_size)

        self._dataset_flattened = True
        self.dataset.examples = list(zip(flat_xs, flat_ys))

    def get_class_from_reconstruction_error(
        self,
        preds: np.ndarray,
        pred_path: str,
        input_parent_path: str,
    ) -> np.ndarray:
        sample_name = pred_path.split("/")[-2]
        input_path = os.path.join(input_parent_path, sample_name)
        inps = np.fromfile(input_path, dtype=preds.dtype).reshape(preds.shape)

        # high error -> anomaly (above threshold)
        return np.mean(np.square(inps - preds), axis=1) > self.CLASS_THRESHOLD

    def _init_dataset(self) -> Dataset:
        if self._use_random_dataset:
            num_classes = len(self.labels)
            sample_shape = tuple(self.input_shape)
            return RandomCalibrationDataset(
                self._num_samples, sample_shape, num_classes
            )
        else:
            return CalibrationDataset(self._dataset_path)

    # noinspection PyMethodMayBeStatic
    def _init_eager_model(self) -> torch.nn.Module:
        return DeepAutoEncoderModel().get_eager_model()

    def get_calibration_inputs(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor]]:
        if not self._dataset_flattened:
            self._flatten_dataset()  # For Anomaly detection data have to flattened/unbatched first
        return super().get_calibration_inputs(batch_size)

    def get_qat_train_inputs(
        self, batch_size: int = 5, dataset_portion: float = 0.1
    ) -> Iterator[tuple[torch.Tensor]]:
        if not self._dataset_flattened:
            self._flatten_dataset()  # For Anomaly detection data have to flattened/unbatched first for calibration
        return super().get_qat_train_inputs(
            batch_size=batch_size, dataset_portion=dataset_portion
        )

    def train_model_fn(
        self, model, num_epochs=None, batch_size=None, channels_last=False
    ):
        assert not channels_last, "This model does not support channels last."
        hyperparameters = self.TRAIN_HYPERPARAMETERS
        num_epochs = (
            num_epochs if num_epochs is not None else hyperparameters["num_epochs"]
        )
        batch_size = (
            batch_size if batch_size is not None else hyperparameters["batch_size"]
        )

        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=hyperparameters["lr"],
            eps=hyperparameters["eps"],
        )
        loss_fn = torch.nn.MSELoss()

        log.warning("Starting training...")

        data = self.get_qat_train_inputs(batch_size=batch_size)
        for nepoch in range(num_epochs):
            for samples, labels in tqdm(data):
                # Skip anomaly samples to evaluate using reconstruction error
                if sum(labels) > 0:
                    continue

                optimizer.zero_grad()
                outputs = model(samples)
                loss = loss_fn(outputs, samples)
                loss.backward()
                optimizer.step()

            if nepoch >= num_epochs / 3:
                model.apply(disable_observer)

        return model
