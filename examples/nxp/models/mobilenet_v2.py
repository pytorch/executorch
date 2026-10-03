# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torchvision

from executorch.backends.nxp.tests.calibration_dataset import (
    CalibrationDataset,
    RandomCalibrationDataset,
)
from executorch.examples.models.mobilenet_v2 import MV2Model

from executorch.examples.nxp.models.nxp_test_base_model import NXPTestBaseModel
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.utils.data import DataLoader, Dataset
from torchao.quantization.pt2e import disable_observer
from torchvision import transforms
from tqdm import tqdm

log = logging.getLogger(__name__)


class MobileNetV2(NXPTestBaseModel):
    """MobileNet V2 model."""

    INPUT_SHAPE = (1, 3, 224, 224)
    IDX_TO_LABEL = {
        0: "bench703",
        1: "English_springer217",
        2: "cassette_player482",
        3: "chain_saw491",
        4: "church497",
        5: "French_horn566",
        6: "garbage_truck569",
        7: "gas_pump571",
        8: "golf_ball574",
        9: "parachute701",
    }

    TRAIN_HYPERPARAMETERS = {
        "num_epochs": 20,
        "batch_size": 64,
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-5,
    }

    @property
    def input_shape(self):
        return self.INPUT_SHAPE

    @property
    def labels(self):
        return self.IDX_TO_LABEL

    def _init_eager_model(self) -> torch.nn.Module:
        return MV2Model().get_eager_model().eval()

    def train_model_fn(
        self, model, num_epochs=None, batch_size=None, channels_last=False
    ):
        hyperparameters = self.TRAIN_HYPERPARAMETERS
        num_epochs = (
            num_epochs if num_epochs is not None else hyperparameters["num_epochs"]
        )
        batch_size = (
            batch_size if batch_size is not None else hyperparameters["batch_size"]
        )

        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=hyperparameters["lr"],
            momentum=hyperparameters["momentum"],
            weight_decay=hyperparameters["weight_decay"],
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        log.warning("Starting training...")

        data = self.get_qat_train_inputs(batch_size=batch_size)
        for nepoch in range(num_epochs):
            for samples, labels in tqdm(data):
                if channels_last:
                    samples = samples.to(memory_format=torch.channels_last)

                optimizer.zero_grad()
                outputs = model(samples)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if nepoch >= 15:
                model.apply(disable_observer)

            # freeze BN stats
            if nepoch >= 18:
                model.apply(freeze_bn_stats)

        return model

    def _init_dataset(self) -> Dataset:
        if self._use_random_dataset:
            num_classes = len(self.labels)
            sample_shape = tuple(self.input_shape)[1:]
            return RandomCalibrationDataset(
                self._num_samples, sample_shape, num_classes, self._balanced_dataset
            )
        else:
            dataset = CalibrationDataset(self._dataset_path)
            # Dataset was generated with batch size dim, but we need examples without it
            dataset.examples = [
                (data.squeeze(0), label) for (data, label) in dataset.examples
            ]
            return dataset


def get_dataloader(batch_size):
    # Define data transformations
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet stats
        ]
    )

    dataset = torchvision.datasets.Imagenette(
        root="./data", split="val", transform=data_transforms, download=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return dataloader


def gather_samples_per_class_from_dataloader(
    dataloader, num_samples_per_class=10
) -> list[tuple]:
    """
    Gathers a specified number of samples for each class from a DataLoader.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader object.
        num_samples_per_class (int): The number of samples to gather for each class. Defaults to 10.

    Returns:
        samples: A list of (sample, label) tuples.
    """

    if not isinstance(dataloader, DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader object")
    if not isinstance(num_samples_per_class, int) or num_samples_per_class <= 0:
        raise ValueError("num_samples_per_class must be a positive integer")

    labels = sorted(
        set([label for _, label in dataloader.dataset])
    )  # Get unique labels from the dataset
    samples_per_label = {label: [] for label in labels}  # Initialize dictionary

    for sample, label in dataloader:
        label = label.item()
        if len(samples_per_label[label]) < num_samples_per_class:
            samples_per_label[label].append((sample, label))

    samples = []

    for label in labels:
        samples.extend(samples_per_label[label])

    return samples


def generate_input_samples_file():
    """Generate data for MobileNet V2 model from Imagenette dataset.
    Generated file then can be passed as dataset_path parameter to MobileNetV2 model.
    """
    dataloader = get_dataloader(batch_size=1)
    samples = gather_samples_per_class_from_dataloader(
        dataloader, num_samples_per_class=2
    )

    torch.save(samples, "calibration_data.pt")


if __name__ == "__main__":
    generate_input_samples_file()
