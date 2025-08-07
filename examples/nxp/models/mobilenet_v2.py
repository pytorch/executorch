# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Iterator

import torch
import torchvision

from executorch.examples.models.mobilenet_v2 import MV2Model
from torch.utils.data import DataLoader
from torchvision import transforms


class MobilenetV2(MV2Model):

    def get_calibration_inputs(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor]]:
        """
        Returns an iterator for the Imagenette validation dataset, downloading it if necessary.

        Args:
            batch_size (int): The batch size for the iterator.

        Returns:
            iterator: An iterator that yields batches of images from the Imagnetette validation dataset.
        """
        dataloader = self.get_dataset(batch_size)

        # Return the iterator
        dataloader_iterable = itertools.starmap(
            lambda data, label: (data,), iter(dataloader)
        )

        # We want approximately 500 samples
        batch_count = 500 // batch_size
        return itertools.islice(dataloader_iterable, batch_count)

    def get_dataset(self, batch_size):
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
    model = MobilenetV2()
    dataloader = model.get_dataset(batch_size=1)
    samples = gather_samples_per_class_from_dataloader(
        dataloader, num_samples_per_class=2
    )

    torch.save(samples, "calibration_data.pt")


if __name__ == "__main__":
    generate_input_samples_file()
