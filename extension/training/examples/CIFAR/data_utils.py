# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import os
import pickle
import typing
from collections import defaultdict

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


class BalancedCIFARDataset(Dataset):
    """
    Custom dataset class to load balanced
    CIFAR-10 data from binary file.
    """

    def __init__(
        self,
        data_path: str,
        transform: typing.Optional[torchvision.transforms.Compose] = None,
    ) -> None:
        """
        Args:
            data_path: Path to the balanced dataset binary file
            transform: Optional transformation to be applied on a sample
        """
        self.data = []
        self.labels = []

        # Read binary format: 1 byte label + 3072 bytes image data per record
        with open(data_path, "rb") as f:
            while True:
                # Read label (1 byte)
                label_byte = f.read(1)
                if not label_byte:  # End of file
                    break
                label = int.from_bytes(label_byte, byteorder="big")

                # Read image data (3 * 32 * 32 = 3072 bytes)
                image_bytes = f.read(3072)
                if len(image_bytes) != 3072:
                    break  # Incomplete record

                # Convert bytes to numpy array
                image_data = np.frombuffer(image_bytes, dtype=np.uint8)

                self.data.append(image_data)
                self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.transform = transform

        print(f"Loaded {len(self.data)} images from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> typing.Tuple[Image.Image, int]:
        # Reshape from (3072,) to (32, 32, 3) and convert to PIL Image
        image_data = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image_data)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_balanced_cifar_dataset(
    data_batch_path: str = "./data/cifar-10/cifar-10-batches-py/data_batch_1",
    output_path: str = "./data/cifar-10/extracted_data/train_data.bin",
    images_per_class: int = 100,
) -> str:
    """
    Reads CIFAR-10 data from data_batch_1 file and creates a balanced dataset
    with specified number of images per class, saved in binary format
    compatible with Android.

    Args:
        data_batch_path: Path to the CIFAR-10 data_batch_1 file
        output_path: Path where the balanced dataset will be saved
        images_per_class: Number of images to extract per class (default: 100)
    """
    # Load the CIFAR-10 data batch
    with open(data_batch_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    # Extract data and labels
    data = data_dict[b"data"]  # Shape: (10000, 3072)
    labels = data_dict[b"labels"]  # List of 10000 labels

    # Group images by class
    class_images = defaultdict(list)
    class_labels = defaultdict(list)

    for i, label in enumerate(labels):
        if len(class_images[label]) < images_per_class:
            class_images[label].append(data[i])
            class_labels[label].append(label)

    # Combine all selected images and labels
    selected_data = []
    selected_labels = []

    for class_id in range(10):  # CIFAR-10 has 10 classes (0-9)
        if class_id in class_images:
            selected_data.extend(class_images[class_id])
            selected_labels.extend(class_labels[class_id])
            print(
                f"Class {class_id}: " f"{len(class_images[class_id])} images selected"
            )

    # Convert to numpy arrays
    selected_data = np.array(selected_data, dtype=np.uint8)
    selected_labels = np.array(selected_labels, dtype=np.uint8)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save in binary format compatible with Android CIFAR-10 reader
    # Format: 1 byte label + 3072 bytes image data per record
    with open(output_path, "wb") as f:
        for i in range(len(selected_data)):
            # Write label as single byte
            f.write(bytes([selected_labels[i]]))
            # Write image data (3072 bytes)
            f.write(selected_data[i].tobytes())

    print(f"Balanced dataset saved to {output_path}")
    print(f"Total images: {len(selected_data)}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    print(f"Expected size: {len(selected_data) * (1 + 3072)} bytes")
    return output_path


def get_data_loaders(
    batch_size: int = 4,
    num_workers: int = 2,
    data_dir: str = "./data",
    use_balanced_dataset: bool = True,
    images_per_class: int = 100,
) -> typing.Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_dir: Root directory for data
        use_balanced_dataset: Whether to use balanced dataset or
                              standard CIFAR-10
        images_per_class: Number of images per class for balanced dataset
    """
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    if use_balanced_dataset:
        # Download CIFAR-10 first to ensure the raw data exists
        print("Downloading CIFAR-10 dataset...")
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

        # The actual path where torchvision stores CIFAR-10 data
        cifar_data_dir = os.path.join(data_dir, "cifar-10-batches-py")

        # Create balanced dataset if it doesn't exist
        balanced_data_path = os.path.join(
            data_dir, "cifar-10/extracted_data/train_data.bin"
        )
        data_batch_path = os.path.join(cifar_data_dir, "data_batch_1")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(balanced_data_path), exist_ok=True)

        # Create balanced dataset if it doesn't exist
        if not os.path.exists(balanced_data_path):
            print("Creating balanced train dataset...")
            create_balanced_cifar_dataset(
                data_batch_path=data_batch_path,
                output_path=balanced_data_path,
                images_per_class=images_per_class,
            )

        # Use balanced dataset for training
        trainset = BalancedCIFARDataset(balanced_data_path, transform=transforms)

        indices = torch.randperm(len(trainset)).tolist()

        train_subset = Subset(trainset, indices)

        balanced_test_data_path = os.path.join(
            data_dir, "cifar-10/extracted_data/test_data.bin"
        )
        test_data_batch_path = os.path.join(cifar_data_dir, "test_batch")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(balanced_test_data_path), exist_ok=True)
        # Create balanced dataset if it doesn't exist
        if not os.path.exists(balanced_test_data_path):
            print("Creating balanced test dataset...")
            create_balanced_cifar_dataset(
                data_batch_path=test_data_batch_path,
                output_path=balanced_test_data_path,
                images_per_class=images_per_class,
            )
        # Use balanced dataset for testing
        test_set = BalancedCIFARDataset(balanced_test_data_path, transform=transforms)

    else:
        # Use standard CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transforms
        )

        train_set_indices = torch.randperm(len(trainset)).tolist()

        train_subset = Subset(trainset, train_set_indices)

        # Test set always uses standard CIFAR-10
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def count_images_per_class(loader: DataLoader) -> typing.Dict[int, int]:
    """
    Count the number of images per class in a DataLoader.

    This function iterates through a DataLoader and counts how many images
    belong to each class based on their labels.

    Args:
        loader (DataLoader): The DataLoader containing image-label pairs

    Returns:
        Dict[int, int]: A dictionary mapping class IDs to their counts
    """
    class_counts = defaultdict(int)
    for _, labels in loader:
        for label in labels:
            class_counts[label.item()] += 1
    return class_counts


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the CIFAR-10 training script.

    This function sets up an argument parser with various configuration options
    for training a CIFAR-10 model with ExecuTorch, including data paths,
    training hyperparameters, and model save locations.

    Returns:
        argparse.Namespace: An object containing all the parsed command line
        arguments with their respective values (either user-provided or
        defaults).

    """
    parser = argparse.ArgumentParser(description="CIFAR-10 Data Preparation Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data loaders (default: 4)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading (default: 2)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download CIFAR-10 dataset (default: ./data)",
    )

    parser.add_argument(
        "--use-balanced-dataset",
        action="store_true",
        default=True,
        help="Use balanced dataset instead of full CIFAR-10 (default: True)",
    )

    parser.add_argument(
        "--train-data-batch-path",
        type=str,
        default="./data/cifar-10/cifar-10-batches-py/data_batch_1",
        help="Directory for cifar-10-batches-py",
    )

    parser.add_argument(
        "--train-output-path",
        type=str,
        default="./data/cifar-10/extracted_data/train_data.bin",
        help="Directory for saving the train_data.bin",
    )

    parser.add_argument(
        "--test-data-batch-path",
        type=str,
        default="./data/cifar-10/cifar-10-batches-py/test_batch_1",
        help="Directory for cifar-10-batches-py",
    )

    parser.add_argument(
        "--test-output-path",
        type=str,
        default="./data/cifar-10/extracted_data/train_data.bin",
        help="Directory for saving the train_data.bin",
    )

    parser.add_argument(
        "--train-images-per-class",
        type=int,
        default=100,
        help="Number of images per class for balanced dataset (default: 100 and max: 1000)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Utility function to demonstrate data loading and class distribution analysis.

    This function creates data loaders for CIFAR-10 dataset using the get_data_loaders
    function, then counts and prints the number of images per class in both the
    training and test datasets to verify balanced distribution.

    Returns:
        None
    """

    args = parse_args()

    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_balanced_dataset=args.use_balanced_dataset,
        images_per_class=args.train_images_per_class,
    )

    # Count images per class
    class_counts = count_images_per_class(train_loader)

    print("Class counts in train dataset:", class_counts)

    class_counts = count_images_per_class(test_loader)

    print("Class counts in test dataset:", class_counts)


if __name__ == "__main__":
    main()
