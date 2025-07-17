# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import pickle
import time
import typing
from collections import defaultdict

import numpy as np
import torch
import torchvision
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
    ExecuTorchModule,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


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


def save_json(
    history: typing.Dict[int, typing.Dict[str, float]], json_path: str
) -> str:
    """
    Save training/validation history to a JSON file.

    This function takes a dictionary containing training/validation metrics
    organized by epoch and saves it to a JSON file at the specified path.

    Args:
        history (Dict[int, Dict[str, float]]): Dictionary with epoch numbers
            as keys and dictionaries of metrics (loss, accuracy, etc.) as
            values.
        json_path (str): File path where the JSON file will be saved.

    Returns:
        str: The path where the JSON file was saved.
    """
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {json_path}")
    return json_path


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    momentum: float = 0.9,
    save_path: str = "./best_cifar10_model.pth",
) -> typing.Tuple[torch.nn.Module, typing.Dict[int, typing.Dict[str, float]]]:
    """
    The train_model function takes a model, a train_loader, and the number of
    epochs as input.It then trains the model on the training data for the
    specified number of epochs using the SGD optimizer and a cross-entropy loss
    function. The function returns the trained model.

    args:
            model (Required): The model to be trained.
            train_loader (tuple, Required): The training data loader.
            test_loader (tuple, Optional): The testing data loader.
            epochs (int, optional): The number of epochs to train the model.
            lr (float, optional): The learning rate for the SGD optimizer.
            momentum (float, optional): The momentum for the SGD optimizer.
            save_path (str, optional): Path to save the best model.
    """

    history = {}
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Initialize best testing loss to a high value for checkpointing
    # on the best model
    best_test_loss = float("inf")

    # Create directory for save_path if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_start_time = time.time()
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for data in train_loader:
            # Get the input data as a list of [inputs, labels]
            inputs, labels = data

            # Set the gradients to zero for the next backward pass
            optimizer.zero_grad()

            # Forward + Backward pass and optimization
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate correct predictions for epoch statistics
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            # Accumulate statistics for epoch summary
            epoch_loss += loss.detach().item()
            epoch_correct += correct
            epoch_total += total

        train_end_time = time.time()
        # Calculate the stats for average loss and accuracy for
        # the entire epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = 100 * epoch_correct / epoch_total
        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_epoch_loss:.4f}, "
            f"Train Accuracy: {avg_epoch_accuracy:.2f}%"
        )

        test_start_time = time.time()
        # Testing phase
        if test_loader is not None:
            model.eval()  # Set model to evaluation mode
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():  # No need to track gradients
                for data in test_loader:
                    images, labels = data
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.detach().item()

                    # Calculate Testing accuracy as well
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            # Calculate average Testing loss and accuracy
            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = 100 * test_correct / test_total
            test_end_time = time.time()
            print(
                f"\t Testing Loss: {avg_test_loss:.4f}, "
                f"Testing Accuracy: {test_accuracy:.2f}%"
            )

            # Save the model with the best Testing loss
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), save_path)
                print(
                    f"New best model saved with Testing loss: "
                    f"{avg_test_loss:.4f} and Testing accuracy: "
                    f"{test_accuracy:.2f}%"
                )

            history[epoch] = {
                "train_loss": avg_epoch_loss,
                "train_accuracy": avg_epoch_accuracy,
                "testing_loss": avg_test_loss,
                "testing_accuracy": test_accuracy,
                "training_time": train_end_time - train_start_time,
                "train_time_per_image": (train_end_time - train_start_time)
                / epoch_total,
                "testing_time": test_end_time - test_start_time,
                "test_time_per_image": (test_end_time - test_start_time) / test_total,
            }

    print("\nTraining Completed!\n")
    print("\n###########SUMMARY#############\n")
    print(f"Best Testing loss: {best_test_loss:.4f}")
    print(f"Model saved at: {save_path}\n")
    print("################################\n")

    return model, history


def fine_tune_executorch_model(
    model_path: str,
    save_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
) -> tuple[ExecuTorchModule, typing.Dict[str, typing.Any]]:
    """
    Fine-tune an ExecutorTorch model using a training and validation dataset.

    This function loads an ExecutorTorch model from a file, fine-tunes it using
    the provided training data loader, and evaluates it on the validation data
    loader. The function returns the fine-tuned model and a history dictionary
    containing training and validation metrics.

    Args:
        model_path (str): Path to the ExecutorTorch model file to be
        fine-tuned.
        save_path (str): Path where the fine-tuned model will be saved.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int, optional): Number of epochs for fine-tuning.
        learning_rate (float, optional): Learning rate for parameter
        updates (default: 0.001).
        momentum (float, optional): Momentum for parameter updates
        (default: 0.9).

    Returns:
        tuple: A tuple containing the fine-tuned ExecutorTorchModule
               and a dictionary with training and validation metrics.
    """
    with open(model_path, "rb") as f:
        model_bytes = f.read()
        et_mod = _load_for_executorch_from_buffer(model_bytes)

    grad_start = et_mod.run_method("__et_training_gradients_index_forward", [])[0]
    param_start = et_mod.run_method("__et_training_parameters_index_forward", [])[0]
    history = {}

    # Initialize momentum buffers for SGD with momentum
    momentum_buffers = {}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        train_start_time = time.time()

        for batch in tqdm(train_loader):
            inputs, labels = batch
            # Process each image-label pair individually
            for i in range(len(inputs)):
                input_image = inputs[
                    i : i + 1
                ]  # Use list slicing to extract single image
                label = labels[i : i + 1]
                # Forward pass
                out = et_mod.forward((input_image, label), clone_outputs=False)
                loss = out[0]
                predicted = out[1]
                epoch_loss += loss.item()

                # Calculate accuracy
                if predicted.item() == label.item():
                    train_correct += 1
                train_total += 1

                # Update parameters using SGD with momentum
                with torch.no_grad():
                    for param_idx, (grad, param) in enumerate(
                        zip(out[grad_start:param_start], out[param_start:])
                    ):
                        if momentum > 0:
                            # Initialize momentum buffer if not exists
                            if param_idx not in momentum_buffers:
                                momentum_buffers[param_idx] = torch.zeros_like(grad)

                            # Update momentum buffer: v = momentum * v + grad
                            momentum_buffers[param_idx].mul_(momentum).add_(grad)
                            # Update parameter: param = param - lr * v
                            param.sub_(learning_rate * momentum_buffers[param_idx])
                        else:
                            # Standard SGD without momentum
                            param.sub_(learning_rate * grad)

        train_end_time = time.time()
        train_accuracy = 100 * train_correct / train_total if train_total != 0 else 0

        avg_epoch_loss = epoch_loss / len(train_loader) / (train_loader.batch_size or 1)

        # Evaluate on validation set

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_samples = 100  # Limiting validation samples to 100
        val_start_time = time.time()

        for i, val_batch in tqdm(enumerate(val_loader)):
            if i == val_samples:
                print(f"Reached {val_samples} samples for validation")
                break

            inputs, labels = val_batch

            for i in range(len(inputs)):
                input_image = inputs[
                    i : i + 1
                ]  # Use list slicing to extract single image
                label = labels[i : i + 1]
                # Forward pass
                out = et_mod.forward((input_image, label), clone_outputs=False)
                loss = out[0]
                predicted = out[1]
                val_loss += loss.item()
                # Calculate accuracy
                if predicted.item() == label.item():
                    val_correct += 1
                val_total += 1

        val_end_time = time.time()
        val_accuracy = 100 * val_correct / val_total if val_total != 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss /= val_loader.batch_size or 1

        history[epoch] = {
            "train_loss": avg_epoch_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": avg_val_loss,
            "validation_accuracy": val_accuracy,
            "training_time": train_end_time - train_start_time,
            "train_time_per_image": (train_end_time - train_start_time) / train_total,
            "testing_time": val_end_time - val_start_time,
            "test_time_per_image": (val_end_time - val_start_time) / val_total,
        }

    return et_mod, history
