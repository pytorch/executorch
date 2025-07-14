# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import time
import typing

import torch
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
    ExecuTorchModule,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class CIFAR10Model(torch.nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(CIFAR10Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        """
        The forward function takes the input image and applies the
        convolutional layers and the fully connected layers to
        extract the features and classify the image respectively.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModelWithLoss(torch.nn.Module):
    """
    NOTE: A wrapper class that combines a model and the loss function
    into a single module. Used for capturing the entire computational
    graph, i.e. forward pass and the loss calculation, to be captured
    during export. Our objective is to enable on-device training, so
    the loss calculation should also be included in the exported graph.
    """

    def __init__(
        self, model: torch.nn.Module, criterion: torch.nn.CrossEntropyLoss
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass through the model
        output = self.model(x)
        # Calculate loss
        loss = self.criterion(output, target)
        # Return loss and predicted class
        return loss, output.detach().argmax(dim=1)


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
                "test_time_per_image": (test_end_time - test_start_time)/ test_total,
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
            "train_time_per_image": (train_end_time - train_start_time)/ train_total,
            "testing_time": val_end_time - val_start_time,
            "test_time_per_image": (val_end_time - val_start_time)/ val_total,
        }

    return et_mod, history
