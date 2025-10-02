# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import time
import typing

import torch
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
    ExecuTorchModule,
)
from executorch.extension.training import (
    _load_for_executorch_for_training_from_buffer,
    get_sgd_optimizer,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


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
) -> tuple[ExecuTorchModule, typing.Dict[int, typing.Dict[str, float]]]:
    """
    Fine-tune an ExecuTorch model using a training and validation dataset.

    This function loads an ExecuTorch model from a file, fine-tunes it using
    the provided training data loader, and evaluates it on the validation data
    loader. The function returns the fine-tuned model and a history dictionary
    containing training and validation metrics.

    Args:
        model_path (str): Path to the ExecuTorch model file to be
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
        tuple: A tuple containing the fine-tuned ExecuTorchModule
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
            # Forward pass
            out = et_mod.forward((inputs, labels), clone_outputs=False)
            loss = out[0]
            predicted = out[1]
            epoch_loss += loss.item()

            # Calculate accuracy
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

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

        avg_epoch_loss = epoch_loss / len(train_loader)

        # Evaluate on validation set

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_samples = 100  # Limiting validation samples to 100
        val_start_time = time.time()
        val_batches_processed = 0

        for i, val_batch in tqdm(enumerate(val_loader)):
            if i >= val_samples:
                print(f"Reached {val_samples} batches for validation")
                break

            inputs, labels = val_batch
            val_batches_processed += 1

            # Forward pass with full batch
            out = et_mod.forward((inputs, labels), clone_outputs=False)
            loss = out[0]
            predicted = out[1]
            val_loss += loss.item()

            # Calculate accuracy
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

        val_end_time = time.time()
        val_accuracy = 100 * val_correct / val_total if val_total != 0 else 0
        avg_val_loss = (
            val_loss / val_batches_processed if val_batches_processed > 0 else 0
        )

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


def train_both_models(
    pytorch_model: torch.nn.Module,
    et_model_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    momentum: float = 0.9,
    pytorch_save_path: str = "./best_cifar10_model.pth",
    et_save_path: str = "./best_cifar10_et_model.pte",
) -> typing.Tuple[
    torch.nn.Module,
    typing.Any,
    typing.Dict[int, typing.Dict[str, float]],
    typing.Dict[int, typing.Dict[str, float]],
]:
    """
    Train both a PyTorch model and an ExecuTorch model simultaneously using the same data.

    This function trains both models in parallel, using the same data batches for both,
    which makes debugging and comparison easier. It tracks metrics for both models
    and provides a comparison of their performance.

    Args:
        pytorch_model (torch.nn.Module): The PyTorch model to be trained
        et_model_path (str): Path to the ExecuTorch model file
        train_loader (DataLoader): DataLoader for the training dataset
        test_loader (DataLoader): DataLoader for the testing/validation dataset
        epochs (int, optional): Number of epochs for training. Defaults to 10.
        lr (float, optional): Learning rate for parameter updates. Defaults to 0.001.
        momentum (float, optional): Momentum for parameter updates. Defaults to 0.9.
        pytorch_save_path (str, optional): Path to save the best PyTorch model. Defaults to "./best_cifar10_model.pth".

    Returns:
        tuple: A tuple containing:
            - The trained PyTorch model
            - The trained ExecuTorch model
            - Dictionary with PyTorch training and validation metrics
            - Dictionary with ExecuTorch training and validation metrics
    """
    # Load the ExecuTorch model
    with open(et_model_path, "rb") as f:
        model_bytes = f.read()
        et_mod = _load_for_executorch_for_training_from_buffer(model_bytes)

    # Initialize histories for both models
    pytorch_history = {}
    et_history = {}

    # Initialize criterion and optimizer for PyTorch model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=lr, momentum=momentum)

    # TODO: Fix "RuntimeError: Must call forward_backward before named_params.
    #            This will be fixed in a later version"
    # Evaluating the model for 1 epoch to initialize the parameters and get unblocked for now
    # get one batch of data for initialization
    images, labels = next(iter(train_loader))
    # Forward pass
    et_out = et_mod.forward_backward(method_name="forward", inputs=(images, labels))

    et_model_optimizer = get_sgd_optimizer(
        et_mod.named_parameters(),
        lr,
        momentum,
    )

    # Initialize best testing loss for checkpointing
    best_pytorch_test_loss = float("inf")
    best_et_test_loss = float("inf")

    # Create directories for save paths if they don't exist
    for path in [pytorch_save_path]:
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        pytorch_model.train()

        # Initialize metrics for this epoch
        pytorch_epoch_loss = 0.0
        pytorch_correct = 0
        pytorch_total = 0

        et_epoch_loss = 0.0
        et_correct = 0
        et_total = 0

        # Training loop
        pytorch_train_time = 0.0
        et_train_time = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels = batch
            batch_size = labels.size(0)

            # ---- PyTorch model training ----
            pytorch_start_time = time.time()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pytorch_outputs = pytorch_model(inputs)
            pytorch_loss = criterion(pytorch_outputs, labels)

            # Backward pass and optimization
            pytorch_loss.backward()
            optimizer.step()

            pytorch_end_time = time.time()
            pytorch_train_time += pytorch_end_time - pytorch_start_time

            # Calculate accuracy
            _, pytorch_predicted = torch.max(pytorch_outputs.data, 1)
            pytorch_correct += (pytorch_predicted == labels).sum().item()
            pytorch_total += batch_size

            # Accumulate loss
            pytorch_epoch_loss += pytorch_loss.detach().item()

            # ---- ExecuTorch model training ----
            et_start_time = time.time()

            # Forward pass
            et_out = et_mod.forward_backward(
                method_name="forward", inputs=(inputs, labels)
            )
            et_loss = et_out[0]
            et_predicted = et_out[1]

            # Backward pass and optimize using the ExecutorchProgramManager's step method
            et_model_optimizer.step(et_mod.named_gradients())

            et_end_time = time.time()
            et_train_time += et_end_time - et_start_time

            # Calculate accuracy
            et_correct += (et_predicted == labels).sum().item()
            et_total += batch_size

            # Accumulate loss
            et_epoch_loss += et_loss.item()

        # Calculate training metrics
        avg_pytorch_train_loss = pytorch_epoch_loss / len(train_loader)
        pytorch_train_accuracy = 100 * pytorch_correct / pytorch_total

        avg_et_train_loss = et_epoch_loss / len(train_loader)
        et_train_accuracy = 100 * et_correct / et_total

        print(
            f"PyTorch - Train Loss: {avg_pytorch_train_loss:.4f}, Train Accuracy: {pytorch_train_accuracy:.2f}%"
        )
        print(
            f"ExecuTorch - Train Loss: {avg_et_train_loss:.4f}, Train Accuracy: {et_train_accuracy:.2f}%"
        )

        # Testing/Validation phase
        pytorch_model.eval()

        pytorch_test_loss = 0.0
        pytorch_test_correct = 0
        pytorch_test_total = 0
        pytorch_test_time = 0.0

        et_test_loss = 0.0
        et_test_correct = 0
        et_test_total = 0
        et_test_time = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                inputs, labels = batch
                batch_size = labels.size(0)

                # ---- PyTorch model testing ----
                pytorch_test_start = time.time()

                pytorch_outputs = pytorch_model(inputs)
                pytorch_loss = criterion(pytorch_outputs, labels)

                pytorch_test_end = time.time()
                pytorch_test_time += pytorch_test_end - pytorch_test_start

                pytorch_test_loss += pytorch_loss.item()

                # Calculate accuracy
                _, pytorch_predicted = torch.max(pytorch_outputs.data, 1)
                pytorch_test_correct += (pytorch_predicted == labels).sum().item()
                pytorch_test_total += batch_size

                # ---- ExecuTorch model testing ----
                et_test_start = time.time()

                et_out = et_mod.forward_backward(
                    method_name="forward", inputs=(inputs, labels)
                )
                et_loss = et_out[0]
                et_predicted = et_out[1]

                et_test_end = time.time()
                et_test_time += et_test_end - et_test_start

                et_test_loss += et_loss.item()
                et_test_correct += (et_predicted == labels).sum().item()
                et_test_total += batch_size

        # Calculate testing metrics
        avg_pytorch_test_loss = pytorch_test_loss / len(test_loader)
        pytorch_test_accuracy = 100 * pytorch_test_correct / pytorch_test_total

        avg_et_test_loss = et_test_loss / len(test_loader)
        et_test_accuracy = 100 * et_test_correct / et_test_total

        print(
            f"PyTorch - Test Loss: {avg_pytorch_test_loss:.4f}, Test Accuracy: {pytorch_test_accuracy:.2f}%"
        )
        print(
            f"ExecuTorch - Test Loss: {avg_et_test_loss:.4f}, Test Accuracy: {et_test_accuracy:.2f}%"
        )

        # Compare losses
        loss_diff = abs(avg_pytorch_test_loss - avg_et_test_loss)
        print(f"Loss Difference: {loss_diff:.6f}")

        # Save the best PyTorch model
        if avg_pytorch_test_loss < best_pytorch_test_loss:
            best_pytorch_test_loss = avg_pytorch_test_loss
            torch.save(pytorch_model.state_dict(), pytorch_save_path)
            print(
                f"New best PyTorch model saved with test loss: {avg_pytorch_test_loss:.4f}"
            )

        # Save the best ExecuTorch model
        if avg_et_test_loss < best_et_test_loss:
            best_et_test_loss = avg_et_test_loss
            # Save the ExecuTorch model
            save_dir = os.path.dirname(et_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"New best ExecuTorch model with test loss: {avg_et_test_loss:.4f}")

        # Store history for both models
        pytorch_history[epoch] = {
            "train_loss": avg_pytorch_train_loss,
            "train_accuracy": pytorch_train_accuracy,
            "test_loss": avg_pytorch_test_loss,
            "test_accuracy": pytorch_test_accuracy,
        }

        et_history[epoch] = {
            "train_loss": avg_et_train_loss,
            "train_accuracy": et_train_accuracy,
            "test_loss": avg_et_test_loss,
            "test_accuracy": et_test_accuracy,
        }

        # Add timing information
        pytorch_history[epoch].update(
            {
                "training_time": pytorch_train_time,
                "train_time_per_image": pytorch_train_time / pytorch_total,
                "testing_time": pytorch_test_time,
                "test_time_per_image": pytorch_test_time / pytorch_test_total,
            }
        )

        et_history[epoch].update(
            {
                "training_time": et_train_time,
                "train_time_per_image": et_train_time / et_total,
                "testing_time": et_test_time,
                "test_time_per_image": et_test_time / et_test_total,
            }
        )

        # Print timing comparison
        print(
            f"PyTorch training time: {pytorch_train_time:.4f}s, testing time: {pytorch_test_time:.4f}s"
        )
        print(
            f"ExecuTorch training time: {et_train_time:.4f}s, testing time: {et_test_time:.4f}s"
        )
        print(f"Training time ratio (ET/PT): {et_train_time/pytorch_train_time:.4f}")
        print(f"Testing time ratio (ET/PT): {et_test_time/pytorch_test_time:.4f}")

    print("\nTraining Completed!\n")
    print("\n###########SUMMARY#############\n")
    print(f"Best PyTorch test loss: {best_pytorch_test_loss:.4f}")
    print(f"Best ExecuTorch test loss: {best_et_test_loss:.4f}")
    print(
        f"Final loss difference: {abs(best_pytorch_test_loss - best_et_test_loss):.6f}"
    )
    print(f"PyTorch model saved at: {pytorch_save_path}")
    print(f"ExecuTorch model path: {et_save_path}")
    print("################################\n")

    return pytorch_model, et_mod, pytorch_history, et_history
