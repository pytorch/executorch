# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from executorch.extension.training.examples.CIFAR.model import (
    CIFAR10Model,
    fine_tune_executorch_model,
    ModelWithLoss,
    train_model,
)
from executorch.extension.training.examples.CIFAR.utils import (
    get_data_loaders,
    save_json,
)
from torch.export import export
from torch.export.experimental import _export_forward_backward


def export_model(
    net: torch.nn.Module,input_tensor: torch.Tensor,label_tensor: torch.Tensor
) -> ExecuTorchModule:
    """
    Export a PyTorch model to an ExecutorTorch module format.

    This function takes a PyTorch model and sample input/label
    tensors, wraps the model with a loss function, exports it
    using torch.export, applies forward-backward pass
    optimization, converts it to edge format, and finally to
    ExecutorTorch format.

    Args:
        net (torch.nn.Module): The PyTorch model to be exported
        input_tensor (torch.Tensor): A sample input tensor with
        the correct shape
        label_tensor (torch.Tensor): A sample label tensor with
        the correct shape

    Returns:
        ExecuTorchModule: The exported model in ExecutorTorch
        format ready for deployment
    """
    criterion = torch.nn.CrossEntropyLoss()
    model_with_loss = ModelWithLoss(net, criterion)
    ep = export(model_with_loss, (input_tensor, label_tensor), strict=True)
    ep = _export_forward_backward(ep)
    ep = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    ep = ep.to_executorch()
    return ep


def export_model_with_ptd(
    net: torch.nn.Module,input_tensor: torch.Tensor,label_tensor: torch.Tensor
) -> ExecuTorchModule:
    """
    Export a PyTorch model to an ExecutorTorch module format with external
    tensor data.

    This function takes a PyTorch model and sample input/label tensors,
    wraps the model with a loss function, exports it using torch.export,
    applies forward-backward pass optimization, converts it to edge format,
    and finally to ExecutorTorch format with external constants and mutable
    weights.

    Args:
        net (torch.nn.Module): The PyTorch model to be exported
        input_tensor (torch.Tensor): A sample input tensor with the correct
        shape
        label_tensor (torch.Tensor): A sample label tensor with the correct
        shape

    Returns:
        ExecuTorchModule: The exported model in ExecutorTorch format ready for
        deployment
    """
    criterion = torch.nn.CrossEntropyLoss()
    model_with_loss = ModelWithLoss(net, criterion)
    ep = export(model_with_loss, (input_tensor, label_tensor), strict=True)
    ep = _export_forward_backward(ep)
    ep = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    ep = ep.to_executorch(
        config=ExecutorchBackendConfig(
            external_constants=True,  # This is the flag that
            # enables the external constants to be stored in a
            # separate file external to the PTE file.
            external_mutable_weights=True,  # This is the flag
            # that enables all trainable weights will be stored
            # in a separate file external to the PTE file.
        )
    )
    return ep


def save_model(ep: ExecuTorchModule, model_path: str) -> None:
    """
    Save an ExecutorTorch model to a specified file path.

    This function writes the buffer of an ExecutorTorchModule to a
    file in binary format.

    Args:
        ep (ExecuTorchModule): The ExecutorTorch module to be saved.
        model_path (str): The file path where the model will be saved.
    """
    with open(model_path, "wb") as file:
        file.write(ep.buffer)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the CIFAR-10 training script.

    This function sets up an argument parser with various configuration options
    for training a CIFAR-10 model with ExecutorTorch, including data paths,
    training hyperparameters, and model save locations.

    Returns:
        argparse.Namespace: An object containing all the parsed command line
        arguments with their respective values (either user-provided or
        defaults).
    """
    parser = argparse.ArgumentParser(description="CIFAR-10 Training Example")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download CIFAR-10 dataset (default: ./data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data loaders (default: 4)",
    )
    parser.add_argument(
        "--use-balanced-dataset",
        action="store_true",
        default=True,
        help="Use balanced dataset instead of full CIFAR-10 (default: True)",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=100,
        help="Number of images per class for balanced dataset (default: 100)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="cifar10_model.pth",
        help="PyTorch model path (default: cifar10_model.pth)",
    )

    parser.add_argument(
        "--pte-model-path",
        type=str,
        default="cifar10_model.pte",
        help="PTE model path (default: cifar10_model.pte)",
    )

    parser.add_argument(
        "--split-pte-model-path",
        type=str,
        default="split_cifar10_model.pte",
        help="Split PTE model path (default: split_cifar10_model.pte)",
    )

    parser.add_argument(
        "--ptd-model-dir",type=str, default=".",help="PTD model path (default: .)"
    )

    parser.add_argument(
        "--save-pt-json",
        type=str,
        default="cifar10_pt_model_finetuned_history.json",
        help="Save the et json file",
    )

    parser.add_argument(
        "--save-et-json",
        type=str,
        default="cifar10_et_pte_only_model_finetuned_history.json",
        help="Save the et json file",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for training (default: 1)",
    )

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs for fine-tuning (default: 150)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for fine-tuning (default: 0.001)",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_balanced_dataset=args.use_balanced_dataset,
        images_per_class=args.images_per_class,
    )

    # initialize the main model
    model = CIFAR10Model()

    model, train_hist = train_model(
        model,
        train_loader,
        test_loader,
        epochs=1,
        lr=0.001,
        momentum=0.9,
        save_path=args.model_path,
    )

    save_json(train_hist, args.save_pt_json)

    # Export the model for et runtime
    validation_sample_data = next(iter(test_loader))
    img, lbl = validation_sample_data
    sample_input = img[0:1, :]
    sample_label = lbl[0:1]

    ep = export_model(model, sample_input, sample_label)

    save_model(ep, args.pte_model_path)

    et_model, et_hist = fine_tune_executorch_model(
        args.pte_model_path,
        args.pte_model_path,
        train_loader,
        test_loader,
        epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
    )

    save_json(et_hist, args.save_et_json)

    # Split the model into the pte and ptd files
    exported_program = export_model_with_ptd(model, sample_input, sample_label)

    exported_program._tensor_data["generic_cifar"] = exported_program._tensor_data.pop(
        "_default_external_constant"
    )
    exported_program.write_tensor_data_to_file(args.ptd_model_dir)
    save_model(exported_program, args.split_pte_model_path)

    # Finetune the PyTorch model
    model, train_hist = train_model(
        model,
        train_loader,
        test_loader,
        epochs=args.fine_tune_epochs,
        lr=args.learning_rate,
        momentum=0.9,
        save_path=args.model_path,
    )

    save_json(train_hist, args.save_pt_json)


if __name__ == "__main__":
    main()
