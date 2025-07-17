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
from executorch.extension.training.examples.CIFAR.data_utils import get_data_loaders
from executorch.extension.training.examples.CIFAR.model import (
    CIFAR10Model,
    ModelWithLoss,
)
from torch.export import export
from torch.export.experimental import _export_forward_backward


def export_model_combined(
    net: torch.nn.Module,
    input_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    with_external_tensor_data: bool = False,
) -> ExecuTorchModule:
    """
    Export a PyTorch model to an ExecutorTorch module format, optionally with external tensor data.

    This function takes a PyTorch model and sample input/label tensors,
    wraps the model with a loss function, exports it using torch.export,
    applies forward-backward pass optimization, converts it to edge format,
    and finally to ExecutorTorch format. If with_external_tensor_data is True,
    the model will be exported with external constants and mutable weights.

    TODO: set dynamic shape for the batch size here.

    Args:
        net (torch.nn.Module): The PyTorch model to be exported
        input_tensor (torch.Tensor): A sample input tensor with the correct shape
        label_tensor (torch.Tensor): A sample label tensor with the correct shape
        with_external_tensor_data (bool, optional): Whether to export with external tensor data.
            Defaults to False.

    Returns:
        ExecuTorchModule: The exported model in ExecutorTorch format ready for deployment
    """
    criterion = torch.nn.CrossEntropyLoss()
    model_with_loss = ModelWithLoss(net, criterion)
    ep = export(model_with_loss, (input_tensor, label_tensor), strict=True)
    ep = _export_forward_backward(ep)
    ep = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    if with_external_tensor_data:
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
    else:
        ep = ep.to_executorch()

    return ep


def get_pte_only(net: torch.nn.Module) -> ExecuTorchModule:
    """
    Generate an ExecutorTorch module from a PyTorch model without external tensor data.

    This function retrieves a sample input and label tensor from the test data loader,
    and uses them to export the given PyTorch model to an ExecutorTorch module format
    without external constants or mutable weights.

    Args:
        net (torch.nn.Module): The PyTorch model to be exported.

    Returns:
        ExecuTorchModule: The exported model in ExecutorTorch format.
    """
    _, test_loader = get_data_loaders()
    # get a sample input and label tensor
    validation_sample_data = next(iter(test_loader))
    sample_input, sample_label = validation_sample_data
    return export_model_combined(
        net, sample_input, sample_label, with_external_tensor_data=False
    )


def get_pte_with_ptd(net: torch.nn.Module) -> ExecuTorchModule:
    """
    Generate an ExecutorTorch module from a PyTorch model with external tensor data.

    This function retrieves a sample input and label tensor from the test data loader,
    and uses them to export the given PyTorch model to an ExecutorTorch module format
    with external constants and mutable weights.

    Args:
        net (torch.nn.Module): The PyTorch model to be exported.

    Returns:
        ExecuTorchModule: The exported model in ExecutorTorch format with external tensor data.
    """
    _, test_loader = get_data_loaders()
    # get a sample input and label tensor
    validation_sample_data = next(iter(test_loader))
    sample_input, sample_label = validation_sample_data
    return export_model_combined(
        net, sample_input, sample_label, with_external_tensor_data=True
    )


def export_model(
    net: torch.nn.Module,
    with_ptd: bool = False,
) -> ExecuTorchModule:
    """
    Export a PyTorch model to ExecutorTorch format, optionally with external tensor data.

    This function is a high-level wrapper that handles getting sample data and
    calling the appropriate export function based on the with_ptd flag.

    Args:
        net (torch.nn.Module): The PyTorch model to be exported
        with_ptd (bool, optional): Whether to export with external tensor data.
            Defaults to False.

    Returns:
        ExecuTorchModule: The exported model in ExecutorTorch format
    """
    _, test_loader = get_data_loaders()
    validation_sample_data = next(iter(test_loader))
    sample_input, sample_label = validation_sample_data

    return export_model_combined(
        net, sample_input, sample_label, with_external_tensor_data=with_ptd
    )


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
    parser = argparse.ArgumentParser(description="CIFAR-10 Data Preparation Example")
    parser.add_argument(
        "--train-model-path",
        type=str,
        default="./cifar10_model.pth",
        help="Path to the saved PyTorch model",
    )
    parser.add_argument(
        "--pte-only-model-path",
        type=str,
        default="./cifar10_pte_only_model.pte",
        help="Path to the saved PTE only",
    )
    parser.add_argument(
        "--with-ptd",
        action="store_true",
        help="Whether to export the model with ptd",
    )
    parser.add_argument(
        "--pte-model-path",
        type=str,
        default="./cifar10_model.pte",
        help="Path to the saved PTE",
    )
    parser.add_argument(
        "--ptd-model-path",
        type=str,
        default="./cifar10_model.ptd",
        help="Path to the saved PTD",
    )

    return parser.parse_args()


def update_tensor_data_and_save(exported_program, ptd_model_path, pte_model_path):
    exported_program._tensor_data["generic_cifar"] = exported_program._tensor_data.pop(
        "_default_external_constant"
    )
    exported_program.write_tensor_data_to_file(ptd_model_path)
    save_model(exported_program, pte_model_path)


def main():
    args = parse_args()
    net = CIFAR10Model()
    state_dict = torch.load(args.train_model_path, weights_only=True)
    net.load_state_dict(state_dict)
    if args.with_ptd:
        exported_program = get_pte_with_ptd(net)
        update_tensor_data_and_save(
            exported_program, args.ptd_model_path, args.pte_model_path
        )
    else:
        exported_program = get_pte_only(net)
        save_model(exported_program, args.pte_only_model_path)


if __name__ == "__main__":
    main()
