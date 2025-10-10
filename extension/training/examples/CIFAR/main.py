# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

from executorch.extension.training.examples.CIFAR.data_utils import get_data_loaders
from executorch.extension.training.examples.CIFAR.export import (
    get_pte_only,
    get_pte_with_ptd,
    save_model,
    update_tensor_data_and_save,
)
from executorch.extension.training.examples.CIFAR.model import CIFAR10Model
from executorch.extension.training.examples.CIFAR.train_utils import (
    save_json,
    train_both_models,
    train_model,
)


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
        "--ptd-model-dir", type=str, default=".", help="PTD model path (default: .)"
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

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for fine-tuning (default: 0.9)",
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

    ep = get_pte_only(model)

    save_model(ep, args.pte_model_path)

    pytorch_model, et_mod, pytorch_history, et_history = train_both_models(
        pytorch_model=model,
        et_model_path=args.pte_model_path,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.fine_tune_epochs,
        lr=args.learning_rate,
        momentum=args.momentum,
        pytorch_save_path=args.model_path,
    )

    save_json(et_history, args.save_et_json)
    save_json(pytorch_history, args.save_pt_json)

    # Split the model into the pte and ptd files
    exported_program = get_pte_with_ptd(model)

    update_tensor_data_and_save(
        exported_program, args.ptd_model_dir, args.split_pte_model_path
    )
    print("\n\nProcess complete!!!\n\n")


if __name__ == "__main__":
    main()
