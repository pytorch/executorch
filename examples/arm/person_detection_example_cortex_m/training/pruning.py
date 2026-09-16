# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Compact a channel-masked µYOLO checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(EXAMPLE_DIR))
from model import (  # type: ignore[import-not-found]
    ChannelMask,
    DEFAULT_BACKBONE_CHANNELS,
    DEFAULT_HIDDEN_FEATURES,
    MicroYolo,
    model_from_checkpoint,
)
from torch import nn
from utils.artifacts import (  # type: ignore[import-not-found]
    PRUNED_PATH,
    report_input,
    report_output,
    save_checkpoint,
    TRAINED_PATH,
)

COMPACT_BACKBONE_CHANNELS = (32, 64, 64, 64, 64, 32, 32, 32)
COMPACT_HIDDEN_FEATURES = 512
FLATTENED_CHANNEL_SIZE = 16


def pruning_indices(mask: ChannelMask, count: int) -> tuple[torch.Tensor, bool]:
    """Returns the pruned indices for a given ChannelMask"""
    active = torch.sort(mask.active_indices.cpu()).values
    if active.numel() == count:
        return active, False

    all_indices = torch.arange(mask.mask.numel())
    inactive = all_indices[~torch.isin(all_indices, active)]
    return torch.cat((active, inactive))[:count], True


def get_pruned_indices(source: MicroYolo) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Returns pruned indices for a MicroYolo model"""
    feature_results = [
        pruning_indices(mask, count)
        for mask, count in zip(source.feature_masks, COMPACT_BACKBONE_CHANNELS)
    ]
    hidden, hidden_changed = pruning_indices(
        source.hidden_mask, COMPACT_HIDDEN_FEATURES
    )
    features = [indices for indices, _ in feature_results]
    if hidden_changed or any(changed for _, changed in feature_results):
        print(
            "Warning: input checkpoint is not masked with the expected final "
            "channel counts; normalizing its masks before compaction."
        )

    for mask, indices in zip(source.feature_masks, features):
        mask.mask.zero_()
        mask.mask[indices] = 1
    source.hidden_mask.mask.zero_()
    source.hidden_mask.mask[hidden] = 1
    return features, hidden


def copy_conv(
    destination: nn.Conv2d,
    source: nn.Conv2d,
    output_indices: torch.Tensor,
    input_indices: torch.Tensor | None = None,
) -> None:
    with torch.no_grad():
        weights = source.weight[output_indices]
        if input_indices is not None:
            weights = weights[:, input_indices]
        destination.weight.copy_(weights)
        if destination.bias is not None and source.bias is not None:
            destination.bias.copy_(source.bias[output_indices])


def copy_batch_norm(
    destination: nn.Module, source: nn.Module, indices: torch.Tensor
) -> None:
    if not isinstance(destination, (nn.BatchNorm1d, nn.BatchNorm2d)) or not isinstance(
        source, (nn.BatchNorm1d, nn.BatchNorm2d)
    ):
        raise ValueError("Expected matching batch normalization layers.")
    assert destination.weight is not None and source.weight is not None
    assert destination.bias is not None and source.bias is not None
    assert destination.running_mean is not None and source.running_mean is not None
    assert destination.running_var is not None and source.running_var is not None
    assert destination.num_batches_tracked is not None
    assert source.num_batches_tracked is not None
    with torch.no_grad():
        destination.weight.copy_(source.weight[indices])
        destination.bias.copy_(source.bias[indices])
        destination.running_mean.copy_(source.running_mean[indices])
        destination.running_var.copy_(source.running_var[indices])
        destination.num_batches_tracked.copy_(source.num_batches_tracked)


def compact_model(
    source: MicroYolo,
    grid_size: int,
    num_boxes: int,
) -> MicroYolo:
    """Takes a MicroYolo model which has been pruned using channel masks and produces a
    corresponding model with channel masks removed and half the number of channels.

    The compacting is done by creating the new smaller model and then copying over only
    weight channels which are not masked, given by the indices from `features` and `hidden`.
    """
    features, hidden = get_pruned_indices(source)

    if (
        source.backbone_channels != DEFAULT_BACKBONE_CHANNELS
        or source.hidden_features != DEFAULT_HIDDEN_FEATURES
    ):
        raise ValueError(
            "pruning.py currently rewrites the standard µYOLO architecture only."
        )

    destination = MicroYolo(
        grid_size=grid_size,
        num_boxes=num_boxes,
        backbone_channels=COMPACT_BACKBONE_CHANNELS,
        hidden_features=COMPACT_HIDDEN_FEATURES,
        pruning_masks=False,
    )

    source_stem, destination_stem = source.backbone[0], destination.backbone[0]
    copy_conv(destination_stem, source_stem, features[0])
    copy_batch_norm(destination.backbone[1], source.backbone[1], features[0])

    layer_indices = (4, 5, 6, 8, 9, 10, 11)
    previous = features[0]
    for feature, index in zip(features[1:], layer_indices):
        source_layer = source.backbone[index]
        destination_layer = destination.backbone[index]
        copy_conv(destination_layer[0], source_layer[0], previous)
        copy_conv(destination_layer[1], source_layer[1], feature, previous)
        copy_batch_norm(destination_layer[2], source_layer[2], feature)
        previous = feature

    flattened = (
        features[-1][:, None] * FLATTENED_CHANNEL_SIZE
        + torch.arange(FLATTENED_CHANNEL_SIZE)
    ).reshape(-1)
    source_first, destination_first = source.head[0], destination.head[0]
    with torch.no_grad():
        destination_first.weight.copy_(source_first.weight[hidden][:, flattened])
        destination_first.bias.copy_(source_first.bias[hidden])
        destination.head[3].weight.copy_(source.head[3].weight[:, hidden])
        destination.head[3].bias.copy_(source.head[3].bias)
    copy_batch_norm(destination.head[1], source.head[1], hidden)
    return destination


def verify_compaction(source: MicroYolo, compact: MicroYolo) -> None:
    source.eval()
    compact.eval()
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(2, 3, 128, 128, generator=generator)
    with torch.no_grad():
        expected = source(inputs)
        actual = compact(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=PRUNED_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or TRAINED_PATH
    report_input("checkpoint", input_path)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        raise ValueError(f"{input_path} is not a µYOLO detector checkpoint.")

    checkpoint["architecture"] = {
        **checkpoint.get("architecture", {}),
        "pruning_masks": True,
    }
    source = model_from_checkpoint(checkpoint)
    incompatible = source.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.endswith(".mask") for key in incompatible.missing_keys
    ):
        raise ValueError(f"{input_path} is not a compatible µYOLO detector checkpoint.")
    if source.backbone_channels != DEFAULT_BACKBONE_CHANNELS:
        raise ValueError("Input checkpoint is already structurally pruned.")

    compact = compact_model(
        source,
        checkpoint["grid_size"],
        checkpoint["num_boxes"],
    )
    if compact.head[3].out_features != source.head[3].out_features:
        raise ValueError("Compact model changed the detector output shape.")
    verify_compaction(source, compact)
    print("Verified masked and compact model outputs match.")

    output_checkpoint = {
        "model": compact.state_dict(),
        "grid_size": checkpoint["grid_size"],
        "num_classes": checkpoint["num_classes"],
        "num_boxes": checkpoint["num_boxes"],
        "dataset": checkpoint.get("dataset"),
        "architecture": {
            "backbone_channels": list(COMPACT_BACKBONE_CHANNELS),
            "hidden_features": COMPACT_HIDDEN_FEATURES,
            "pruning_masks": False,
        },
        "pruning": {
            "source": str(input_path),
            "training_method": "iterative_gradual_l1_channel_masking",
            "rewrite_method": "saved_channel_mask_structural",
        },
    }
    save_checkpoint(output_checkpoint, args.output)
    original_parameters = sum(parameter.numel() for parameter in source.parameters())
    compact_parameters = sum(parameter.numel() for parameter in compact.parameters())
    original_weight_bytes = sum(
        module.weight.numel()
        for module in source.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    )
    compact_weight_bytes = sum(
        module.weight.numel()
        for module in compact.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    )
    report_output(args.output)
    print(f"Parameters: {original_parameters} -> {compact_parameters}")
    print(
        f"Int8 weight estimate: {original_weight_bytes} -> {compact_weight_bytes} bytes"
    )


if __name__ == "__main__":
    main()
