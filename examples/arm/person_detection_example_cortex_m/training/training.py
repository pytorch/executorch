# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Train µYOLO for Open Images V7 Person detection with pruning."""

import argparse
import copy
import sys
from itertools import permutations
from pathlib import Path

import torch
import torch.nn.functional as functional

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(EXAMPLE_DIR))

from model import (  # type: ignore[import-not-found]
    ChannelMask,
    DEFAULT_GRID_SIZE,
    DEFAULT_NUM_BOXES,
    MicroYolo,
)
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import (  # type: ignore[import-not-found, import-untyped]
    MeanAveragePrecision,
)
from torchvision.ops import (  # type: ignore[import-not-found, import-untyped]
    batched_nms,
    generalized_box_iou_loss,
)

from utils.artifacts import (  # type: ignore[import-not-found]
    PRETRAINED_PATH,
    report_input,
    report_output,
    save_checkpoint,
    TRAINED_FULLY_PRUNED_PATH,
    TRAINED_PATH,
    TRAINED_UNPRUNED_PATH,
    TRAINING_RESUME_PATH,
)
from utils.dataset import (  # type: ignore[import-not-found]
    collate,
    DATASET_DIR,
    exported_image_count,
    IMAGE_SIZE,
    MicroYoloDataset,
    setup_dataset,
    Target,
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
)

PRUNING_TARGETS = {300: 0.1, 320: 0.2, 340: 0.3, 360: 0.4, 380: 0.5}
PRUNING_RECOVERY_LR = 0.0002
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5


def encode_targets(
    targets: list[Target], grid_size: int, num_boxes: int, device: torch.device
) -> torch.Tensor:
    """Encode dataset boxes into features used in MicroYolo training.

    Every box is retained so matching can select the best num_boxes targets in
    crowded cells. Boxes use (confidence, center_x, center_y, size_x, size_y).

    Args:
        targets (list[Target]): Per-image boxes in pixel XYXY format.
        grid_size (int): Number of rows and columns in the detection grid.
        num_boxes (int): Number of box predictions produced per grid cell.
        device (torch.device): Device on which to create the encoded targets.

    Returns:
        torch.Tensor: Encoded targets grouped by grid cell.
    """
    batch_size = len(targets)
    cells: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    maximum_targets = num_boxes
    for batch_index, target in enumerate(targets):
        target_boxes = target["boxes"].to(device)
        for box in target_boxes:
            x1, y1, x2, y2 = box / IMAGE_SIZE
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            column = min(int(center_x * grid_size), grid_size - 1)
            row = min(int(center_y * grid_size), grid_size - 1)
            cell = cells.setdefault((batch_index, row, column), [])
            cell.append(
                torch.tensor(
                    [
                        1,
                        center_x * grid_size - column,
                        center_y * grid_size - row,
                        x2 - x1,
                        y2 - y1,
                    ],
                    device=device,
                )
            )
            maximum_targets = max(maximum_targets, len(cell))

    boxes = torch.zeros(
        batch_size, grid_size, grid_size, maximum_targets, 5, device=device
    )
    for (batch_index, row, column), cell in cells.items():
        boxes[batch_index, row, column, : len(cell)] = torch.stack(cell)
    return boxes


def yolo_loss(
    predictions: torch.Tensor,
    targets: list[Target],
    grid_size: int,
    num_boxes: int,
) -> torch.Tensor:
    """Compute the µYOLO confidence and box regression loss."""
    batch_size = predictions.shape[0]
    box_logits = predictions.reshape(batch_size, grid_size, grid_size, num_boxes, 5)
    target_boxes = encode_targets(targets, grid_size, num_boxes, predictions.device)
    target_boxes = match_targets_to_predictions(box_logits, target_boxes)
    object_mask = target_boxes[..., 0].bool()
    predicted_coordinates = torch.sigmoid(box_logits[..., 1:])
    predicted_boxes = center_size_to_corners(predicted_coordinates, grid_size)
    expected_boxes = center_size_to_corners(target_boxes[..., 1:], grid_size)
    matched_iou = pairwise_box_iou(predicted_boxes, expected_boxes).diagonal(
        dim1=-2, dim2=-1
    )
    confidence_targets = torch.where(
        object_mask, matched_iou.detach(), torch.zeros_like(matched_iou)
    )
    confidence_loss = functional.binary_cross_entropy_with_logits(
        box_logits[..., 0], confidence_targets, reduction="none"
    )
    confidence_weight = torch.where(object_mask, 5.0, 0.5)
    confidence_loss = (confidence_loss * confidence_weight).mean()
    coordinate_loss = functional.smooth_l1_loss(
        predicted_coordinates[object_mask],
        target_boxes[..., 1:][object_mask],
        reduction="sum",
    ) / object_mask.sum().clamp_min(1)
    giou_loss = generalized_box_iou_loss(
        predicted_boxes[object_mask], expected_boxes[object_mask], reduction="sum"
    ) / object_mask.sum().clamp_min(1)
    return 5 * coordinate_loss + giou_loss + confidence_loss


def match_targets_to_predictions(
    box_logits: torch.Tensor, target_boxes: torch.Tensor
) -> torch.Tensor:
    """Match each cell's targets using a one-to-one coordinate and IoU cost."""
    predicted_coordinates = torch.sigmoid(box_logits[..., 1:]).detach()
    target_coordinates = target_boxes[..., 1:]
    pairwise_predictions, pairwise_targets = torch.broadcast_tensors(
        predicted_coordinates.unsqueeze(-2), target_coordinates.unsqueeze(-3)
    )
    coordinate_cost = functional.smooth_l1_loss(
        pairwise_predictions,
        pairwise_targets,
        reduction="none",
    ).sum(dim=-1)

    grid_size = target_boxes.shape[1]
    predicted_boxes = center_size_to_corners(predicted_coordinates, grid_size)
    target_corners = center_size_to_corners(target_coordinates, grid_size)
    iou = pairwise_box_iou(predicted_boxes, target_corners)
    matching_cost = coordinate_cost + 1 - iou

    num_predictions = box_logits.shape[-2]
    num_targets = target_boxes.shape[-2]
    assignments = torch.tensor(
        list(permutations(range(num_targets), num_predictions)),
        device=target_boxes.device,
    )
    prediction_indices = torch.arange(num_predictions, device=target_boxes.device)
    assignment_costs = matching_cost[..., prediction_indices, assignments]
    valid_targets = target_boxes[..., 0].bool()
    selected_targets = valid_targets[..., assignments]
    required_targets = valid_targets.sum(dim=-1).clamp_max(num_predictions)
    valid_assignments = selected_targets.sum(dim=-1) == required_targets.unsqueeze(-1)
    assignment_costs = (assignment_costs * selected_targets).sum(dim=-1)
    assignment_costs = assignment_costs.masked_fill(~valid_assignments, torch.inf)
    best_assignment = assignments[assignment_costs.argmin(dim=-1)]
    matched = target_boxes.gather(
        -2,
        best_assignment.unsqueeze(-1).expand(*best_assignment.shape, 5),
    )
    return matched


def center_size_to_corners(coordinates: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert cell-relative centers and image-relative sizes to comparable corners."""
    center = coordinates[..., :2] / grid_size
    half_size = coordinates[..., 2:] / 2
    return torch.cat((center - half_size, center + half_size), dim=-1)


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two collections of corner-format boxes.

    Args:
        boxes1 (torch.Tensor): Boxes with shape ``[..., N, 4]``.
        boxes2 (torch.Tensor): Boxes with shape ``[..., M, 4]``.

    Returns:
        torch.Tensor: Pairwise IoU values with shape ``[..., N, M]``.
    """
    intersection_minimum = torch.maximum(
        boxes1[..., None, :2], boxes2[..., None, :, :2]
    )
    intersection_maximum = torch.minimum(
        boxes1[..., None, 2:], boxes2[..., None, :, 2:]
    )
    intersection = (intersection_maximum - intersection_minimum).clamp_min(0).prod(-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).prod(-1)[..., None]
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).prod(-1)[..., None, :]
    return intersection / (area1 + area2 - intersection).clamp_min(1e-7)


def decode_predictions(
    predictions: torch.Tensor, grid_size: int, num_boxes: int
) -> list[dict[str, torch.Tensor]]:
    """Decode µYOLO outputs into normalized boxes and confidence scores."""
    batch_size = predictions.shape[0]
    box_logits = predictions.reshape(batch_size, grid_size, grid_size, num_boxes, 5)
    confidence = torch.sigmoid(box_logits[..., 0])
    coordinates = torch.sigmoid(box_logits[..., 1:])
    rows, columns = torch.meshgrid(
        torch.arange(grid_size, device=predictions.device),
        torch.arange(grid_size, device=predictions.device),
        indexing="ij",
    )
    center_x = (coordinates[..., 0] + columns[None, :, :, None]) / grid_size
    center_y = (coordinates[..., 1] + rows[None, :, :, None]) / grid_size
    width = coordinates[..., 2]
    height = coordinates[..., 3]
    boxes = torch.stack(
        [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ],
        dim=-1,
    ).clamp(0, 1)
    return [
        {"boxes": boxes[index].reshape(-1, 4), "scores": confidence[index].reshape(-1)}
        for index in range(batch_size)
    ]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    grid_size: int,
    num_boxes: int,
) -> float:
    """Evaluate the model's mAP at an IoU threshold of 0.5."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5])
    with torch.no_grad():
        for images, targets in loader:
            predictions = decode_predictions(
                model(images.to(device)), grid_size, num_boxes
            )
            metric_predictions = []
            metric_targets = []
            for prediction, target in zip(predictions, targets):
                boxes = prediction["boxes"]
                scores = prediction["scores"]
                keep = (
                    batched_nms(
                        boxes,
                        scores,
                        torch.zeros_like(scores, dtype=torch.long),
                        NMS_IOU_THRESHOLD,
                    )
                    if len(boxes)
                    else torch.empty(0, dtype=torch.long, device=boxes.device)
                )
                metric_predictions.append(
                    {
                        "boxes": (boxes[keep] * IMAGE_SIZE).cpu(),
                        "scores": scores[keep].cpu(),
                        "labels": torch.zeros(len(keep), dtype=torch.long),
                    }
                )
                metric_targets.append(
                    {
                        "boxes": target["boxes"],
                        "labels": torch.zeros(len(target["boxes"]), dtype=torch.long),
                    }
                )
            metric.update(metric_predictions, metric_targets)
    return metric.compute()["map_50"].item()


def l1(tensor: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Compute detached L1 magnitudes over the specified dimensions."""
    return tensor.detach().abs().sum(dim=dimensions)


def feature_importance(model: MicroYolo) -> list[torch.Tensor]:
    """Score backbone channels by the L1 magnitude of connected weights.

    This is used for selecting which channels to mask for pruning.

    Args:
        model (MicroYolo): Model whose backbone channels are scored.

    Returns:
        list[torch.Tensor]: Per-layer channel importance scores.
    """
    stem = model.backbone[0]
    layers = [model.backbone[index] for index in (4, 5, 6, 8, 9, 10, 11)]
    if not isinstance(stem, nn.Conv2d) or not all(
        isinstance(layer, nn.Sequential) for layer in layers
    ):
        raise ValueError("Unexpected µYOLO backbone.")
    depthwise = [layer[0] for layer in layers]
    pointwise = [layer[1] for layer in layers]
    if not all(isinstance(layer, nn.Conv2d) for layer in depthwise + pointwise):
        raise ValueError("Unexpected µYOLO depthwise-separable layer.")

    scores = [
        l1(stem.weight, (1, 2, 3))
        + l1(depthwise[0].weight, (1, 2, 3))
        + l1(pointwise[0].weight, (0, 2, 3))
    ]
    for index in range(6):
        scores.append(
            l1(pointwise[index].weight, (1, 2, 3))
            + l1(depthwise[index + 1].weight, (1, 2, 3))
            + l1(pointwise[index + 1].weight, (0, 2, 3))
        )

    first_linear = model.head[0]
    if not isinstance(first_linear, nn.Linear):
        raise ValueError("Unexpected µYOLO detection head.")
    spatial_size = first_linear.in_features // pointwise[-1].out_channels
    columns = (
        torch.arange(pointwise[-1].out_channels, device=first_linear.weight.device)[
            :, None
        ]
        * spatial_size
        + torch.arange(spatial_size, device=first_linear.weight.device)
    ).reshape(-1)
    scores.append(
        l1(pointwise[-1].weight, (1, 2, 3))
        + first_linear.weight[:, columns]
        .detach()
        .abs()
        .reshape(first_linear.out_features, -1, spatial_size)
        .sum(dim=(0, 2))
    )
    return scores


@torch.no_grad()
def update_channel_mask(
    mask: ChannelMask, importance: torch.Tensor, target_sparsity: float
) -> None:
    """Keep the most important active channels at the requested sparsity."""
    keep_count = max(1, round(importance.numel() * (1 - target_sparsity)))
    active = mask.mask.nonzero(as_tuple=True)[0]
    kept = active[importance[active].topk(keep_count).indices]
    mask.mask.zero_()
    mask.mask[kept] = 1


def apply_pruning(model: MicroYolo, target_sparsity: float) -> None:
    """Apply structured channel pruning to the backbone and detection head."""
    for mask, importance in zip(model.feature_masks, feature_importance(model)):
        update_channel_mask(mask, importance, target_sparsity)

    first_linear = model.head[0]
    output_linear = model.head[3]
    hidden_importance = l1(first_linear.weight, (1,)) + l1(output_linear.weight, (0,))
    update_channel_mask(model.hidden_mask, hidden_importance, target_sparsity)


def pruning_target(model: MicroYolo) -> float:
    """Return the highest sparsity among the model's channel masks."""
    masks = [*model.feature_masks, model.hidden_mask]
    return max(1 - float(mask.mask.float().mean()) for mask in masks)


def pruning_complete(model: MicroYolo, target: float) -> bool:
    """Return whether every channel mask has reached the pruning target."""
    return all(
        1 - float(mask.mask.float().mean()) >= target
        for mask in [*model.feature_masks, model.hidden_mask]
    )


def make_scheduler(optimizer: torch.optim.Optimizer):
    """Create the validation-mAP learning-rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5
    )


def model_checkpoint(
    model: MicroYolo, mean_average_precision: float, target_sparsity: float
) -> dict:
    """Build a deployable checkpoint with model and pruning metadata."""
    return {
        "model": copy.deepcopy(model.state_dict()),
        "grid_size": model.grid_size,
        "num_classes": model.num_classes,
        "num_boxes": model.num_boxes,
        "mAP@0.5": mean_average_precision,
        "dataset": "openimages_v7_person",
        "architecture": {
            "backbone_channels": list(model.backbone_channels),
            "hidden_features": model.hidden_features,
            "pruning_masks": model.pruning_masks,
        },
        "pruning": {
            "method": "iterative_gradual_l1_channel_masking",
            "target": target_sparsity,
            "targets": PRUNING_TARGETS,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Initialize model weights from a detector checkpoint.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="First epoch for a warm-started or new run.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901
    """Set up data, train µYOLO, and save the resulting checkpoints."""
    args = parse_args()
    if not all(
        exported_image_count(DATASET_DIR / split / "labels.json") == samples
        for split, samples in (
            ("train", TRAIN_SAMPLES),
            ("validation", VALIDATION_SAMPLES),
        )
    ):
        setup_dataset()
    backbone_checkpoint = PRETRAINED_PATH
    epochs = 400
    batch_size = 64
    workers = 8
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.005
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MicroYoloDataset("train", training=True)
    validation_dataset = MicroYoloDataset("validation", training=False)
    train_loader: DataLoader[tuple[torch.Tensor, list[Target]]] = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        drop_last=True,
    )
    validation_loader: DataLoader[tuple[torch.Tensor, list[Target]]] = DataLoader(
        validation_dataset,
        batch_size,
        num_workers=workers,
        collate_fn=collate,
    )
    model = MicroYolo().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = make_scheduler(optimizer)
    last_checkpoint = TRAINING_RESUME_PATH
    best_unpruned_checkpoint = TRAINED_UNPRUNED_PATH
    best_pruned_checkpoint = TRAINED_FULLY_PRUNED_PATH
    final_checkpoint = TRAINED_PATH
    best_unpruned_map = -1.0
    best_pruned_map = -1.0
    mean_average_precision = -1.0
    start_epoch = args.start_epoch
    resumed = False
    resumable_keys = {
        "epoch",
        "model",
        "grid_size",
        "num_boxes",
        "optimizer",
        "best_map",
        "rng_state",
        "dataset",
    }
    if args.input is not None:
        report_input("checkpoint", args.input)
        checkpoint = torch.load(args.input, map_location=device, weights_only=True)
        if "model" not in checkpoint:
            raise ValueError(f"{args.input} is not a model checkpoint.")
        if (
            checkpoint.get("grid_size") != model.grid_size
            or checkpoint.get("num_boxes") != model.num_boxes
        ):
            raise ValueError(
                f"{args.input} is not a {model.grid_size}x{model.grid_size}, "
                f"{model.num_boxes}-box µYOLO detector checkpoint."
            )
        model.load_state_dict(checkpoint["model"])
        print(f"Warm-starting training from {args.input} at epoch {start_epoch}.")
    elif last_checkpoint.exists():
        report_input("resume checkpoint", last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("dataset") == "openimages_v7_person"
            and checkpoint.get("grid_size") == DEFAULT_GRID_SIZE
            and checkpoint.get("num_boxes") == DEFAULT_NUM_BOXES
            and resumable_keys <= checkpoint.keys()
        ):
            model.load_state_dict(checkpoint["model"])
            optimizer = torch.optim.SGD(
                model.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler = make_scheduler(optimizer)
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            best_unpruned_map = checkpoint.get("best_unpruned_map", -1.0)
            best_pruned_map = checkpoint.get("best_pruned_map", -1.0)
            start_epoch = checkpoint["epoch"] + 1
            torch.set_rng_state(checkpoint["rng_state"])
            if device.type == "cuda" and "cuda_rng_state" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            resumed = True
            print(f"Resuming training from {last_checkpoint} at epoch {start_epoch}.")
        else:
            print(
                f"Ignoring {last_checkpoint}; it is not a resumable "
                "training checkpoint."
            )
    if args.input is None and not resumed and backbone_checkpoint.exists():
        report_input("pretrained backbone", backbone_checkpoint)
        model.backbone.load_state_dict(
            torch.load(backbone_checkpoint, map_location=device, weights_only=True)[
                "backbone"
            ]
        )
        print(f"Initializing detector backbone from {backbone_checkpoint}.")
    if not 1 <= start_epoch <= epochs:
        raise ValueError(f"start epoch must be between 1 and {epochs}")
    final_pruning_target = max(PRUNING_TARGETS.values())
    if args.input is not None:
        mean_average_precision = evaluate(
            model.eval(), validation_loader, device, model.grid_size, model.num_boxes
        )
        warm_start_target = pruning_target(model)
        if warm_start_target == 0:
            best_unpruned_map = mean_average_precision
            save_checkpoint(
                model_checkpoint(model, mean_average_precision, warm_start_target),
                best_unpruned_checkpoint,
            )
        if pruning_complete(model, final_pruning_target):
            best_pruned_map = mean_average_precision
            save_checkpoint(
                model_checkpoint(model, mean_average_precision, warm_start_target),
                best_pruned_checkpoint,
            )
        print(f"warm-start baseline mAP@0.5={mean_average_precision:.3%}")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = yolo_loss(
                model(images.to(device)), targets, model.grid_size, model.num_boxes
            )
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_sum += loss_value * len(images)
            average_loss = loss_sum / (batch_index * batch_size)
            print(
                f"\repoch={epoch:03d}/{epochs} "
                f"batch={batch_index:03d}/{len(train_loader)} "
                f"loss={loss_value:.4f} avg_loss={average_loss:.4f}",
                end="",
                flush=True,
            )
        if epoch in PRUNING_TARGETS:
            apply_pruning(model, PRUNING_TARGETS[epoch])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = PRUNING_RECOVERY_LR
            scheduler = make_scheduler(optimizer)
        mean_average_precision = evaluate(
            model.eval(), validation_loader, device, model.grid_size, model.num_boxes
        )
        print()
        current_learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} loss={loss_sum / len(train_dataset):.4f} "
            f"mAP@0.5={mean_average_precision:.3%} "
            f"lr={current_learning_rate:.2e}"
        )
        scheduler.step(mean_average_precision)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "grid_size": model.grid_size,
            "num_classes": model.num_classes,
            "num_boxes": model.num_boxes,
            "architecture": {
                "backbone_channels": list(model.backbone_channels),
                "hidden_features": model.hidden_features,
                "pruning_masks": model.pruning_masks,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map": best_unpruned_map,
            "best_unpruned_map": best_unpruned_map,
            "best_pruned_map": best_pruned_map,
            "rng_state": torch.get_rng_state(),
            "dataset": "openimages_v7_person",
            "pruning": {
                "method": "iterative_gradual_l1_channel_masking",
                "target": pruning_target(model),
                "targets": PRUNING_TARGETS,
            },
        }
        if device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        pruning_level = pruning_target(model)
        if pruning_level == 0 and mean_average_precision > best_unpruned_map:
            best_unpruned_map = mean_average_precision
            checkpoint["best_map"] = best_unpruned_map
            checkpoint["best_unpruned_map"] = best_unpruned_map
            save_checkpoint(
                model_checkpoint(model, mean_average_precision, pruning_level),
                best_unpruned_checkpoint,
            )
        if pruning_complete(model, final_pruning_target) and (
            mean_average_precision > best_pruned_map
        ):
            best_pruned_map = mean_average_precision
            checkpoint["best_pruned_map"] = best_pruned_map
            save_checkpoint(
                model_checkpoint(model, mean_average_precision, pruning_level),
                best_pruned_checkpoint,
            )
        save_checkpoint(checkpoint, last_checkpoint)
    save_checkpoint(
        {
            "model": model.state_dict(),
            "grid_size": model.grid_size,
            "num_classes": model.num_classes,
            "num_boxes": model.num_boxes,
            "dataset": "openimages_v7_person",
            "mAP@0.5": mean_average_precision,
            "architecture": {
                "backbone_channels": list(model.backbone_channels),
                "hidden_features": model.hidden_features,
                "pruning_masks": model.pruning_masks,
            },
            "pruning": {
                "method": "iterative_gradual_l1_channel_masking",
                "target": pruning_target(model),
                "targets": PRUNING_TARGETS,
            },
        },
        final_checkpoint,
    )
    report_output(final_checkpoint)


if __name__ == "__main__":
    main()
