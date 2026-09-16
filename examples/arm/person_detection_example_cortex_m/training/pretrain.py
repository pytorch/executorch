# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-train the µYOLO backbone for classification on Caltech-256."""

import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(EXAMPLE_DIR))
from model import MicroYolo  # type: ignore[import-not-found]
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import (  # type: ignore[import-not-found, import-untyped]
    datasets,
    transforms,
)
from utils.artifacts import (  # type: ignore[import-not-found]
    PRETRAIN_RESUME_PATH,
    PRETRAINED_PATH,
    report_input,
    report_output,
    save_checkpoint,
)

TRAINING_DIR = Path(__file__).parent
DATASETS_DIR = TRAINING_DIR / "datasets"


class BackboneClassifier(nn.Module):
    """MicroYolo backbone with classifier head used in pretraining."""

    def __init__(self) -> None:
        super().__init__()
        detector = MicroYolo()
        self.backbone = detector.backbone
        self.classifier = nn.Linear(1024, 256)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.backbone(images), 1))


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Computes top1/ top5 score on the given datset."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            labels = labels.to(device)
            correct_top1 += (logits.argmax(1) == labels).sum().item()
            correct_top5 += (
                logits.topk(5, dim=1).indices.eq(labels[:, None]).any(1).sum().item()
            )
            total += labels.numel()
    return correct_top1 / total, correct_top5 / total


def main() -> None:
    # Set constants
    data_root = DATASETS_DIR / "caltech256"
    last_checkpoint = PRETRAIN_RESUME_PATH
    best_checkpoint = PRETRAINED_PATH
    epochs = 400
    batch_size = 64
    workers = 4
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.005
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(146),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize,
        ]
    )
    base_dataset = datasets.Caltech256(data_root, download=True)
    generator = torch.Generator().manual_seed(0)
    indices = [index for index, target in enumerate(base_dataset.y) if target < 256]
    indices = torch.tensor(indices)[
        torch.randperm(len(indices), generator=generator)
    ].tolist()
    split = int(0.8 * len(indices))

    # Setup datasets
    train_dataset = Subset(
        datasets.Caltech256(data_root, transform=train_transform, download=True),
        indices[:split],
    )
    validation_dataset = Subset(
        datasets.Caltech256(data_root, transform=validation_transform, download=True),
        indices[split:],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    # Model and training setup
    model = BackboneClassifier().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_top1 = -1.0
    start_epoch = 1
    if last_checkpoint.exists():
        report_input("resume checkpoint", last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_top1 = checkpoint["best_top1"]
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["rng_state"])
        print(f"Resuming pretraining from {last_checkpoint} at epoch {start_epoch}.")

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            print(
                f"\repoch={epoch:03d}/{epochs} batch={batch_index:03d}/{len(train_loader)} "
                f"loss={loss.item():.4f} avg_loss={running_loss / (batch_index * batch_size):.4f}",
                end="",
                flush=True,
            )
        top1, top5 = evaluate(model, validation_loader, device)
        print()
        print(
            f"epoch={epoch:03d} loss={running_loss / len(train_dataset):.4f} "
            f"top1={top1:.3%} top5={top5:.3%}"
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_top1": best_top1,
            "rng_state": torch.get_rng_state(),
        }
        if top1 > best_top1:
            best_top1 = top1
            checkpoint["best_top1"] = best_top1
            save_checkpoint(
                {"backbone": model.backbone.state_dict(), "top1": top1, "top5": top5},
                best_checkpoint,
            )
        save_checkpoint(checkpoint, last_checkpoint)

    report_output(best_checkpoint)


if __name__ == "__main__":
    main()
