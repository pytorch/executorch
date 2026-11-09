# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical paths and reporting for generated µYOLO artifacts."""

from collections.abc import Callable
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = EXAMPLE_DIR / "artifacts"  # Base folder for all artifacts
PRETRAIN_RESUME_PATH = (
    ARTIFACTS_DIR / "pretrain.resume.pt"
)  # Intermediate pretrained model
PRETRAINED_PATH = (
    ARTIFACTS_DIR / "pretrained.pt"
)  # Final pretrained model, input to training.py
TRAINING_RESUME_PATH = (
    ARTIFACTS_DIR / "training.resume.pt"
)  # Intermediate trained model
TRAINED_UNPRUNED_PATH = (
    ARTIFACTS_DIR / "trained_unpruned.pt"
)  # Final best validation mAP@05 unpruned model
TRAINED_FULLY_PRUNED_PATH = (
    ARTIFACTS_DIR / "trained_fully_pruned.pt"
)  # Final best validation mAP@05 masking pruned model, input to pruning.py
PRUNED_PATH = (
    ARTIFACTS_DIR / "pruned.pt"
)  # Structurally pruned model, input to export_model.py
EAGER_PATH = (
    ARTIFACTS_DIR / "person_detection.eager"
)  # Eager lowered artifact, used by test_exported_model.py
PTE_PATH = (
    ARTIFACTS_DIR / "person_detection.pte"
)  # Final lowered artifact, input to runtime
RESULTS_DIR = ARTIFACTS_DIR / "results"  # Prediction outputs


def relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(EXAMPLE_DIR)
    except ValueError:
        return path


def report_input(label: str, path: Path) -> None:
    print(f"Using {label}: {relative(path)}")


def report_output(path: Path) -> None:
    print(f"Produced {relative(path)} ({path.stat().st_size} bytes)")


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    """Atomically save a checkpoint to disk."""
    save_artifact(path, lambda temporary_path: torch.save(checkpoint, temporary_path))


def save_artifact(path: Path, writer: Callable[[Path], None]) -> None:
    """Atomically write an exported artifact to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    writer(temporary_path)
    temporary_path.replace(path)
