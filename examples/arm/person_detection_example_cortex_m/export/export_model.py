# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize a trained µYOLO checkpoint for Cortex-M and eager evaluation."""

import argparse
import sys
from pathlib import Path

import torch

from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.exir import save as save_exported_program, to_edge
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs
from torch.utils.data import DataLoader
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = EXAMPLE_DIR / "training"
sys.path.insert(0, str(EXAMPLE_DIR))
from utils.artifacts import (  # type: ignore[import-not-found]
    EAGER_PATH,
    PRUNED_PATH,
    PTE_PATH,
    report_input,
    report_output,
    save_artifact,
)

CALIBRATION_SAMPLES = 100

sys.path.insert(0, str(TRAINING_DIR))
from model import (  # type: ignore[import-not-found]
    DEFAULT_BACKBONE_CHANNELS,
    DEFAULT_GRID_SIZE,
    DEFAULT_HIDDEN_FEATURES,
    DEFAULT_NUM_BOXES,
    MicroYolo,
)
from utils.dataset import collate, MicroYoloDataset  # type: ignore[import-not-found]


def model_from_checkpoint(checkpoint: dict) -> MicroYolo:
    """Create a µYOLO model matching checkpoint architecture metadata."""
    architecture = checkpoint.get("architecture", {})
    return MicroYolo(
        grid_size=checkpoint.get("grid_size", DEFAULT_GRID_SIZE),
        num_boxes=checkpoint.get("num_boxes", DEFAULT_NUM_BOXES),
        backbone_channels=tuple(
            architecture.get("backbone_channels", DEFAULT_BACKBONE_CHANNELS)
        ),
        hidden_features=architecture.get("hidden_features", DEFAULT_HIDDEN_FEATURES),
        pruning_masks=architecture.get("pruning_masks", True),
    )


def load_model(checkpoint_path: Path = PRUNED_PATH) -> MicroYolo:
    """Load a µYOLO detector checkpoint for inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a µYOLO detector checkpoint.")
    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def export_cortex_m(model: MicroYolo):
    """Quantize and lower a µYOLO model for Cortex-M."""
    example_input = torch.ones(1, 3, 128, 128)
    captured = torch.export.export(model, (example_input,))

    # Quantize
    prepared = prepare_pt2e(
        captured.module(check_guards=False), CortexMQuantizer(use_explicit_layout=True)
    )
    calibration_loader = DataLoader(
        MicroYoloDataset("train", training=False),
        batch_size=1,
        num_workers=0,
        collate_fn=collate,
    )

    for index, (images, _) in enumerate(calibration_loader, start=1):
        prepared(images)
        print(f"Calibrating {index}/{CALIBRATION_SAMPLES}", end="\r", flush=True)
        if index == CALIBRATION_SAMPLES:
            break
    print()
    quantized = convert_pt2e(prepared, fold_quantize=True)
    quantized_captured = torch.export.export(quantized, (example_input,))

    # Lower to Cortex-M operators
    edge = to_edge(quantized_captured, compile_config=cortex_m_edge_compile_config())
    edge = edge.transform(
        passes=[QuantizeInputs(edge, [0]), QuantizeOutputs(edge, [0])]
    )
    pass_manager = CortexMPassManager(
        target_config=CortexMTargetConfig(cpu=CortexM.M55),
        use_explicit_layout=True,
    )
    edge = edge.transform(pass_manager)

    return edge


def main() -> None:
    """Export checkpoint artifacts for Cortex-M and eager evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--pte", type=Path, default=PTE_PATH)
    parser.add_argument("--eager", type=Path, default=EAGER_PATH)
    args = parser.parse_args()
    input_path = args.input or PRUNED_PATH
    report_input("checkpoint", input_path)
    model = load_model(input_path)
    edge = export_cortex_m(model)
    save_artifact(
        args.eager, lambda path: save_exported_program(edge.exported_program(), path)
    )
    pte_bytes = edge.to_executorch().buffer
    save_artifact(args.pte, lambda path: path.write_bytes(pte_bytes))
    report_output(args.pte)
    report_output(args.eager)


if __name__ == "__main__":
    main()
