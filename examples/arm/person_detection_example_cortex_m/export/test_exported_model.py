# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate the exported µYOLO Cortex-M graph in eager mode."""

import argparse
import sys
from pathlib import Path

import executorch.backends.cortex_m.ops.operators  # noqa: F401

import torch
from executorch.exir import load as load_exported_program
from executorch.exir._serialize._program import deserialize_pte_binary
from torch.utils.data import DataLoader

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))
from utils.artifacts import (  # type: ignore[import-not-found]
    EAGER_PATH,
    PTE_PATH,
    report_input,
)
from utils.pte_metadata import read_io_qparams  # type: ignore[import-not-found]

sys.path.insert(0, str(EXAMPLE_DIR / "training"))
from model import DEFAULT_GRID_SIZE, DEFAULT_NUM_BOXES  # type: ignore[import-not-found]
from test_model import display_webcam_predictions  # type: ignore[import-not-found]
from training import decode_predictions, evaluate  # type: ignore[import-not-found]
from utils.dataset import collate, MicroYoloDataset  # type: ignore[import-not-found]


class EagerCortexMModel(torch.nn.Module):
    """Adapt the eager graph's int8 I/O for float pre- and post-processing."""

    def __init__(self, module: torch.nn.Module, manifest: dict) -> None:
        """Store the eager graph and its input/output quantization metadata."""
        super().__init__()
        self.module = module
        self.input_info = manifest["input"]
        self.output_scale = float(manifest["output"]["scale"])
        self.output_zero_point = int(manifest["output"]["zero_point"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run quantized eager inference and dequantize its output."""
        quantized = torch.round(
            images / self.input_info["scale"] + self.input_info["zero_point"]
        )
        quantized = quantized.clamp(
            self.input_info["quant_min"], self.input_info["quant_max"]
        ).to(torch.int8)
        output = self.module(quantized)
        return (output.to(torch.float32) - self.output_zero_point) * self.output_scale


def parse_args() -> argparse.Namespace:
    """Parse eager evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webcam", action="store_true", help="run continuous inference on webcam 0"
    )
    parser.add_argument("--pte", type=Path, default=PTE_PATH)
    parser.add_argument("--eager", type=Path, default=EAGER_PATH)
    return parser.parse_args()


def main() -> None:
    """Evaluate exported Cortex-M artifacts in eager mode."""
    args = parse_args()
    if not args.pte.exists() or not args.eager.exists():
        raise FileNotFoundError("Run `python export/export_model.py` first.")
    report_input("PTE", args.pte)
    report_input("eager graph", args.eager)
    deserialize_pte_binary(args.pte.read_bytes())
    try:
        exported_program = load_exported_program(args.eager)
    except (KeyError, RuntimeError) as error:
        raise RuntimeError(
            "The eager artifact was produced by an older exporter. Run "
            "`python export/export_model.py` to replace it."
        ) from error
    qparams = read_io_qparams(args.pte)
    manifest = {
        "input": {
            "scale": qparams["input0_scale"],
            "zero_point": qparams["input0_zp"],
            "quant_min": qparams["input0_quant_min"],
            "quant_max": qparams["input0_quant_max"],
        },
        "output": {
            "scale": qparams["output0_scale"],
            "zero_point": qparams["output0_zp"],
        },
        "model": {"grid_size": DEFAULT_GRID_SIZE, "num_boxes": DEFAULT_NUM_BOXES},
        "postprocessing": {
            "confidence_threshold": 0.5,
            "nms_iou_threshold": 0.5,
        },
    }
    eager_graph = exported_program.module()
    eager_model = EagerCortexMModel(eager_graph, manifest)
    if args.webcam:
        display_webcam_predictions(
            eager_model,
            decode_predictions,
            manifest["model"]["grid_size"],
            manifest["model"]["num_boxes"],
            torch.device("cpu"),
            confidence_threshold=manifest["postprocessing"]["confidence_threshold"],
            nms_iou_threshold=manifest["postprocessing"]["nms_iou_threshold"],
            title="µYOLO exported predictions",
        )
        return
    loader = DataLoader(
        MicroYoloDataset("validation", training=False),
        batch_size=1,
        num_workers=0,
        collate_fn=collate,
    )
    accuracy = evaluate(
        eager_model,
        loader,
        torch.device("cpu"),
        manifest["model"]["grid_size"],
        manifest["model"]["num_boxes"],
    )
    print(f"Validated {args.pte} and {args.eager}.")
    print(f"Open Images Person AP@0.5: {accuracy:.3%}")


if __name__ == "__main__":
    main()
