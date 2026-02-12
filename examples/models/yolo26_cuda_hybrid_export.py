#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export YOLO26 with hybrid CPU/CUDA execution:
- Preprocessing: CPU
- Core inference (backbone + head): CUDA
- Postprocessing (NMS, topk): CPU

This avoids CUDA AOTI backend limitations while maximizing GPU utilization.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from ultralytics import YOLO

# CRITICAL: Set multiprocessing to 'spawn' mode for CUDA compatibility
# AOT Inductor spawns subprocesses to compile Triton kernels, and 'fork' doesn't work with CUDA
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


MODEL_SIZES = ["n", "s", "m", "l", "x"]
TASK_TYPES = ["", "-seg", "-pose", "-obb", "-cls"]

EDGE_COMPILE_CONFIG = EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,
)


def get_all_model_variants() -> list[str]:
    """Generate all YOLO26 model variant names."""
    variants = []
    for size in MODEL_SIZES:
        for task in TASK_TYPES:
            variants.append(f"yolo26{size}{task}")
    return variants


def create_test_image(task_type: str) -> np.ndarray:
    """Create test image for different task types."""
    width, height = 640, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    if task_type == "-cls":
        cv2.putText(img, "Test", (250, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    else:
        cv2.rectangle(img, (100, 150), (180, 400), (100, 150, 100), -1)
        cv2.circle(img, (140, 150), 40, (150, 120, 100), -1)
        cv2.rectangle(img, (300, 250), (500, 400), (150, 50, 50), -1)

    return img


class YOLOBackboneOnly(torch.nn.Module):
    """
    Wrapper that extracts only the backbone and head raw outputs.
    Excludes postprocessing (NMS, topk, etc) which runs on CPU.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._original_postprocess = {}

        # Replace postprocess method in Detect head to bypass it during tracing
        for m in self.model.modules():
            if type(m).__name__ in ['Detect', 'Segment', 'Pose', 'OBB', 'Classify']:
                # Save original postprocess method
                if hasattr(m, 'postprocess'):
                    self._original_postprocess[id(m)] = m.postprocess
                    # Replace with passthrough that just concatenates outputs
                    m.postprocess = lambda x: x  # noqa: E731
                    print(f"  Disabled postprocessing on {type(m).__name__} head")
                m.export = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run YOLO backbone and head, return raw predictions before postprocessing.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Raw predictions before NMS/postprocessing
        """
        # Run inference - postprocess is disabled, returns raw outputs
        output = self.model(x)

        # Return the raw tensor
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


def export_model_variant(
    model_name: str,
    base_path: Path,
    test_model: bool = True,
    skip_existing: bool = False,
) -> Tuple[bool, str]:
    """Export YOLO26 model with CUDA backend (backbone only, CPU for pre/post)."""
    variant_dir = base_path / model_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_pte_path = variant_dir / "model.pte"
    data_path = variant_dir / "aoti_cuda_blob.ptd"

    if skip_existing and model_pte_path.exists() and data_path.exists():
        return True, "Skipped (already exists)"

    try:
        print(f"\n{'='*70}")
        print(f"Exporting {model_name} (CUDA backbone + CPU pre/post)")
        print(f"{'='*70}")

        print(f"Loading model from Ultralytics...")
        model = YOLO(f"{model_name}.pt")

        input_dims = (640, 640)
        task_type = ""
        if "-seg" in model_name:
            task_type = "-seg"
        elif "-pose" in model_name:
            task_type = "-pose"
        elif "-obb" in model_name:
            task_type = "-obb"
        elif "-cls" in model_name:
            task_type = "-cls"

        dummy_img = create_test_image(task_type)

        print(f"Preparing model (backbone only)...")
        model.predict(dummy_img, imgsz=input_dims, device="cpu")
        pt_model = model.model.to(torch.device("cpu"))
        pt_model.eval()

        # Wrap model to extract backbone only
        backbone_model = YOLOBackboneOnly(pt_model)
        backbone_model.eval()

        def transform_fn(frame):
            return model.predictor.preprocess([frame])

        example_input = transform_fn(dummy_img)

        print(f"Input shape: {example_input.shape}")

        print(f"Exporting to ATEN dialect...")
        with torch.no_grad():
            exported_program = torch.export.export(backbone_model, args=(example_input,))

        # Print operators used
        ops_used = set()
        for node in exported_program.graph.nodes:
            if node.op == 'call_function':
                ops_used.add(str(node.target))

        print(f"\nOperators in exported graph: {len(ops_used)}")

        # Check for problematic operators
        problematic_ops = ['index_put', 'topk', 'index.Tensor']
        found_problematic = [op for op in problematic_ops if any(op in str(used_op) for used_op in ops_used)]

        if found_problematic:
            print(f"⚠️  Warning: Found potentially problematic operators: {found_problematic}")
            print(f"   These may not be supported by CUDA AOTI backend")
        else:
            print(f"✓ No problematic operators found")

        print(f"\nLowering to CUDA backend...")

        # Ensure CUDA is initialized before compilation
        if torch.cuda.is_available():
            # Touch CUDA to ensure it's initialized
            _ = torch.zeros(1, device='cuda')
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        partitioner = CudaPartitioner(
            [CudaBackend.generate_method_name_compile_spec(model_name)]
        )

        et_prog = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=EDGE_COMPILE_CONFIG,
        )

        print(f"Converting to ExecuTorch program...")
        exec_program = et_prog.to_executorch()

        print(f"Saving model to {model_pte_path}...")
        with open(model_pte_path, "wb") as f:
            exec_program.write_to_file(f)

        print(f"Saved:")
        print(f"  - {model_pte_path} ({model_pte_path.stat().st_size / 1024 / 1024:.2f} MB)")
        if data_path.exists():
            print(f"  - {data_path} ({data_path.stat().st_size / 1024 / 1024:.2f} MB)")

        if test_model:
            print(f"\nTesting model inference...")
            test_success, test_msg = test_model_inference(
                model_pte_path, data_path, example_input
            )
            if not test_success:
                return False, f"Export succeeded but test failed: {test_msg}"
            print(f"✓ Test passed: {test_msg}")

        return True, "Success"

    except Exception as e:
        error_msg = f"Failed: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg


def test_model_inference(
    model_path: Path,
    data_path: Path,
    example_input: torch.Tensor,
) -> Tuple[bool, str]:
    """Test inference on exported CUDA model."""
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available for testing"

        from executorch.runtime import Runtime

        runtime = Runtime.get()

        with open(model_path, "rb") as f:
            program = runtime.load_program(f.read())

        method = program.load_method("forward")
        if method is None:
            return False, "Failed to load forward method"

        input_tensor = example_input.contiguous()

        outputs = method.execute((input_tensor,))

        if outputs is None or len(outputs) == 0:
            return False, "No outputs returned"

        output_shape = outputs[0].shape if hasattr(outputs[0], 'shape') else None

        return True, f"Raw output shape: {output_shape} (needs CPU postprocessing)"

    except Exception as e:
        return False, f"Inference failed: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO26 with CUDA backend (hybrid CPU/GPU execution)"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base directory for exported models",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific model variants to export (default: all)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Test models after export (default: True)",
    )
    parser.add_argument(
        "--no-test",
        action="store_false",
        dest="test",
        help="Skip testing models after export",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have exported files",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available.")
        print("The model will export but may not run correctly without CUDA.")

    args.base_path.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_variants = args.models
    else:
        model_variants = get_all_model_variants()

    print(f"\n{'='*70}")
    print(f"YOLO26 Hybrid CUDA Export")
    print(f"{'='*70}")
    print(f"Architecture: CPU preprocessing -> CUDA backbone -> CPU postprocessing")
    print(f"Base path: {args.base_path.absolute()}")
    print(f"Models to export: {len(model_variants)}")
    print(f"\nModels: {', '.join(model_variants)}")

    results = {}

    for i, model_name in enumerate(model_variants, 1):
        print(f"\n[{i}/{len(model_variants)}] Processing {model_name}...")

        success, message = export_model_variant(
            model_name,
            args.base_path,
            test_model=args.test,
            skip_existing=args.skip_existing,
        )

        results[model_name] = (success, message)

    print(f"\n{'='*70}")
    print(f"Export Summary")
    print(f"{'='*70}")

    successful = sum(1 for s, _ in results.values() if s)
    failed = len(results) - successful

    print(f"\nTotal: {len(results)} | Success: {successful} | Failed: {failed}\n")

    if successful > 0:
        print("✓ Successful exports:")
        for model_name, (success, msg) in results.items():
            if success:
                print(f"  {model_name}: {msg}")

    if failed > 0:
        print("\n✗ Failed exports:")
        for model_name, (success, msg) in results.items():
            if not success:
                print(f"  {model_name}: {msg}")

    print(f"\nNote: Exported models contain CUDA backbone only.")
    print(f"You must implement CPU preprocessing and postprocessing separately.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
