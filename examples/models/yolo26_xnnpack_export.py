#!/usr/bin/env python3
# Export YOLO26 models with XNNPACK backend (CUDA AOTI has operator limitations)

import argparse
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from ultralytics import YOLO


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
    """Create appropriate test image for different task types."""
    width, height = 640, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    if task_type == "-cls":
        cv2.putText(img, "Test Classification", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.rectangle(img, (100, 150), (180, 400), (100, 150, 100), -1)
        cv2.ellipse(img, (140, 150), (40, 50), 0, 0, 360, (150, 120, 100), -1)
        cv2.rectangle(img, (300, 250), (500, 400), (150, 50, 50), -1)
        cv2.circle(img, (340, 400), 20, (30, 30, 30), -1)
        cv2.circle(img, (480, 400), 20, (30, 30, 30), -1)
        cv2.rectangle(img, (520, 300), (600, 450), (80, 60, 40), -1)
        cv2.rectangle(img, (525, 200), (595, 310), (80, 60, 40), -1)

    return img


def export_model_variant(
    model_name: str,
    base_path: Path,
    test_model: bool = True,
    skip_existing: bool = False,
) -> Tuple[bool, str]:
    """Export a single YOLO26 model variant to XNNPACK backend."""
    variant_dir = base_path / model_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_pte_path = variant_dir / "model.pte"

    if skip_existing and model_pte_path.exists():
        return True, f"Skipped (already exists)"

    try:
        print(f"\n{'='*70}")
        print(f"Exporting {model_name}")
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

        print(f"Preparing model for export...")
        model.predict(dummy_img, imgsz=input_dims, device="cpu")
        pt_model = model.model.to(torch.device("cpu"))
        pt_model.eval()

        def transform_fn(frame):
            return model.predictor.preprocess([frame])

        example_input = transform_fn(dummy_img)

        print(f"Exporting to ATEN dialect...")
        with torch.no_grad():
            exported_program = torch.export.export(pt_model, args=(example_input,))

        print(f"Lowering to XNNPACK backend...")
        partitioner = XnnpackPartitioner()

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

        print(f"Saved: {model_pte_path} ({model_pte_path.stat().st_size / 1024 / 1024:.2f} MB)")

        if test_model:
            print(f"\nTesting model inference...")
            test_success, test_msg = test_model_inference(
                model_pte_path, task_type, example_input
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
    task_type: str,
    example_input: torch.Tensor,
) -> Tuple[bool, str]:
    """Test inference on exported model."""
    try:
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
        return True, f"Output shape: {output_shape}"

    except Exception as e:
        return False, f"Inference failed: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Export all YOLO26 model variants on XNNPACK backend"
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
    args.base_path.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_variants = args.models
    else:
        model_variants = get_all_model_variants()

    print(f"\n{'='*70}")
    print(f"YOLO26 XNNPACK Export Script")
    print(f"{'='*70}")
    print(f"Base path: {args.base_path.absolute()}")
    print(f"Models to export: {len(model_variants)}")
    print(f"Backend: XNNPACK (recommended for YOLO)")
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

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
