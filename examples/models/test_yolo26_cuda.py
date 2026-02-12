#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for YOLO26 CUDA exported models.

Usage:
    python test_yolo26_cuda.py --model-dir yolo26_cuda_exports/yolo26n
    python test_yolo26_cuda.py --model-dir yolo26_cuda_exports/yolo26n-seg --image test_image.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_and_preprocess_image(image_path: str, target_size: tuple = (640, 640)) -> torch.Tensor:
    """Load and preprocess image for YOLO inference."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)

    img_array = np.array(img).astype(np.float32) / 255.0

    x = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    x = x.contiguous()

    return x, img


def create_default_test_image() -> tuple:
    """Create a default test image if none provided."""
    width, height = 640, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    cv2.rectangle(img, (100, 150), (180, 400), (100, 150, 100), -1)
    cv2.ellipse(img, (140, 150), (40, 50), 0, 0, 360, (150, 120, 100), -1)

    cv2.rectangle(img, (300, 250), (500, 400), (150, 50, 50), -1)
    cv2.circle(img, (340, 400), 20, (30, 30, 30), -1)
    cv2.circle(img, (480, 400), 20, (30, 30, 30), -1)

    cv2.rectangle(img, (520, 300), (600, 450), (80, 60, 40), -1)
    cv2.rectangle(img, (525, 200), (595, 310), (80, 60, 40), -1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype(np.float32) / 255.0

    x = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    x = x.contiguous()

    pil_img = Image.fromarray(img_rgb)

    return x, pil_img


def detect_task_type(model_name: str) -> str:
    """Detect task type from model name."""
    if "-seg" in model_name:
        return "segmentation"
    elif "-pose" in model_name:
        return "pose"
    elif "-obb" in model_name:
        return "obb"
    elif "-cls" in model_name:
        return "classification"
    else:
        return "detection"


def print_detection_results(outputs, task_type: str, conf_threshold: float = 0.5):
    """Print inference results based on task type."""
    if task_type == "detection" or task_type == "obb":
        detections = outputs[0]
        print(f"\nDetection output shape: {detections.shape}")

        if len(detections.shape) == 3:
            num_detections = detections.shape[1]
            print(f"Max detections: {num_detections}")

            confident_dets = 0
            for i, det in enumerate(detections[0]):
                if len(det) >= 5:
                    conf = det[4]
                    if conf > conf_threshold:
                        confident_dets += 1
                        if confident_dets <= 5:
                            if task_type == "obb":
                                x, y, w, h, angle, conf, cls = det[:7]
                                print(f"  Detection {i}: class={int(cls)}, conf={conf:.3f}, "
                                      f"bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}), angle={angle:.1f}°")
                            else:
                                x, y, w, h, conf = det[:5]
                                cls = det[5] if len(det) > 5 else -1
                                print(f"  Detection {i}: class={int(cls)}, conf={conf:.3f}, "
                                      f"bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

            if confident_dets > 5:
                print(f"  ... and {confident_dets - 5} more detections")
            print(f"Total confident detections (conf > {conf_threshold}): {confident_dets}")

    elif task_type == "segmentation":
        print(f"\nSegmentation outputs:")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
        if len(outputs) >= 2:
            print(f"  Detections: {outputs[0].shape}")
            print(f"  Masks: {outputs[1].shape}")

    elif task_type == "pose":
        print(f"\nPose estimation outputs:")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
        if len(outputs) >= 2:
            print(f"  Detections: {outputs[0].shape}")
            print(f"  Keypoints: {outputs[1].shape}")

    elif task_type == "classification":
        logits = outputs[0]
        print(f"\nClassification output shape: {logits.shape}")

        probs = torch.softmax(logits, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

        print(f"Top 5 predictions:")
        for prob, idx in zip(top5_prob[0], top5_idx[0]):
            print(f"  Class {idx}: {prob:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO26 CUDA exported model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing model.pte and aoti_cuda_blob.ptd",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (optional, will create synthetic image if not provided)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )

    args = parser.parse_args()

    model_path = args.model_dir / "model.pte"
    data_path = args.model_dir / "aoti_cuda_blob.ptd"

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not data_path.exists():
        print(f"Warning: Data file not found: {data_path}")
        print("The model may not load correctly without the CUDA blob file.")

    print(f"{'='*70}")
    print(f"Testing YOLO26 CUDA Model")
    print(f"{'='*70}")
    print(f"Model directory: {args.model_dir}")
    print(f"Model file: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
    if data_path.exists():
        print(f"Data file: {data_path.name} ({data_path.stat().st_size / 1024 / 1024:.2f} MB)")

    if not torch.cuda.is_available():
        print("\nError: CUDA is not available. Cannot test CUDA model.")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_name = args.model_dir.name
    task_type = detect_task_type(model_name)
    print(f"Task type: {task_type}")

    if args.image:
        print(f"Loading image: {args.image}")
        input_tensor, img = load_and_preprocess_image(args.image)
    else:
        print("No image provided, creating synthetic test image...")
        input_tensor, img = create_default_test_image()

    print(f"Input tensor shape: {input_tensor.shape}")

    print("\nLoading model...")
    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()

        with open(model_path, "rb") as f:
            program = runtime.load_program(f.read())

        method = program.load_method("forward")
        if method is None:
            print("Error: Failed to load forward method")
            return 1

        print("Model loaded successfully!")

        print("\nRunning inference...")
        outputs = method.execute((input_tensor,))

        if outputs is None or len(outputs) == 0:
            print("Error: No outputs returned")
            return 1

        print("Inference completed successfully!")

        print_detection_results(outputs, task_type, args.conf_threshold)

        print(f"\n{'='*70}")
        print("✓ Test passed!")
        print(f"{'='*70}")

        return 0

    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
