#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example: Real-world inference with YOLO26 CUDA models.

This demonstrates:
1. Loading exported YOLO26 model
2. Processing images/video frames
3. Parsing detection outputs
4. Drawing results on images
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def load_model(model_dir: Path):
    """Load ExecuTorch YOLO26 model."""
    from executorch.runtime import Runtime

    model_path = model_dir / "model.pte"

    with open(model_path, "rb") as f:
        pte_buffer = f.read()

    runtime = Runtime.get()
    program = runtime.load_program(pte_buffer)
    method = program.load_method("forward")

    if method is None:
        raise ValueError("Failed to load forward method")

    return method


def preprocess_image(image: np.ndarray, target_size: tuple = (640, 640)) -> tuple:
    """
    Preprocess image for YOLO inference.

    Args:
        image: Input image (BGR or RGB format)
        target_size: Target size (height, width)

    Returns:
        (input_tensor, scale, pad) tuple
    """
    original_shape = image.shape[:2]

    img_h, img_w = original_shape
    target_h, target_w = target_size

    scale = min(target_h / img_h, target_w / img_w)

    new_h = int(img_h * scale)
    new_w = int(img_w * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB) if len(padded.shape) == 3 else padded
    img_float = img_rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.contiguous()

    return tensor, scale, (pad_w, pad_h)


def postprocess_detections(
    outputs: tuple,
    original_shape: tuple,
    scale: float,
    pad: tuple,
    conf_threshold: float = 0.5
) -> list:
    """
    Post-process detection outputs.

    Args:
        outputs: Model outputs
        original_shape: Original image shape (h, w)
        scale: Resize scale factor
        pad: Padding (pad_w, pad_h)
        conf_threshold: Confidence threshold

    Returns:
        List of detections: [(x1, y1, x2, y2, conf, cls), ...]
    """
    detections = outputs[0]

    if len(detections.shape) != 3:
        return []

    pad_w, pad_h = pad
    detections_list = []

    for det in detections[0]:
        if len(det) < 6:
            continue

        x_center, y_center, width, height, conf, cls = det[:6]

        if conf < conf_threshold:
            continue

        x1 = (x_center - width / 2 - pad_w) / scale
        y1 = (y_center - height / 2 - pad_h) / scale
        x2 = (x_center + width / 2 - pad_w) / scale
        y2 = (y_center + height / 2 - pad_h) / scale

        x1 = max(0, min(x1, original_shape[1]))
        y1 = max(0, min(y1, original_shape[0]))
        x2 = max(0, min(x2, original_shape[1]))
        y2 = max(0, min(y2, original_shape[0]))

        detections_list.append((
            int(x1), int(y1), int(x2), int(y2),
            float(conf), int(cls)
        ))

    return detections_list


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    img_draw = image.copy()

    for x1, y1, x2, y2, conf, cls in detections:
        color = (0, 255, 0)

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        label = f"{COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else cls}: {conf:.2f}"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y1 = max(y1, label_size[1] + 10)

        cv2.rectangle(
            img_draw,
            (x1, label_y1 - label_size[1] - 10),
            (x1 + label_size[0], label_y1),
            color,
            -1
        )

        cv2.putText(
            img_draw,
            label,
            (x1, label_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    return img_draw


def process_image(
    model,
    image_path: str,
    output_path: str = None,
    conf_threshold: float = 0.5,
    target_size: tuple = (640, 640)
):
    """Process a single image."""
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    original_shape = image.shape[:2]
    print(f"Original image shape: {original_shape}")

    input_tensor, scale, pad = preprocess_image(image, target_size)
    print(f"Preprocessed tensor shape: {input_tensor.shape}")

    print("Running inference...")
    outputs = model.execute((input_tensor,))

    print("Post-processing detections...")
    detections = postprocess_detections(
        outputs, original_shape, scale, pad, conf_threshold
    )

    print(f"Found {len(detections)} detections")

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections[:10]):
        class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
        print(f"  {i+1}. {class_name}: {conf:.3f} at ({x1}, {y1}, {x2}, {y2})")

    if len(detections) > 10:
        print(f"  ... and {len(detections) - 10} more")

    result_image = draw_detections(image, detections)

    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Saved result to: {output_path}")
    else:
        cv2.imshow("YOLO26 Detection Results", result_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    model,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.5,
    target_size: tuple = (640, 640)
):
    """Process video file or camera stream."""
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        print(f"Opening camera {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps} FPS")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing output to: {output_path}")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            input_tensor, scale, pad = preprocess_image(frame, target_size)

            outputs = model.execute((input_tensor,))

            detections = postprocess_detections(
                outputs, frame.shape[:2], scale, pad, conf_threshold
            )

            result_frame = draw_detections(frame, detections)

            cv2.putText(
                result_frame,
                f"Frame {frame_count} | Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if writer:
                writer.write(result_frame)
            else:
                cv2.imshow("YOLO26 Video Detection", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"Processed {frame_count} total frames")
        if output_path:
            print(f"Saved result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 CUDA inference example"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing model.pte and aoti_cuda_blob.ptd",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image or video file (use '0' for webcam)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, will display if not provided)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Target size for inference (default: 640 640)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    print(f"{'='*70}")
    print("YOLO26 CUDA Inference Example")
    print(f"{'='*70}")
    print(f"Model: {args.model_dir}")
    print(f"Input: {args.input}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Target size: {args.target_size}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    print("Loading model...")
    model = load_model(args.model_dir)
    print("Model loaded successfully!\n")

    target_size = tuple(args.target_size)

    is_video = (
        args.input.isdigit()
        or args.input.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )

    if is_video:
        process_video(
            model, args.input, args.output,
            args.conf_threshold, target_size
        )
    else:
        process_image(
            model, args.input, args.output,
            args.conf_threshold, target_size
        )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
