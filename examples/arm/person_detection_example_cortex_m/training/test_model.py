# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Run a trained µYOLO detector on a validation image."""

import argparse
import secrets
import sys
from collections.abc import Callable
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(EXAMPLE_DIR))
from model import (  # type: ignore[import-not-found]
    DEFAULT_BACKBONE_CHANNELS,
    DEFAULT_GRID_SIZE,
    DEFAULT_HIDDEN_FEATURES,
    DEFAULT_NUM_BOXES,
    MicroYolo,
)
from PIL import Image, ImageDraw  # type: ignore[import-not-found, import-untyped]
from torchvision.ops import nms  # type: ignore[import-not-found, import-untyped]

from training import (  # type: ignore[import-not-found]
    CONFIDENCE_THRESHOLD,
    decode_predictions,
    NMS_IOU_THRESHOLD,
)
from utils.artifacts import (  # type: ignore[import-not-found]
    PRUNED_PATH,
    report_input,
    report_output,
    RESULTS_DIR,
)
from utils.dataset import (  # type: ignore[import-not-found]
    DATASET_DIR,
    make_image_transform,
)


def random_validation_image() -> Path:
    images = list((DATASET_DIR / "validation" / "data").glob("*.jpg"))
    return secrets.choice(images)


def preprocess_webcam_frame(frame) -> torch.Tensor:
    import cv2  # type: ignore[import-not-found, import-untyped]

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return make_image_transform(training=False)(image).unsqueeze(0)


@torch.no_grad()
def display_webcam_predictions(
    model: torch.nn.Module,
    decode_predictions: Callable[
        [torch.Tensor, int, int], list[dict[str, torch.Tensor]]
    ],
    grid_size: int,
    num_boxes: int,
    device: torch.device,
    confidence_threshold: float,
    nms_iou_threshold: float,
    title: str,
) -> None:
    try:
        import cv2  # type: ignore[import-not-found, import-untyped]
    except ImportError as error:
        raise RuntimeError("Webcam support requires opencv-python.") from error
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam 0.")
    print("Webcam active. Press q or Escape to stop.")
    try:
        while True:
            captured, frame = camera.read()
            if not captured:
                raise RuntimeError("Could not read a frame from webcam 0.")
            prediction = decode_predictions(
                model(preprocess_webcam_frame(frame).to(device)), grid_size, num_boxes
            )[0]
            height, width = frame.shape[:2]
            prediction_boxes = prediction["boxes"]
            keep = (
                (prediction["scores"] >= confidence_threshold)
                & (prediction_boxes[:, 2] > prediction_boxes[:, 0])
                & (prediction_boxes[:, 3] > prediction_boxes[:, 1])
            )
            boxes, scores = prediction["boxes"][keep], prediction["scores"][keep]
            if len(boxes):
                keep = nms(boxes, scores, nms_iou_threshold)
                boxes, scores = boxes[keep].cpu(), scores[keep].cpu()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.tolist()
                cv2.rectangle(
                    frame,
                    (int(x1 * width), int(y1 * height)),
                    (int(x2 * width), int(y2 * height)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"person {score:.0%}",
                    (int(x1 * width), max(int(y1 * height) - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyWindow(title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--webcam", action="store_true", help="run continuous inference on webcam 0"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.pt or PRUNED_PATH
    report_input("checkpoint", checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (
        checkpoint.get("grid_size") != DEFAULT_GRID_SIZE
        or checkpoint.get("num_boxes") != DEFAULT_NUM_BOXES
    ):
        raise ValueError(
            f"{checkpoint_path} is not a 7x7, one-box µYOLO detector checkpoint."
        )
    architecture = checkpoint.get("architecture", {})
    model = MicroYolo(
        grid_size=checkpoint["grid_size"],
        num_boxes=checkpoint["num_boxes"],
        backbone_channels=tuple(
            architecture.get("backbone_channels", DEFAULT_BACKBONE_CHANNELS)
        ),
        hidden_features=architecture.get("hidden_features", DEFAULT_HIDDEN_FEATURES),
        pruning_masks=architecture.get("pruning_masks", True),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.webcam:
        display_webcam_predictions(
            model,
            decode_predictions,
            checkpoint["grid_size"],
            checkpoint["num_boxes"],
            device,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            title="µYOLO predictions",
        )
        return

    image_path = args.image or random_validation_image()
    with Image.open(image_path) as source:
        model_input = make_image_transform(training=False)(source).unsqueeze(0)

    with torch.no_grad():
        prediction = decode_predictions(
            model(model_input.to(device)),
            checkpoint["grid_size"],
            checkpoint["num_boxes"],
        )[0]
    prediction_boxes = prediction["boxes"]
    keep = (
        (prediction["scores"] >= CONFIDENCE_THRESHOLD)
        & (prediction_boxes[:, 2] > prediction_boxes[:, 0])
        & (prediction_boxes[:, 3] > prediction_boxes[:, 1])
    )
    boxes = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    if len(boxes):
        keep = nms(boxes, scores, NMS_IOU_THRESHOLD)
        boxes = boxes[keep].cpu()
        scores = scores[keep].cpu()

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        coordinates = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(coordinates, outline="red", width=2)
        label = f"person {score:.0%}"
        label_box = draw.textbbox(coordinates[:2], label)
        draw.rectangle(label_box, fill="red")
        draw.text(coordinates[:2], label, fill="white")
    output_path = RESULTS_DIR / "test_model.predictions.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    image.show(title="µYOLO predictions")
    print(f"image: {image_path}")
    print(f"detections: {len(boxes)}")
    report_output(output_path)


if __name__ == "__main__":
    main()
