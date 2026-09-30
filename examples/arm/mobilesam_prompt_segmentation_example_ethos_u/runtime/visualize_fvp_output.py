# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = ROOT / "arm_test" / "mobilesam"
EXPORT_DIR = WORK_DIR / "export"
RESULT_DIR = WORK_DIR / "result"
OUTPUT_SIZE = 112
POINT = (219, 193)


def load_mask(path: Path) -> np.ndarray:
    logits = np.fromfile(path, dtype=np.float32)
    if logits.size != OUTPUT_SIZE * OUTPUT_SIZE:
        raise ValueError(f"Expected {OUTPUT_SIZE**2} logits, got {logits.size}.")
    return (logits.reshape(OUTPUT_SIZE, OUTPUT_SIZE) > 0).astype(np.uint8)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(intersection / union)


def overlay(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    resized = Image.fromarray(mask * 255).resize(image.size, Image.Resampling.NEAREST)
    pixels = np.asarray(image, dtype=np.float32).copy()
    selected = np.asarray(resized) > 0
    pixels[selected] = pixels[selected] * 0.55 + np.asarray(color) * 0.45
    return Image.fromarray(pixels.astype(np.uint8))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.open(EXPORT_DIR / "input.png").convert("RGB")
    reference = (np.asarray(Image.open(EXPORT_DIR / "quantized_mask.png")) > 0).astype(
        np.uint8
    )
    fvp_mask = load_mask(WORK_DIR / "io" / "output-0.bin")
    score = iou(fvp_mask, reference)
    if score < 0.9:
        raise RuntimeError(f"FVP/reference mask IoU is too low: {score:.4f}")

    prompted = image.copy()
    ImageDraw.Draw(prompted).ellipse(
        (POINT[0] - 6, POINT[1] - 6, POINT[0] + 6, POINT[1] + 6), fill="red"
    )
    reference_overlay = overlay(image, reference, (0, 170, 255))
    fvp_overlay = overlay(image, fvp_mask, (0, 220, 120))
    panels = (
        ("Input and prompt", prompted),
        ("Host quantized mask", reference_overlay),
        ("FVP output", fvp_overlay),
    )
    comparison = Image.new("RGB", (image.width * 3, image.height + 32), "white")
    drawing = ImageDraw.Draw(comparison)
    for index, (label, panel) in enumerate(panels):
        x = index * image.width
        drawing.text((x + 10, 10), label, fill="black")
        comparison.paste(panel, (x, 32))

    Image.fromarray(fvp_mask * 255).save(RESULT_DIR / "fvp_mask.png")
    comparison.save(RESULT_DIR / "fvp_comparison.png")
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps({"fvp_reference_iou": score}, indent=2) + "\n"
    )
    print(f"FVP/reference mask IoU: {score:.4f}")
    print(f"Saved {RESULT_DIR / 'fvp_comparison.png'}")


if __name__ == "__main__":
    main()
