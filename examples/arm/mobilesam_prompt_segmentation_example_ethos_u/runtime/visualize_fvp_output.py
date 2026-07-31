# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


RLE_PREFIX = "Segmentation mask RLE chunk "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize and validate a MobileSAM mask dumped by the FVP."
    )
    parser.add_argument("--fvp-log", required=True, type=Path)
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--minimum-iou", type=float)
    return parser.parse_args()


def parse_rle_mask(log_path: Path, expected_pixels: int) -> list[int]:
    mask: list[int] = []
    in_dump = False
    for line in log_path.read_text().splitlines():
        if "executorch:main.cpp:" in line:
            continue
        if "Segmentation mask RLE begin" in line:
            in_dump = True
            continue
        if "Segmentation mask RLE end" in line:
            break
        if not in_dump or RLE_PREFIX not in line:
            continue
        payload = line.split(RLE_PREFIX, maxsplit=1)[1]
        for value, count in re.findall(r"([01]):([0-9]+),", payload):
            mask.extend([int(value)] * int(count))

    if len(mask) != expected_pixels:
        raise ValueError(
            f"FVP RLE contains {len(mask)} pixels; expected {expected_pixels}."
        )
    return mask


def prepare_input_image(image_path: Path, input_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = input_size / max(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.BILINEAR,
    )
    padded = Image.new("RGB", (input_size, input_size))
    padded.paste(resized, (0, 0))
    return padded


def create_overlay(
    image: Image.Image, mask: Image.Image, color: tuple[int, int, int]
) -> Image.Image:
    resized_mask = mask.resize(image.size, Image.Resampling.NEAREST)
    color_layer = Image.new("RGB", image.size, color)
    blended = Image.blend(image, color_layer, 0.45)
    overlay = image.copy()
    overlay.paste(blended, mask=resized_mask)
    return overlay


def draw_prompts(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for point_x, point_y in metadata["point_prompts_xy"]:
        radius = max(4, metadata["input_size"] // 80)
        draw.ellipse(
            (
                point_x - radius,
                point_y - radius,
                point_x + radius,
                point_y + radius,
            ),
            fill=(255, 48, 48),
            outline=(255, 255, 255),
            width=2,
        )
    return result


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 30
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    panel.paste(image, (0, title_height))
    ImageDraw.Draw(panel).text((10, 8), title, fill="black")
    return panel


def binary_metrics(mask: list[int], reference: list[int]) -> tuple[float, float]:
    intersection = sum(a == 1 and b == 1 for a, b in zip(mask, reference))
    union = sum(a == 1 or b == 1 for a, b in zip(mask, reference))
    iou = intersection / union if union else 1.0
    agreement = sum(a == b for a, b in zip(mask, reference)) / len(mask)
    return iou, agreement


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    _, _, mask_height, mask_width = metadata["output_shape"]
    mask = parse_rle_mask(args.fvp_log, mask_width * mask_height)
    mask_image = Image.new("L", (mask_width, mask_height))
    mask_image.putdata([value * 255 for value in mask])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_image.save(args.output_dir / "fvp_mask.png")
    input_image = prepare_input_image(args.input_image, metadata["input_size"])
    prompted_input = draw_prompts(input_image, metadata)
    prompted_input.save(args.output_dir / "input_with_prompts.png")
    fvp_overlay = draw_prompts(
        create_overlay(input_image, mask_image, (0, 220, 120)), metadata
    )
    fvp_overlay.save(args.output_dir / "fvp_overlay.png")

    panels = [add_title(prompted_input, "Input and positive prompt")]
    metrics: dict[str, Any] = {
        "background_pixels": mask.count(0),
        "foreground_pixels": mask.count(1),
        "output_mask_size": [mask_width, mask_height],
    }
    if metrics["foreground_pixels"] in (0, len(mask)):
        raise RuntimeError(
            "FVP produced a degenerate mask with "
            f"{metrics['foreground_pixels']} foreground pixels."
        )

    below_minimum_iou = False
    if args.reference_mask is not None:
        reference_image = (
            Image.open(args.reference_mask)
            .convert("L")
            .resize((mask_width, mask_height), Image.Resampling.NEAREST)
        )
        reference = [int(value > 0) for value in reference_image.tobytes()]
        iou, agreement = binary_metrics(mask, reference)
        metrics["fvp_reference_iou"] = iou
        metrics["fvp_reference_pixel_agreement"] = agreement
        reference_overlay = draw_prompts(
            create_overlay(input_image, reference_image, (0, 170, 255)), metadata
        )
        panels.append(add_title(reference_overlay, "Host quantized mask"))
        below_minimum_iou = args.minimum_iou is not None and iou < args.minimum_iou
    elif args.minimum_iou is not None:
        raise ValueError("--minimum-iou requires --reference-mask.")

    panels.append(add_title(fvp_overlay, "FVP mask"))
    comparison = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    offset = 0
    for panel in panels:
        comparison.paste(panel, (offset, 0))
        offset += panel.width
    comparison.save(args.output_dir / "fvp_comparison.png")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"FVP mask: {metrics['foreground_pixels']} foreground pixels, "
        f"artifacts saved to {args.output_dir}"
    )
    if "fvp_reference_iou" in metrics:
        print(
            f"FVP/reference IoU={metrics['fvp_reference_iou']:.4f} "
            f"agreement={metrics['fvp_reference_pixel_agreement']:.4f}"
        )
    if below_minimum_iou:
        raise RuntimeError(
            f"FVP/reference IoU {metrics['fvp_reference_iou']:.4f} is below "
            f"{args.minimum_iou:.4f}."
        )


if __name__ == "__main__":
    main()
