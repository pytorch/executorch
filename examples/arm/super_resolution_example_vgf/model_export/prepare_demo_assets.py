# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

RUNTIME_DEMO_CROP = ("demo", 60, 420)
CALIBRATION_CROPS = (
    ("calib_0", 60, 280),
    ("calib_1", 60, 420),
    ("calib_2", 60, 560),
    ("calib_3", 60, 700),
)
EVAL_CROPS = (
    ("eval_0", 60, 420),
    ("eval_1", 60, 700),
)
DEFAULT_SOURCE_IMAGE = (
    Path(__file__).resolve().parents[4] / "docs/source/_static/img/ios_demo_app.jpg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible Swin2SR demo dataset from a repo-local image."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where calibration, evaluation, and runtime demo assets are written.",
    )
    parser.add_argument(
        "--source-image",
        default=str(DEFAULT_SOURCE_IMAGE),
        help="Source image used for the fixed text-heavy crops.",
    )
    parser.add_argument(
        "--hr-size",
        type=int,
        default=128,
        help="High-resolution crop size.",
    )
    parser.add_argument(
        "--lr-size",
        type=int,
        default=64,
        help="Low-resolution size written for runtime/calibration inputs.",
    )
    return parser.parse_args()


def downsample_to_lr(image: Image.Image, lr_size: int) -> Image.Image:
    return image.resize((lr_size, lr_size), Image.Resampling.BICUBIC)


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def crop_square(image: Image.Image, left: int, top: int, size: int) -> Image.Image:
    right = left + size
    bottom = top + size
    if right > image.width or bottom > image.height:
        raise ValueError(
            f"Crop {(left, top, size)} exceeds source image bounds {image.size}."
        )
    return image.crop((left, top, right, bottom))


def write_metadata(
    output_dir: Path,
    source_image: Path,
    hr_size: int,
    lr_size: int,
) -> None:
    metadata = {
        "source_image": str(source_image),
        "hr_size": hr_size,
        "lr_size": lr_size,
        "runtime_demo_crop": {
            "name": RUNTIME_DEMO_CROP[0],
            "left": RUNTIME_DEMO_CROP[1],
            "top": RUNTIME_DEMO_CROP[2],
        },
        "calibration_crops": [
            {"name": name, "left": left, "top": top}
            for name, left, top in CALIBRATION_CROPS
        ],
        "eval_crops": [
            {"name": name, "left": left, "top": top} for name, left, top in EVAL_CROPS
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.lr_size <= 0 or args.hr_size <= 0:
        raise ValueError("--lr-size and --hr-size must be positive.")
    if args.hr_size != args.lr_size * 2:
        raise ValueError("This demo helper expects x2 super-resolution sizes.")

    output_dir = Path(args.output_dir).resolve()
    source_image = Path(args.source_image).resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    image = Image.open(source_image).convert("RGB")

    for name, left, top in CALIBRATION_CROPS:
        hr_crop = crop_square(image, left, top, args.hr_size)
        lr_crop = downsample_to_lr(hr_crop, args.lr_size)
        save_image(output_dir / "calibration/hr" / f"{name}.png", hr_crop)
        save_image(output_dir / "calibration/lr" / f"{name}.png", lr_crop)

    for name, left, top in EVAL_CROPS:
        hr_crop = crop_square(image, left, top, args.hr_size)
        lr_crop = downsample_to_lr(hr_crop, args.lr_size)
        save_image(output_dir / "eval/hr" / f"{name}.png", hr_crop)
        save_image(output_dir / "eval/lr" / f"{name}.png", lr_crop)

    demo_name, demo_left, demo_top = RUNTIME_DEMO_CROP
    demo_hr = crop_square(image, demo_left, demo_top, args.hr_size)
    demo_lr = downsample_to_lr(demo_hr, args.lr_size)
    save_image(output_dir / "runtime" / f"{demo_name}_hr_{args.hr_size}.png", demo_hr)
    save_image(output_dir / "runtime" / f"{demo_name}_lr_{args.lr_size}.png", demo_lr)

    write_metadata(output_dir, source_image, args.hr_size, args.lr_size)

    print(f"Prepared demo assets under {output_dir}")
    print(
        f"Runtime input: {output_dir / 'runtime' / f'{demo_name}_lr_{args.lr_size}.png'}"
    )
    print(
        f"Runtime reference: {output_dir / 'runtime' / f'{demo_name}_hr_{args.hr_size}.png'}"
    )


if __name__ == "__main__":
    main()
