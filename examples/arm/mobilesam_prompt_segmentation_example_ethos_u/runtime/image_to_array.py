# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def convert_image_to_c_array(
    image_path: str,
    output_path: str,
    image_size: tuple[int, int],
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
    array_name: str = "image_data",
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    target_width, target_height = image_size
    if target_width != target_height:
        raise ValueError("MobileSAM runtime preprocessing expects a square input.")

    scale = target_width / max(height, width)
    resized_size = (round(width * scale), round(height * scale))
    image = image.resize(resized_size, resample=Image.Resampling.BILINEAR)

    data = np.asarray(image, dtype=np.float32)
    data = (data - np.asarray(pixel_mean, dtype=np.float32)) / np.asarray(
        pixel_std, dtype=np.float32
    )
    padded_data = np.zeros((target_height, target_width, 3), dtype=np.float32)
    padded_data[: resized_size[1], : resized_size[0], :] = data
    data = np.transpose(padded_data, (2, 0, 1)).flatten()

    array_lines = []
    for i in range(0, len(data), 12):
        line = ", ".join(f"{value:.8f}" for value in data[i : i + 12])
        array_lines.append("    " + line + ",")

    c_array = f"""#include <stdint.h>
#include <stddef.h>

const size_t image_width = {image_size[0]};
const size_t image_height = {image_size[1]};
const size_t image_channels = 3;
__attribute__((section("input_data_sec"), aligned(16))) float {array_name}[{len(data)}] = {{
{os.linesep.join(array_lines)}
}};
"""
    with open(output_path, "w") as output_file:
        output_file.write(c_array)
    print(f"Converted '{image_path}' to '{output_path}' ({len(data)} floats)")


def load_model_metadata(
    metadata_path: str,
) -> tuple[tuple[int, int], tuple[float, float, float], tuple[float, float, float]]:
    metadata = json.loads(Path(metadata_path).read_text())
    input_shape = metadata.get("input_shape")
    if not isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError("Model metadata must contain a four-dimensional input_shape.")
    if input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError("MobileSAM runtime expects input shape [1, 3, H, W].")

    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Model metadata does not contain preprocessing values.")
    pixel_mean_values = preprocessing.get("pixel_mean")
    pixel_std_values = preprocessing.get("pixel_std")
    if not isinstance(pixel_mean_values, list) or len(pixel_mean_values) != 3:
        raise ValueError("MobileSAM pixel_mean must contain three RGB values.")
    if not isinstance(pixel_std_values, list) or len(pixel_std_values) != 3:
        raise ValueError("MobileSAM pixel_std must contain three RGB values.")

    image_size = (int(input_shape[3]), int(input_shape[2]))
    pixel_mean = tuple(float(value) for value in pixel_mean_values)
    pixel_std = tuple(float(value) for value in pixel_std_values)
    return (
        image_size,
        (pixel_mean[0], pixel_mean[1], pixel_mean[2]),
        (pixel_std[0], pixel_std[1], pixel_std[2]),
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an RGB image.")
    parser.add_argument(
        "--output", required=True, help="Output path for the generated C array."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Exporter metadata containing input shape and preprocessing values.",
    )
    args = parser.parse_args()

    image_size, pixel_mean, pixel_std = load_model_metadata(args.metadata)
    convert_image_to_c_array(
        args.image,
        args.output,
        image_size,
        pixel_mean,
        pixel_std,
    )


if __name__ == "__main__":
    main()
