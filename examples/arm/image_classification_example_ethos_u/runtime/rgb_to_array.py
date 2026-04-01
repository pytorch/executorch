# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image


def resize_and_crop_center(img: Image.Image, target_size):
    """Resize keeping aspect ratio, then crop center to (width, height)"""
    target_w, target_h = target_size
    # Compute scaling factor to preserve aspect ratio
    scale_ratio = max(target_h, target_w) / min(img.height, img.width)
    resized_w = int(img.width * scale_ratio)
    resized_h = int(img.height * scale_ratio)
    # Resize the image
    img = img.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)
    # Calculate crop box (center crop)
    left = (resized_w - target_w) / 2
    top = (resized_h - target_h) / 2
    right = (resized_w + target_w) / 2
    bottom = (resized_h + target_h) / 2
    # Keep the center of the image, crop everything outside of the image
    return img.crop((left, top, right, bottom))


def convert_image_to_c_array(
    image_path, output_path, image_size=(224, 224), array_name="image_data"
):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = resize_and_crop_center(img, image_size)
    # NumPy arrays are stored in channels-last format. Convert to channels-first.
    img_channels_first = np.transpose(img, (2, 0, 1))
    data = np.array(img_channels_first, dtype=np.float32) / 255.0
    data = data.flatten()
    # Format as C array
    array_lines = []
    for i in range(0, len(data), 20):  # 20 values per line
        line = ", ".join(f"{val:6f}" for val in data[i : i + 20])
        array_lines.append("    " + line + ",")
    c_array_str = f"""#include <stdint.h>
const float {array_name}[{len(data)}] = {{
{os.linesep.join(array_lines)}
}};
"""
    # Write to output file
    with open(output_path, "w") as f:
        f.write(c_array_str)
    print(f" Converted '{image_path}' → '{output_path}' ({len(data)} bytes)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image", required=True, help="Path to an RGB image (e.g. a jpg file)"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for the generated C array"
    )
    parser.add_argument(
        "--resolution",
        required=False,
        type=int,
        nargs=2,
        default=(224, 224),
        help="Resolution of the output image",
    )
    args = parser.parse_args()
    image_path = args.image
    output_path = args.output
    image_size = args.resolution
    convert_image_to_c_array(image_path, output_path, image_size)
