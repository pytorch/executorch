# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Utility functions for image processing. Run it with your image:

# python image_util.py --image-path <path_to_image>

import logging
from argparse import ArgumentParser

import torch
import torchvision
from PIL import Image
from torch import nn


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def prepare_image(image: Image, target_h: int, target_w: int) -> torch.Tensor:
    """Read image into a tensor and resize the image so that it fits in
    a target_h x target_w canvas.

    Args:
        image (Image): An Image object.
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        torch.Tensor: resized image tensor.
    """
    img = torchvision.transforms.functional.pil_to_tensor(image)
    # height ratio
    ratio_h = img.shape[1] / target_h
    # width ratio
    ratio_w = img.shape[2] / target_w
    # resize the image so that it fits in a target_h x target_w canvas
    ratio = max(ratio_h, ratio_w)
    output_size = (int(img.shape[1] / ratio), int(img.shape[2] / ratio))
    img = torchvision.transforms.Resize(size=output_size)(img)
    return img


def serialize_image(image: torch.Tensor, path: str) -> None:
    copy = torch.tensor(image)
    m = nn.Module()
    par = nn.Parameter(copy, requires_grad=False)
    m.register_parameter("0", par)
    tensors = torch.jit.script(m)
    tensors.save(path)

    logging.info(f"Saved image tensor to {path}")


def image_to_csv(image: torch.Tensor, path: str) -> None:
    im = torch.tensor(image)

    # List of restrictions on the image tensor to ensure our naive csv converter works
    assert im.dim() == 3, "Image must be 3D"
    assert im.shape[0] == 3, "Image must have 3 channels"
    assert im.dtype == torch.uint8, "Image data must be uint8"
    assert im.device == torch.device("cpu"), "Image data must be on CPU"
    assert im.is_contiguous(), "Image data must be contiguous"
    assert im.dim_order() == (0, 1, 2), "Image data must be in CHW format"

    # write header
    with open(path, "w") as f:
        # header
        # dims, shape[...], sizeof(dtype)
        f.write(f"{im.dim()},")
        for i in range(im.dim()):
            f.write(f"{im.shape[i]},")
        f.write("1\n")  # sizeof(uint8_t), add a newline to end header

    # append data as bytes
    with open(path, "ab") as f:
        # data as bytes
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                for k in range(im.shape[2]):
                    b = int(im[i][j][k]).to_bytes(1, byteorder="little")
                    f.write(b)

    logging.info(f"Saved image csv to {path}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the image.",
    )
    parser.add_argument(
        "--output-path",
        default="image.pt",
    )
    parser.add_argument(
        "--output-csv",
        required=False,
    )
    args = parser.parse_args()

    image = Image.open(args.image_path)
    image_tensor = prepare_image(image, target_h=336, target_w=336)
    serialize_image(image_tensor, args.output_path)

    if args.output_csv:
        image_to_csv(image_tensor, args.output_csv)


if __name__ == "__main__":
    main()
