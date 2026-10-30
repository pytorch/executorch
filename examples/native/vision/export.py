# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Export a torchvision classifier and its validation artifacts to PTN."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

import torch
from executorch.exir.native import to_native
from PIL import Image
from torchvision import models  # @manual

logger: logging.Logger = logging.getLogger(__name__)

_MODEL_ALIASES = {"mv2": "mobilenet_v2", "vit": "vit_b_16"}
_WEIGHTS = {
    name: models.get_model_weights(name).DEFAULT
    for name in models.list_models(module=models)
}


class _Expected(TypedDict):
    input_shape: list[int]
    output_shape: list[int]
    argmax: int
    label: str
    top5: list[tuple[int, str, float]]


def _default_out_dir() -> Path:
    return Path.home() / "scratch/models/native_classification"


def _load_model(name: str) -> torch.nn.Module:
    return models.get_model(name, weights=_WEIGHTS[name]).eval()


def _load_image(name: str, image_path: Path) -> tuple[torch.Tensor, list[str]]:
    weights = _WEIGHTS[name]
    with Image.open(image_path) as image:
        image_tensor = weights.transforms()(image.convert("RGB"))
    return image_tensor.unsqueeze(0).contiguous(), list(weights.meta["categories"])


def _expected(
    model: torch.nn.Module, example_input: torch.Tensor, categories: list[str]
) -> tuple[torch.Tensor, _Expected]:
    with torch.no_grad():
        logits = model(example_input)
    flat = logits[0]
    probabilities = torch.softmax(flat, dim=0)
    top5 = torch.topk(flat, 5)
    argmax = int(torch.argmax(flat).item())
    return logits.contiguous(), {
        "input_shape": list(example_input.shape),
        "output_shape": list(logits.shape),
        "argmax": argmax,
        "label": categories[argmax],
        "top5": [
            (int(index), categories[int(index)], float(probabilities[int(index)]))
            for index in top5.indices.tolist()
        ],
    }


def _write_fp32(path: Path, tensor: torch.Tensor) -> None:
    if tensor.dtype is not torch.float32:
        raise ValueError(f"expected float32 tensor, got {tensor.dtype}")
    array = tensor.detach().to("cpu").contiguous().numpy()
    path.write_bytes(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())


def _summary(
    model_name: str,
    paths: dict[str, Path],
    methods: set[str],
    expected: _Expected,
) -> str:
    return "\n".join(
        [
            f"model:     {model_name}",
            f"package:   {paths['ptn']} ({paths['ptn'].stat().st_size} bytes)",
            f"methods:   {', '.join(sorted(methods))}",
            f"input:     {paths['input']} (shape {expected['input_shape']})",
            f"expected:  {paths['expected_bin']} (shape {expected['output_shape']})",
            f"           argmax={expected['argmax']} label={expected['label']!r}",
            "top5:      "
            + ", ".join(
                f"{label}={probability:.3f}"
                for _, label, probability in expected["top5"]
            ),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*_WEIGHTS, *_MODEL_ALIASES], default="mv2")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    output_dir = (args.out_dir or _default_out_dir()) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ptn": output_dir / f"{args.model}.ptn",
        "input": output_dir / f"{args.model}_input.bin",
        "expected_bin": output_dir / f"{args.model}_expected.bin",
        "expected_json": output_dir / f"{args.model}_expected.json",
        "categories": output_dir / "categories.txt",
        "summary": output_dir / f"{args.model}.summary.txt",
    }

    model_name = _MODEL_ALIASES.get(args.model, args.model)
    model = _load_model(model_name)
    example_input, categories = _load_image(model_name, args.image)
    logits, expected = _expected(model, example_input, categories)
    native_program = to_native(torch.export.export(model, (example_input,)))

    native_program.save(str(paths["ptn"]))
    _write_fp32(paths["input"], example_input)
    _write_fp32(paths["expected_bin"], logits)
    paths["expected_json"].write_text(json.dumps(expected, indent=2) + "\n")
    paths["categories"].write_text("\n".join(categories) + "\n")
    summary = _summary(args.model, paths, native_program.methods, expected)
    paths["summary"].write_text(summary + "\n")
    logger.info("wrote %s\n%s", output_dir, summary)


if __name__ == "__main__":
    main()
