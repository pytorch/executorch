# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import math
import subprocess  # nosec B404 - executes the trusted local executor_runner binary
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

STRING_TO_NUMPY_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_image_tensor(
    image_path: str | Path,
    expected_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(image_np).permute(2, 0, 1).contiguous().unsqueeze(0).clone()
    )
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"Image {image_path} produces tensor shape {tuple(tensor.shape)}, "
            f"expected {tuple(expected_shape)}."
        )
    return tensor


def save_tensor_bytes(path: str | Path, tensor: torch.Tensor) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().contiguous().numpy()
    path.write_bytes(array.tobytes())
    return path


def load_tensor_bytes(
    path: str | Path,
    shape: Sequence[int],
    dtype_name: str,
) -> torch.Tensor:
    np_dtype = STRING_TO_NUMPY_DTYPE.get(dtype_name)
    if np_dtype is None:
        raise ValueError(f"Unsupported tensor dtype in metadata: {dtype_name}")

    array = np.fromfile(path, dtype=np_dtype)
    expected_numel = math.prod(shape)
    if array.size != expected_numel:
        raise ValueError(
            f"Tensor file {path} contains {array.size} values, expected {expected_numel}."
        )

    reshaped = array.reshape(tuple(shape)).copy()
    return torch.from_numpy(reshaped)


def save_image_tensor(path: str | Path, tensor: torch.Tensor) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor.detach().cpu()
    if image.dim() == 4:
        if image.shape[0] != 1:
            raise ValueError(
                "Only batch size 1 is supported when writing image output."
            )
        image = image[0]

    if image.dim() != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}.")

    channels = image.shape[0]
    if channels not in {1, 3}:
        raise ValueError(f"Expected 1 or 3 channels, got {channels}.")

    image = image.clamp(0.0, 1.0)
    image_np = (
        image.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).contiguous().numpy()
    )

    if channels == 1:
        pil_image = Image.fromarray(image_np[..., 0], mode="L")
    else:
        pil_image = Image.fromarray(image_np, mode="RGB")
    pil_image.save(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a VGF-exported Swin2SR model with executor_runner."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the exported .pte file.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional metadata JSON path. Defaults to <model>.json.",
    )
    parser.add_argument(
        "--runner",
        default="./cmake-out/executor_runner",
        help="Path to the host executor_runner binary built with VGF support.",
    )
    parser.add_argument(
        "--input-image",
        required=True,
        help="Low-resolution input image.",
    )
    parser.add_argument(
        "--output-image",
        required=True,
        help="High-resolution output image path.",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Optional directory for temporary input/output tensor files.",
    )
    return parser.parse_args()


def resolve_metadata_path(model_path: Path, metadata_path: str | None) -> Path:
    if metadata_path is not None:
        return Path(metadata_path).resolve()
    return model_path.with_suffix(".json")


def run_executor_runner(
    runner: Path,
    model_path: Path,
    input_file: Path,
    output_base: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(runner),
        "--model_path",
        str(model_path),
        "--inputs",
        str(input_file),
        "--output_file",
        str(output_base),
        "--print_output",
        "none",
    ]
    return subprocess.run(  # nosec B603 - command list is assembled without a shell
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    metadata_path = resolve_metadata_path(model_path, args.metadata_path)
    runner_path = Path(args.runner).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not runner_path.is_file():
        raise FileNotFoundError(f"executor_runner not found: {runner_path}")

    metadata = read_json(metadata_path)
    if metadata.get("num_outputs") != 1:
        raise ValueError(
            "The runtime helper currently supports single-output models only."
        )

    if args.working_dir is None:
        with tempfile.TemporaryDirectory(prefix="executorch-sr-vgf-") as tmp_dir:
            workdir = Path(tmp_dir)
            run_once(
                model_path=model_path,
                metadata=metadata,
                runner_path=runner_path,
                input_image=Path(args.input_image),
                output_image=Path(args.output_image),
                working_dir=workdir,
            )
    else:
        workdir = Path(args.working_dir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        run_once(
            model_path=model_path,
            metadata=metadata,
            runner_path=runner_path,
            input_image=Path(args.input_image),
            output_image=Path(args.output_image),
            working_dir=workdir,
        )


def run_once(
    model_path: Path,
    metadata: dict,
    runner_path: Path,
    input_image: Path,
    output_image: Path,
    working_dir: Path,
) -> None:
    input_tensor = load_image_tensor(input_image, metadata["input_shape"])
    input_path = save_tensor_bytes(working_dir / "input0.bin", input_tensor)
    output_base = working_dir / "output"

    result = run_executor_runner(runner_path, model_path, input_path, output_base)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "executor_runner failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    output_tensor = load_tensor_bytes(
        output_base.with_name(f"{output_base.name}-0.bin"),
        metadata["output_shape"],
        metadata["output_dtype"],
    )
    save_image_tensor(output_image, output_tensor)
    print(f"Saved super-resolved image to {output_image.resolve()}")


if __name__ == "__main__":
    main()
