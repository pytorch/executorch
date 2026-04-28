# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone QNN repro for the variance-channel materialization issue.

This script isolates two mathematically equivalent toy models:

1. broken: torch.ones(...) * scalar -> cat -> conv
2. fixed:  scalar.reshape(...).expand(...) -> cat -> conv

Both variants use the same randomly initialized conv weights for a given seed.
The script:
  - computes eager vs quantized SQNR
  - lowers each variant to a QNN-backed `.pte`
  - saves inputs and reference outputs
  - optionally runs the local ExecuTorch portable runtime on x86 if available

Requirements:
  - a Python environment with `torch`, `torchao`, and `executorch`
  - `QNN_SDK_ROOT` pointing to a local QNN SDK
  - optional: built ExecuTorch portable bindings for `--run-runtime`

Example:
  export QNN_SDK_ROOT=/path/to/qnn-2.37
  python3 qnn_variance_repro.py \
      --height 64 \
      --width 64 \
      --quant-dtype 8a8w \
      --variance 1.0 \
      --output-dir /tmp/qcom_variance_repro
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
from typing import Iterable

import executorch.backends.qualcomm  # noqa: F401
import torch
import torch.nn as nn
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from torch.export import export
from torch.utils._pytree import tree_flatten
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

try:
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
except ImportError:
    _load_for_executorch = None


UPSCALE = 18.0


class OnesMulCatConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(17, 64, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        avg_variance = variance * UPSCALE
        b, _, h, w = x.shape
        variance_tensor = (
            torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype) * avg_variance
        )
        return self.conv(torch.cat((x, variance_tensor), dim=1))


class ExpandCatConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(17, 64, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        avg_variance = variance * UPSCALE
        b, _, h, w = x.shape
        variance_tensor = avg_variance.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        return self.conv(torch.cat((x, variance_tensor), dim=1))


VARIANT_FACTORIES = {
    "ones_mul_cat_conv": OnesMulCatConv,
    "expand_cat_conv": ExpandCatConv,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("both", *VARIANT_FACTORIES.keys()),
        default="both",
        help="Which toy model to run.",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument(
        "--calibration-runs",
        type=int,
        default=3,
        help="Number of random calibration passes per model.",
    )
    parser.add_argument(
        "--calibration-variance",
        type=float,
        default=0.5,
        help="Scalar variance value used during calibration.",
    )
    parser.add_argument(
        "--quant-dtype",
        choices=("8a8w", "16a8w"),
        default="16a8w",
    )
    parser.add_argument(
        "--soc-model",
        default="SM8750",
        help="Name of the QcomChipset enum member to use, e.g. SM8650 or SM8750.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/qcom_variance_repro"),
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Export `.pte` and references only; do not run the local portable runtime.",
    )
    return parser.parse_args()


def ensure_qnn_env() -> None:
    sdk_root = os.environ.get("QNN_SDK_ROOT")
    if not sdk_root:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set. Importing `executorch.backends.qualcomm` "
            "should auto-stage the SDK on supported x86 hosts; otherwise set "
            "QNN_SDK_ROOT manually before running."
        )

    lib_dir = Path(sdk_root) / "lib" / "x86_64-linux-clang"
    if not lib_dir.is_dir():
        raise RuntimeError(f"QNN x86 lib directory not found: {lib_dir}")

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = [path for path in current.split(":") if path]
    if str(lib_dir) not in current_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([str(lib_dir), *current_paths])

    # Best-effort preload for in-process x86 runtime execution.
    # Setting LD_LIBRARY_PATH after Python starts is not sufficient for all
    # later dlopen-by-name paths, so preload the QNN runtime libraries directly.
    for lib_name in ("libQnnSystem.so", "libQnnHtp.so"):
        lib_path = lib_dir / lib_name
        if lib_path.is_file():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def sqnr_db(reference: torch.Tensor, other: torch.Tensor) -> float:
    reference = reference.detach().float()
    other = other.detach().float()
    noise = (reference - other).pow(2).sum()
    if noise == 0:
        return math.inf
    signal = reference.pow(2).sum()
    if signal == 0:
        return -math.inf
    return 10.0 * torch.log10(signal / noise).item()


def max_abs_diff(reference: torch.Tensor, other: torch.Tensor) -> float:
    return (reference.detach().float() - other.detach().float()).abs().max().item()


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def save_raw(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor.detach().cpu().to(torch.float32).contiguous().numpy().tofile(path)


def build_model(variant: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return VARIANT_FACTORIES[variant]().eval()


def build_inputs(
    height: int, width: int, seed: int, variance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(1, 16, height, width)
    variance_tensor = torch.tensor([variance], dtype=torch.float32)
    return x, variance_tensor


def calibrate(
    prepared: nn.Module,
    reference_input: torch.Tensor,
    calibration_runs: int,
    calibration_variance: float,
    seed: int,
) -> None:
    variance_tensor = torch.tensor([calibration_variance], dtype=torch.float32)
    with torch.no_grad():
        for index in range(calibration_runs):
            torch.manual_seed(seed + 1000 + index)
            prepared(torch.randn_like(reference_input), variance_tensor)


def get_soc_model(name: str) -> QcomChipset:
    try:
        return getattr(QcomChipset, name)
    except AttributeError as exc:
        valid = sorted(item.name for item in QcomChipset)
        raise ValueError(
            f"Unknown QcomChipset '{name}'. Valid values: {valid}"
        ) from exc


def get_quant_dtype(name: str) -> QuantDtype:
    return QuantDtype.use_8a8w if name == "8a8w" else QuantDtype.use_16a8w


def run_portable_runtime(
    pte_path: Path,
    sample_inputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if _load_for_executorch is None:
        raise RuntimeError(
            "executorch portable runtime bindings are not available. "
            "Build/install `executorch.extension.pybindings.portable_lib` "
            "or rerun with --skip-runtime."
        )

    exec_mod = _load_for_executorch(str(pte_path))
    flat_inputs, _ = tree_flatten(sample_inputs)
    outputs = exec_mod.forward(flat_inputs)
    if not outputs:
        raise RuntimeError("Portable runtime returned no outputs")
    output = outputs[0]
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)
    return output


def variant_names(requested: str) -> Iterable[str]:
    if requested == "both":
        return VARIANT_FACTORIES.keys()
    return (requested,)


def run_variant(
    variant: str,
    args: argparse.Namespace,
    x: torch.Tensor,
    variance: torch.Tensor,
    soc_model: QcomChipset,
    quant_dtype: QuantDtype,
) -> dict[str, object]:
    out_dir = args.output_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(variant, args.seed)
    with torch.no_grad():
        eager_out = model(x, variance)

    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=soc_model,
    )
    quantizer.set_default_quant_config(
        quant_dtype=quant_dtype,
        is_qat=False,
        is_conv_per_channel=True,
        is_linear_per_channel=False,
    )

    exported = export(model, (x, variance), strict=False).module()
    prepared = prepare_pt2e(exported, quantizer)
    calibrate(
        prepared,
        x,
        args.calibration_runs,
        args.calibration_variance,
        args.seed,
    )
    quantized = convert_pt2e(prepared)

    with torch.no_grad():
        quantized_out = quantized(x, variance)

    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_model,
        backend_options=backend_options,
    )
    edge_program = to_edge_transform_and_lower_to_qnn(
        module=quantized,
        inputs=(x, variance),
        compiler_specs=compiler_specs,
    )
    executorch_program = edge_program.to_executorch()

    pte_path = out_dir / "model.pte"
    with open(pte_path, "wb") as file:
        executorch_program.write_to_file(file)

    save_tensor(out_dir / "input_0.pt", x)
    save_tensor(out_dir / "input_1.pt", variance)
    save_tensor(out_dir / "eager_out.pt", eager_out)
    save_tensor(out_dir / "quantized_out.pt", quantized_out)
    save_raw(out_dir / "input_0.raw", x)
    save_raw(out_dir / "input_1.raw", variance)
    save_raw(out_dir / "quantized_out.raw", quantized_out)

    result: dict[str, object] = {
        "variant": variant,
        "pte_path": str(pte_path),
        "eager_vs_quant_sqnr_db": sqnr_db(eager_out, quantized_out),
        "eager_vs_quant_max_abs_diff": max_abs_diff(eager_out, quantized_out),
    }

    if not args.skip_runtime:
        runtime_out = run_portable_runtime(pte_path, (x, variance))
        save_tensor(out_dir / "runtime_out.pt", runtime_out)
        save_raw(out_dir / "runtime_out.raw", runtime_out)
        result["quant_vs_runtime_sqnr_db"] = sqnr_db(quantized_out, runtime_out)
        result["quant_vs_runtime_max_abs_diff"] = max_abs_diff(
            quantized_out, runtime_out
        )

    return result


def main() -> None:
    args = parse_args()
    ensure_qnn_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, variance = build_inputs(args.height, args.width, args.seed, args.variance)
    soc_model = get_soc_model(args.soc_model)
    quant_dtype = get_quant_dtype(args.quant_dtype)

    results = []
    for variant in variant_names(args.variant):
        print(f"Running variant: {variant}")
        result = run_variant(variant, args, x, variance, soc_model, quant_dtype)
        print(json.dumps(result, indent=2))
        results.append(result)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
