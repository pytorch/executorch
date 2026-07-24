#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure Practical-RIFE PTQ and QAT accuracy on real frame triples.

The script is intended as a small handover flow for comparing eager FP32,
PTQ, and QAT PT2E models before generating VGF artifacts. The expected
triple order is:

* first frame: model input 0
* third frame: model input 1
* middle frame: quality target

By default the fixed input shape is 768x384.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.export import export
from torchao.quantization.pt2e import (
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

EXECUTORCH_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXECUTORCH_ROOT))

from executorch.backends.arm.quantizer import (  # noqa: E402
    get_symmetric_quantization_config,
    get_uint8_io_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.extension.export_util.utils import save_pte_program  # noqa: E402

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 384
RIFE_SHAPE_ALIGNMENT = 64
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class FrameTriple:
    name: str
    input0: Path | None
    input1: Path | None
    target: Path | None
    synthetic: bool = False


@dataclass(frozen=True)
class QuantizedModel:
    model: torch.nn.Module
    coverage: dict[str, Any]


class RIFEWrapper(torch.nn.Module):
    def __init__(self, flownet: torch.nn.Module) -> None:
        super().__init__()
        self.flownet = flownet

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        imgs = torch.cat((img0, img1), dim=1)
        _, _, merged = self.flownet(imgs, 0.5, [16, 8, 4, 2, 1])
        return merged[-1]


def validate_input_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("--height and --width must be positive.")
    if height % RIFE_SHAPE_ALIGNMENT != 0 or width % RIFE_SHAPE_ALIGNMENT != 0:
        raise ValueError(
            "RIFE inputs must be multiples of "
            f"{RIFE_SHAPE_ALIGNMENT}. Got height={height}, width={width}."
        )


def load_rife_model(model_root: Path, checkpoint: Path | None) -> torch.nn.Module:
    if not model_root.exists():
        raise FileNotFoundError(f"Practical-RIFE repo not found: {model_root}")
    checkpoint = checkpoint or model_root / "train_log" / "flownet.pkl"
    sys.path.insert(0, str(model_root))
    ifnet_module = importlib.import_module("train_log.IFNet_HDv3")
    ifnet = cast(Any, ifnet_module)

    if not checkpoint.exists():
        raise FileNotFoundError(
            "RIFE checkpoint was not found. Pass --checkpoint or place "
            f"flownet.pkl at {checkpoint}."
        )

    checkpoint_data = torch.load(
        checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(checkpoint_data, dict):
        raise TypeError(f"Unexpected RIFE checkpoint type: {type(checkpoint_data)}")

    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint_data.items()
    }
    if hasattr(ifnet, "backwarp_tenGrid"):
        ifnet.backwarp_tenGrid.clear()
    flownet = ifnet.IFNet()
    incompatible_keys = flownet.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys:
        raise RuntimeError(
            f"Missing RIFE checkpoint keys: {incompatible_keys.missing_keys}"
        )
    return RIFEWrapper(flownet).eval()


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def split_triple_line(line: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", line.strip()) if part]


def triple_list_has_header(first_line: str) -> bool:
    fields = {part.lower() for part in split_triple_line(first_line)}
    return {"input0", "input1", "target"}.issubset(fields)


def load_triples_list(path: Path) -> list[FrameTriple]:
    triples: list[FrameTriple] = []
    base_dir = path.parent
    with path.open(encoding="utf-8", newline="") as file:
        first_line = file.readline()
        file.seek(0)
        if triple_list_has_header(first_line):
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                if row is None:
                    continue
                triples.append(
                    FrameTriple(
                        name=row.get("name") or f"triple_{index:05d}",
                        input0=resolve_path(row["input0"], base_dir),
                        input1=resolve_path(row["input1"], base_dir),
                        target=resolve_path(row["target"], base_dir),
                    )
                )
            return triples

        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = split_triple_line(line)
            if len(parts) not in (3, 4):
                raise ValueError(
                    "Triple lines must contain 'input0 input1 target' or "
                    "'name input0 input1 target'. "
                    f"Bad line {line_number}: {line.rstrip()}"
                )
            if len(parts) == 3:
                name = f"triple_{len(triples):05d}"
                input0, input1, target = parts
            else:
                name, input0, input1, target = parts
            triples.append(
                FrameTriple(
                    name=name,
                    input0=resolve_path(input0, base_dir),
                    input1=resolve_path(input1, base_dir),
                    target=resolve_path(target, base_dir),
                )
            )
    return triples


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_triples(root: Path, stride: int) -> list[FrameTriple]:
    if stride <= 0:
        raise ValueError("--sequence-stride must be positive.")

    triples: list[FrameTriple] = []
    directories = [root] + sorted(path for path in root.rglob("*") if path.is_dir())
    for directory in directories:
        images = image_files(directory)
        if len(images) < 3:
            continue
        relative = directory.relative_to(root) if directory != root else Path("root")
        prefix = "__".join(relative.parts)
        for index in range(0, len(images) - 2, stride):
            triples.append(
                FrameTriple(
                    name=f"{prefix}_{index:05d}",
                    input0=images[index],
                    input1=images[index + 2],
                    target=images[index + 1],
                )
            )
    return triples


def make_random_triples(count: int) -> list[FrameTriple]:
    return [
        FrameTriple(
            name=f"random_{index:05d}",
            input0=None,
            input1=None,
            target=None,
            synthetic=True,
        )
        for index in range(count)
    ]


def validate_triples(triples: Iterable[FrameTriple]) -> list[FrameTriple]:
    validated = list(triples)
    if not validated:
        raise ValueError("No frame triples were found.")
    missing = [
        str(path)
        for triple in validated
        for path in (triple.input0, triple.input1, triple.target)
        if path is not None and not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing frame files:\n" + "\n".join(missing[:20]))
    return validated


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def preprocess_image(
    image: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    current_height, current_width = image.shape[-2:]
    if current_height == height and current_width == width:
        return image

    if mode == "center-crop":
        if current_height < height or current_width < width:
            raise ValueError(
                "Image is smaller than the requested crop. Use "
                "--preprocess resize or provide larger frames."
            )
        top = (current_height - height) // 2
        left = (current_width - width) // 2
        return image[:, top : top + height, left : left + width].contiguous()

    if mode == "resize":
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).contiguous()

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def load_triple_tensors(
    triple: FrameTriple,
    *,
    height: int,
    width: int,
    preprocess: str,
    random_seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if triple.synthetic:
        index_text = triple.name.rsplit("_", maxsplit=1)[-1]
        generator = torch.Generator().manual_seed(random_seed + int(index_text))
        input0 = torch.rand((1, 3, height, width), generator=generator)
        input1 = torch.rand((1, 3, height, width), generator=generator)
        target = torch.rand((1, 3, height, width), generator=generator)
        return (input0, input1), target

    if triple.input0 is None or triple.input1 is None or triple.target is None:
        raise ValueError(f"Frame triple is missing paths: {triple.name}")

    frames = [
        preprocess_image(
            load_image(path),
            height=height,
            width=width,
            mode=preprocess,
        )
        for path in (triple.input0, triple.input1, triple.target)
    ]
    input0, input1, target = [frame.unsqueeze(0) for frame in frames]
    return (input0, input1), target


def gaussian_window(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    sigma = max(window_size / 6.0, 1e-3)

    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = prediction.shape[1]
    kernel = gaussian_window(
        channels,
        prediction.device,
        prediction.dtype,
        prediction.shape[-2],
        prediction.shape[-1],
    )
    padding = kernel.shape[-1] // 2
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = (
        F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_xy
    )

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def tensor_metrics(
    prediction: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.detach().cpu().to(torch.float64)
    reference = reference.detach().cpu().to(torch.float64)
    error = prediction - reference
    abs_error = error.abs()
    mse = error.pow(2).mean().item()
    reference_norm = reference.norm().item()
    error_norm = error.norm().item()
    prediction_norm = prediction.norm().item()

    if reference_norm > 0.0 and error_norm > 0.0:
        sqnr = 20.0 * math.log10(reference_norm / error_norm)
    elif error_norm == 0.0:
        sqnr = float("inf")
    else:
        sqnr = float("-inf")

    if reference_norm > 0.0 and prediction_norm > 0.0:
        cosine = torch.dot(reference.flatten(), prediction.flatten()).item() / (
            reference_norm * prediction_norm
        )
    else:
        cosine = float("nan")

    return {
        "mae": float(abs_error.mean().item()),
        "max_abs": float(abs_error.max().item()),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "sqnr": float(sqnr),
        "cosine": float(cosine),
    }


def quality_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    clamped = prediction.detach().cpu().clamp(0.0, 1.0)
    target = target.detach().cpu()
    metrics = tensor_metrics(prediction, target)
    metrics.update(
        {
            "psnr": float(batch_psnr(clamped, target).mean().item()),
            "ssim": float(batch_ssim(clamped, target).mean().item()),
            "output_min": float(prediction.detach().cpu().min().item()),
            "output_max": float(prediction.detach().cpu().max().item()),
        }
    )
    return metrics


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def aggregate_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    aggregate = {}
    for metric in (
        "mae",
        "max_abs",
        "mse",
        "rmse",
        "sqnr",
        "cosine",
        "psnr",
        "ssim",
        "output_min",
        "output_max",
    ):
        values = [
            float(row[f"{prefix}_{metric}"])
            for row in rows
            if f"{prefix}_{metric}" in row and row[f"{prefix}_{metric}"] is not None
        ]
        if values:
            aggregate[f"{prefix}_{metric}_mean"] = mean(values)
            aggregate[f"{prefix}_{metric}_min"] = float(min(values))
            aggregate[f"{prefix}_{metric}_max"] = float(max(values))
    return aggregate


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def run_model(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        output = model(*inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)}")
    return output


def target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if name:
        return str(name)
    return str(target)


def node_arg_names(node: torch.fx.Node) -> list[str]:
    names: list[str] = []

    def collect(value: Any) -> Any:
        if isinstance(value, torch.fx.Node):
            names.append(value.name)
        return value

    torch.fx.node.map_arg((node.args, node.kwargs), collect)
    return names


def is_quantize_node(name: str) -> bool:
    return name.startswith(("quantize_per_tensor", "quantize_per_channel"))


def is_dequantize_node(name: str) -> bool:
    return name.startswith(("dequantize_per_tensor", "dequantize_per_channel"))


def is_quantized_op_name(name: str) -> bool:
    return (
        "quantized_decomposed" in name
        or name.startswith("quantized.")
        or ".quantized_" in name
    )


def quantization_coverage(graph_module: torch.nn.Module) -> dict[str, Any]:
    graph = getattr(graph_module, "graph", None)
    if graph is None:
        return {
            "status": "unavailable",
            "reason": f"Model has no FX graph: {type(graph_module)}",
        }

    quantized_value_names: set[str] = set()
    rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    op_counter: Counter[tuple[str, str]] = Counter()

    for node in graph.nodes:
        name = target_name(node.target)
        input_names = node_arg_names(node)
        quantized_inputs = [
            input_name
            for input_name in input_names
            if input_name in quantized_value_names
        ]
        status = "not_compute"
        output_quantized = False

        if node.op == "placeholder":
            status = "graph_input_fp32"
        elif node.op == "get_attr":
            status = "parameter_or_buffer"
        elif node.op == "output":
            status = "graph_output"
        elif is_quantize_node(name):
            status = "quantize_boundary"
            output_quantized = True
        elif is_dequantize_node(name):
            status = "dequantize_boundary"
        elif is_quantized_op_name(name):
            status = "quantized_op"
            output_quantized = True
        elif quantized_inputs:
            status = "quantized_context_op"
            output_quantized = True
        elif node.op == "call_function":
            status = "fp32_or_fallback_op"

        if output_quantized:
            quantized_value_names.add(node.name)

        rows.append(
            {
                "index": len(rows),
                "name": node.name,
                "op": node.op,
                "target": name,
                "status": status,
                "input_nodes": input_names,
                "quantized_input_nodes": quantized_inputs,
                "output_quantized": output_quantized,
            }
        )
        status_counter[status] += 1
        op_counter[(name, status)] += 1

    op_rows = [
        {"target": target, "status": status, "count": count}
        for (target, status), count in sorted(
            op_counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "status": "ok",
        "summary": {
            "node_count": len(rows),
            "status_counts": dict(status_counter),
            "unique_op_status_count": len(op_rows),
        },
        "op_status_counts": op_rows,
        "nodes": rows,
    }


def make_quantizer(
    *,
    is_qat: bool,
    io_quantization: str,
) -> VgfQuantizer:
    compile_spec = cast(
        VgfCompileSpec, VgfCompileSpec()._set_preserve_io_quantization(True)
    )
    quantizer = VgfQuantizer(compile_spec, use_composable_quantizer=True)
    global_config = get_symmetric_quantization_config(is_qat=is_qat)
    quantizer.set_global(global_config)

    if io_quantization == "int8":
        quantizer.set_io(global_config)
    elif io_quantization == "uint8":
        quantizer.set_io(get_uint8_io_quantization_config(is_qat=is_qat))
    elif io_quantization == "none":
        quantizer.set_io(None)
    else:
        raise ValueError(f"Unsupported IO quantization: {io_quantization}")
    return quantizer


def make_vgf_compile_spec(output_dir: Path) -> VgfCompileSpec:
    compile_spec = cast(
        VgfCompileSpec, VgfCompileSpec()._set_preserve_io_quantization(True)
    )
    compile_spec.dump_intermediate_artifacts_to(str(output_dir))
    return compile_spec


def build_ptq_model(
    model: torch.nn.Module,
    calibration_inputs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    io_quantization: str,
) -> QuantizedModel:
    exported_model = export(model, calibration_inputs[0], strict=True).module(
        check_guards=False
    )
    quantizer = make_quantizer(is_qat=False, io_quantization=io_quantization)
    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model = move_exported_model_to_eval(prepared_model)

    with torch.no_grad():
        for inputs in calibration_inputs:
            prepared_model(*inputs)

    converted_model = convert_pt2e(prepared_model)
    return QuantizedModel(
        model=converted_model,
        coverage=quantization_coverage(converted_model),
    )


def build_qat_model(
    model: torch.nn.Module,
    training_samples: list[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
    *,
    steps: int,
    lr: float,
    io_quantization: str,
) -> QuantizedModel:
    exported_model = export(model, training_samples[0][0], strict=True).module(
        check_guards=False
    )
    quantizer = make_quantizer(is_qat=True, io_quantization=io_quantization)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    prepared_model = move_exported_model_to_train(prepared_model)
    optimizer = torch.optim.SGD(
        prepared_model.parameters(), lr=lr, momentum=0.9, foreach=True
    )

    for step in range(steps):
        total_loss = 0.0
        for inputs, target in training_samples:
            prediction = prepared_model(*inputs)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
        print(
            f"qat_step={step + 1}/{steps} "
            f"loss={total_loss / len(training_samples):.6f}"
        )

    prepared_model = move_exported_model_to_eval(prepared_model)
    converted_model = convert_pt2e(prepared_model)
    return QuantizedModel(
        model=converted_model,
        coverage=quantization_coverage(converted_model),
    )


def export_vgf_artifact(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    output_dir: Path,
    name: str,
    save_pte: bool,
) -> Path:
    artifact_dir = output_dir / f"vgf_{name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compile_spec = make_vgf_compile_spec(artifact_dir)
    partitioner = VgfPartitioner(compile_spec)

    print(f"[vgf] exporting {name} model to {artifact_dir} ...")
    aten_dialect = export(model, args=inputs, strict=True)
    edge_program_manager = to_edge_transform_and_lower(
        aten_dialect,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    if save_pte:
        pte_path = artifact_dir / f"rife_{name}_vgf.pte"
        save_pte_program(edge_program_manager.to_executorch(), str(pte_path))
        print(f"[vgf] wrote {pte_path}")
    return artifact_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    path: Path,
    *,
    aggregate: dict[str, Any],
    quantization_reports: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# RIFE PTQ/QAT Accuracy Report",
        "",
        "## Run",
        "",
        f"- Triples: {aggregate['triple_count']}",
        f"- Input shape: `1x3x{aggregate['height']}x{aggregate['width']}`",
        f"- Mode: `{aggregate['mode']}`",
        f"- IO quantization: `{aggregate['io_quantization']}`",
        f"- Data source: `{aggregate['data_source']}`",
        f"- Random seed: `{aggregate['random_seed']}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Mean | Min | Max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("eager_psnr", "Eager PSNR"),
        ("eager_ssim", "Eager SSIM"),
        ("ptq_psnr", "PTQ PSNR"),
        ("ptq_ssim", "PTQ SSIM"),
        ("ptq_vs_eager_mae", "PTQ vs eager MAE"),
        ("ptq_vs_eager_sqnr", "PTQ vs eager SQNR"),
        ("qat_psnr", "QAT PSNR"),
        ("qat_ssim", "QAT SSIM"),
        ("qat_vs_eager_mae", "QAT vs eager MAE"),
        ("qat_vs_eager_sqnr", "QAT vs eager SQNR"),
    ):
        if f"{key}_mean" not in aggregate:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_metric(aggregate.get(f"{key}_mean")),
                    format_metric(aggregate.get(f"{key}_min")),
                    format_metric(aggregate.get(f"{key}_max")),
                ]
            )
            + " |"
        )

    if quantization_reports:
        lines.extend(
            [
                "",
                "## Quantization Coverage",
                "",
                "| Mode | Nodes | Quantized Ops | Quantized Context Ops | "
                "FP32/Fallback Ops | Quantize Boundaries | "
                "Dequantize Boundaries |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, report in sorted(quantization_reports.items()):
            summary = report.get("summary", {})
            counts = summary.get("status_counts", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        format_metric(summary.get("node_count")),
                        format_metric(counts.get("quantized_op", 0)),
                        format_metric(counts.get("quantized_context_op", 0)),
                        format_metric(counts.get("fp32_or_fallback_op", 0)),
                        format_metric(counts.get("quantize_boundary", 0)),
                        format_metric(counts.get("dequantize_boundary", 0)),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.json`: full aggregate and per-triple metrics.",
            "- `metrics.csv`: tabular per-triple metrics.",
            "- `quantization_coverage/`: PTQ/QAT graph coverage reports.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quantization_coverage(
    output_dir: Path,
    name: str,
    coverage: dict[str, Any],
) -> None:
    coverage_dir = output_dir / "quantization_coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    write_json(coverage_dir / f"{name}.json", coverage)
    if coverage.get("status") != "ok":
        return
    write_csv(coverage_dir / f"{name}_nodes.csv", coverage["nodes"])
    write_csv(
        coverage_dir / f"{name}_op_status_counts.csv", coverage["op_status_counts"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--triples-root",
        type=Path,
        help=(
            "Directory containing frame sequences. Each directory with at "
            "least three images is interpreted as frame0, frame2, target "
            "frame1 triples."
        ),
    )
    source.add_argument(
        "--triples-list",
        type=Path,
        help=(
            "CSV or text file containing triples. Header form: "
            "name,input0,input1,target. Text form: input0 input1 target."
        ),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help=(
            "Path to a standard Practical-RIFE repo containing train_log/ "
            "and train_log/flownet.pkl."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the Practical-RIFE flownet.pkl checkpoint. Defaults "
            "to <model-root>/train_log/flownet.pkl."
        ),
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--preprocess",
        choices=("resize", "center-crop"),
        default="resize",
        help="How to produce the fixed model input shape.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=2,
        help="Stride between discovered sliding-window triples.",
    )
    parser.add_argument(
        "--max-triples",
        type=int,
        default=30,
        help="Maximum number of triples to evaluate.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
        help="Seed used when explicit random smoke-test samples are created.",
    )
    parser.add_argument(
        "--allow-random-data",
        action="store_true",
        help=(
            "Allow deterministic random data when --triples-list is missing. "
            "Only use this for smoke testing, not accuracy measurement."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("eager", "ptq", "both", "qat", "all"),
        default="both",
        help=(
            "eager=FP32 only, ptq=PTQ only, both=eager+PTQ, "
            "qat=eager+QAT, all=eager+PTQ+QAT."
        ),
    )
    parser.add_argument(
        "--io-quantization",
        choices=("uint8", "int8", "none"),
        default="uint8",
        help=(
            "Model IO quantization for PTQ/QAT. uint8 preserves external "
            "image IO; int8 uses symmetric IO; none leaves model IO FP32."
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=8,
        help="Number of triples used to calibrate PTQ.",
    )
    parser.add_argument(
        "--qat-samples",
        type=int,
        default=8,
        help="Number of triples used for QAT training.",
    )
    parser.add_argument(
        "--qat-steps",
        type=int,
        default=3,
        help="Number of QAT epochs over the selected samples.",
    )
    parser.add_argument(
        "--qat-lr",
        type=float,
        default=1.0e-5,
        help="Learning rate for the QAT optimizer.",
    )
    parser.add_argument(
        "--export-vgf",
        choices=("none", "ptq", "qat", "all"),
        default="none",
        help=(
            "Optionally lower the selected quantized model to VGF after "
            "accuracy evaluation. This writes artifacts under --output-dir."
        ),
    )
    parser.add_argument(
        "--save-pte",
        action="store_true",
        help="Also save an ExecuTorch .pte after VGF lowering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXECUTORCH_ROOT / "out" / "rife_qat_example_768x384",
        help="Directory for metrics and coverage reports.",
    )
    return parser.parse_args()


def load_requested_triples(args: argparse.Namespace) -> list[FrameTriple]:
    if args.triples_list is not None:
        if args.triples_list.exists():
            triples = load_triples_list(args.triples_list)
        elif args.allow_random_data:
            print(
                "WARNING: --triples-list was not found: "
                f"{args.triples_list}. Creating and using random data. "
                "Random data is only suitable for flow smoke testing, not "
                "real accuracy assessment."
            )
            triples = make_random_triples(args.max_triples)
        else:
            raise FileNotFoundError(
                f"--triples-list was not found: {args.triples_list}. "
                "Pass an existing triples file for accuracy measurement, "
                "or add --allow-random-data for smoke testing."
            )
    else:
        triples = discover_triples(args.triples_root, args.sequence_stride)
    return validate_triples(triples)[: args.max_triples]


def data_source(triples: list[FrameTriple]) -> str:
    return "random" if all(triple.synthetic for triple in triples) else "frames"


def validate_args(args: argparse.Namespace) -> None:
    validate_input_shape(args.height, args.width)
    if args.max_triples <= 0:
        raise ValueError("--max-triples must be positive.")
    if args.calibration_samples <= 0:
        raise ValueError("--calibration-samples must be positive.")
    if args.qat_samples <= 0:
        raise ValueError("--qat-samples must be positive.")
    if args.qat_steps <= 0:
        raise ValueError("--qat-steps must be positive.")
    if args.qat_lr <= 0.0:
        raise ValueError("--qat-lr must be positive.")
    if args.export_vgf in ("ptq", "all") and args.mode not in (
        "ptq",
        "both",
        "all",
    ):
        raise ValueError("--export-vgf ptq requires --mode ptq, both, or all.")
    if args.export_vgf in ("qat", "all") and args.mode not in ("qat", "all"):
        raise ValueError("--export-vgf qat requires --mode qat or all.")


def build_requested_quantized_models(
    args: argparse.Namespace,
    model: torch.nn.Module,
    triples: list[FrameTriple],
) -> tuple[torch.nn.Module | None, torch.nn.Module | None, dict[str, dict[str, Any]]]:
    run_ptq = args.mode in ("ptq", "both", "all")
    run_qat = args.mode in ("qat", "all")
    ptq_model = None
    quantization_reports: dict[str, dict[str, Any]] = {}

    if run_ptq:
        calibration_inputs = [
            load_triple_tensors(
                triple,
                height=args.height,
                width=args.width,
                preprocess=args.preprocess,
                random_seed=args.random_seed,
            )[0]
            for triple in triples[: args.calibration_samples]
        ]
        print(f"Building PTQ model with {len(calibration_inputs)} samples.")
        ptq = build_ptq_model(
            model,
            calibration_inputs,
            io_quantization=args.io_quantization,
        )
        ptq_model = ptq.model
        quantization_reports["ptq"] = ptq.coverage

    qat_model = None
    if run_qat:
        training_samples = [
            load_triple_tensors(
                triple,
                height=args.height,
                width=args.width,
                preprocess=args.preprocess,
                random_seed=args.random_seed,
            )
            for triple in triples[: args.qat_samples]
        ]
        print(
            "Building QAT model with "
            f"{len(training_samples)} samples and {args.qat_steps} steps."
        )
        qat = build_qat_model(
            model,
            training_samples,
            steps=args.qat_steps,
            lr=args.qat_lr,
            io_quantization=args.io_quantization,
        )
        qat_model = qat.model
        quantization_reports["qat"] = qat.coverage

    return ptq_model, qat_model, quantization_reports


def export_requested_vgf_artifacts(
    args: argparse.Namespace,
    triples: list[FrameTriple],
    ptq_model: torch.nn.Module | None,
    qat_model: torch.nn.Module | None,
) -> None:
    if args.export_vgf == "none":
        return

    inputs, _target = load_triple_tensors(
        triples[0],
        height=args.height,
        width=args.width,
        preprocess=args.preprocess,
        random_seed=args.random_seed,
    )
    if args.export_vgf in ("ptq", "all"):
        if ptq_model is None:
            raise RuntimeError("PTQ VGF export requested, but PTQ was not built.")
        export_vgf_artifact(
            ptq_model,
            inputs,
            output_dir=args.output_dir,
            name="ptq",
            save_pte=args.save_pte,
        )
    if args.export_vgf in ("qat", "all"):
        if qat_model is None:
            raise RuntimeError("QAT VGF export requested, but QAT was not built.")
        export_vgf_artifact(
            qat_model,
            inputs,
            output_dir=args.output_dir,
            name="qat",
            save_pte=args.save_pte,
        )


def update_row_with_prediction_metrics(
    row: dict[str, Any],
    *,
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    for key, value in quality_metrics(prediction, target).items():
        row[f"{prefix}_{key}"] = value


def update_row_with_delta_metrics(
    row: dict[str, Any],
    *,
    prefix: str,
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    for key, value in tensor_metrics(prediction, reference).items():
        row[f"{prefix}_{key}"] = value


def print_progress(index: int, total: int, row: dict[str, Any]) -> None:
    progress = f"{index + 1}/{total} {row['name']}"
    if "eager_psnr" in row:
        progress += f" eager_psnr={row['eager_psnr']:.3f}"
    if "ptq_psnr" in row:
        progress += f" ptq_psnr={row['ptq_psnr']:.3f}"
    if "qat_psnr" in row:
        progress += f" qat_psnr={row['qat_psnr']:.3f}"
    print(progress)


def evaluate_triples(
    args: argparse.Namespace,
    triples: list[FrameTriple],
    model: torch.nn.Module,
    ptq_model: torch.nn.Module | None,
    qat_model: torch.nn.Module | None,
) -> list[dict[str, Any]]:
    run_eager = args.mode in ("eager", "both", "qat", "all")
    rows: list[dict[str, Any]] = []
    for index, triple in enumerate(triples):
        inputs, target = load_triple_tensors(
            triple,
            height=args.height,
            width=args.width,
            preprocess=args.preprocess,
            random_seed=args.random_seed,
        )
        row: dict[str, Any] = {
            "index": index,
            "name": triple.name,
            "input0": str(triple.input0) if triple.input0 is not None else "<random>",
            "input1": str(triple.input1) if triple.input1 is not None else "<random>",
            "target": str(triple.target) if triple.target is not None else "<random>",
            "synthetic": triple.synthetic,
        }
        eager_output = None
        if run_eager:
            eager_output = run_model(model, inputs)
            update_row_with_prediction_metrics(
                row,
                prefix="eager",
                prediction=eager_output,
                target=target,
            )

        if ptq_model is not None:
            ptq_output = run_model(ptq_model, inputs)
            update_row_with_prediction_metrics(
                row,
                prefix="ptq",
                prediction=ptq_output,
                target=target,
            )
            if eager_output is None:
                eager_output = run_model(model, inputs)
            update_row_with_delta_metrics(
                row,
                prefix="ptq_vs_eager",
                prediction=ptq_output,
                reference=eager_output,
            )

        if qat_model is not None:
            qat_output = run_model(qat_model, inputs)
            update_row_with_prediction_metrics(
                row,
                prefix="qat",
                prediction=qat_output,
                target=target,
            )
            if eager_output is None:
                eager_output = run_model(model, inputs)
            update_row_with_delta_metrics(
                row,
                prefix="qat_vs_eager",
                prediction=qat_output,
                reference=eager_output,
            )

        rows.append(row)
        print_progress(index, len(triples), row)

    return rows


def build_aggregate(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "triple_count": len(rows),
        "height": args.height,
        "width": args.width,
        "preprocess": args.preprocess,
        "mode": args.mode,
        "io_quantization": args.io_quantization,
        "data_source": "random" if rows and rows[0].get("synthetic") else "frames",
        "random_seed": args.random_seed,
    }
    for prefix in ("eager", "ptq", "ptq_vs_eager", "qat", "qat_vs_eager"):
        aggregate.update(aggregate_metrics(rows, prefix))
    return aggregate


def write_outputs(
    args: argparse.Namespace,
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    quantization_reports: dict[str, dict[str, Any]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "metrics.json",
        {
            "aggregate": aggregate,
            "rows": rows,
            "quantization_coverage": {
                name: report.get("summary", report)
                for name, report in quantization_reports.items()
            },
        },
    )
    write_csv(args.output_dir / "metrics.csv", rows)
    for name, report in quantization_reports.items():
        write_quantization_coverage(args.output_dir, name, report)
    write_markdown_report(
        args.output_dir / "report.md",
        aggregate=aggregate,
        quantization_reports=quantization_reports,
    )

    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(f"Wrote {args.output_dir / 'metrics.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


def main() -> int:
    args = parse_args()
    validate_args(args)

    triples = load_requested_triples(args)
    source = data_source(triples)
    print(
        f"Loaded {len(triples)} "
        f"{'random samples' if source == 'random' else 'frame triples'}."
    )
    print(f"Input shape: 1x3x{args.height}x{args.width}")
    print(f"Mode: {args.mode}")
    print(f"IO quantization: {args.io_quantization}")
    print(f"Data source: {source}")

    model = load_rife_model(args.model_root, args.checkpoint)
    ptq_model, qat_model, quantization_reports = build_requested_quantized_models(
        args,
        model,
        triples,
    )
    rows = evaluate_triples(args, triples, model, ptq_model, qat_model)
    write_outputs(args, build_aggregate(args, rows), rows, quantization_reports)
    export_requested_vgf_artifacts(args, triples, ptq_model, qat_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
