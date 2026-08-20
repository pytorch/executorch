# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tqdm  # type: ignore[import]
from executorch.backends.arm.common.pipeline_config import (
    ArmPassPipelineConfig,
    SoftmaxDecompositionConfig,
)
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from packaging.version import Version
from PIL import Image
from torchao.quantization.pt2e.quantize_pt2e import (  # type: ignore[import]
    convert_pt2e,
    prepare_pt2e,
)

MOBILE_SAM_SOURCE_URL = "https://github.com/ChaoningZhang/MobileSAM"
MOBILE_SAM_SOURCE_REVISION = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
MOBILE_SAM_PATCH = "0001-Make-TinyViT-image-size-configurable.patch"
DEFAULT_CHECKPOINT_FILENAME = "mobile_sam.pt"
DEFAULT_CHECKPOINT_URL = (
    f"{MOBILE_SAM_SOURCE_URL}/raw/{MOBILE_SAM_SOURCE_REVISION}/weights/"
    f"{DEFAULT_CHECKPOINT_FILENAME}"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
)
MOBILE_SAM_SOURCE_LICENSE = "Apache-2.0"
MINIMUM_VELA_VERSION = Version("5.1.0")
DEFAULT_INPUT_SIZE = 448
MOBILE_SAM_INPUT_ALIGNMENT = 16


@dataclass
class PreparedSample:
    name: str
    image: Image.Image
    pixel_values: torch.Tensor
    labels: torch.Tensor | None


def load_mobile_sam(
    checkpoint_path: str,
    mobile_sam_source: str | None,
    input_size: int,
) -> torch.nn.Module:
    mobile_sam_module = import_mobile_sam_module(mobile_sam_source)
    builder = mobile_sam_module.sam_model_registry["vit_t"]
    if "image_size" not in inspect.signature(builder).parameters:
        raise RuntimeError(
            "The MobileSAM checkout does not provide configurable image sizes. "
            "Run model_export/prepare_mobilesam.py and pass the prepared checkout "
            "with --mobile-sam-source."
        )
    return builder(checkpoint=checkpoint_path, image_size=input_size).eval()


class MobileSAMFixedPrompt(torch.nn.Module):
    image_encoder: Any
    mask_decoder: Any

    def __init__(
        self,
        sam: torch.nn.Module,
        point_prompts: list[tuple[float, float]],
    ) -> None:
        super().__init__()
        sam = cast(Any, sam)

        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder

        with torch.no_grad():
            points = (
                torch.tensor([point_prompts], dtype=torch.float32),
                torch.ones((1, len(point_prompts)), dtype=torch.int64),
            )
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            image_pe = sam.prompt_encoder.get_dense_pe()

        self.register_buffer("sparse_prompt_embeddings", sparse_embeddings)
        self.register_buffer("dense_prompt_embeddings", dense_embeddings)
        self.register_buffer("image_pe", image_pe)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.image_encoder(pixel_values)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=self.sparse_prompt_embeddings,
            dense_prompt_embeddings=self.dense_prompt_embeddings,
            multimask_output=False,
        )
        return low_res_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed-prompt MobileSAM segmentation for Ethos-U."
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional local MobileSAM checkpoint path.",
    )
    parser.add_argument(
        "--mobile-sam-source",
        default=None,
        help=(
            "Optional MobileSAM checkout prepared by prepare_mobilesam.py. "
            "Used before importing the mobile_sam package."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the checkpoint from the local cache only; do not download.",
    )
    parser.add_argument(
        "--calibration-image",
        action="append",
        default=[],
        help="Local RGB image used for PTQ calibration. Can be repeated.",
    )
    parser.add_argument(
        "--eval-image",
        action="append",
        default=[],
        help="Local RGB image used for validation/debugging. Can be repeated.",
    )
    parser.add_argument(
        "--eval-mask",
        action="append",
        default=[],
        help=(
            "Optional binary reference mask used for validation/debugging. "
            "Can be repeated and must match --eval-image count when provided."
        ),
    )
    parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help=(
            "Positive point prompt in the resized square input frame. "
            "Can be repeated for multi-point prompts."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Mask-logit threshold used for metrics and debug masks.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the exported ExecuTorch program.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Square MobileSAM input size. Must be divisible by 16.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=4,
        help="Number of local samples used for PTQ calibration.",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=4,
        help="Number of local samples used for host-side validation.",
    )
    parser.add_argument(
        "--num-debug-samples",
        type=int,
        default=4,
        help="Number of validation samples written as visual debug artifacts.",
    )
    parser.add_argument(
        "--minimum-fp32-quantized-iou",
        type=float,
        default=None,
        help="Fail before lowering when host quantized/FP32 mask IoU is lower.",
    )
    parser.add_argument(
        "--target",
        default="ethos-u85-256",
        help="Ethos-U target passed to Vela.",
    )
    parser.add_argument(
        "--system-config",
        default="Ethos_U85_SYS_DRAM_Mid",
        help="Vela system configuration.",
    )
    parser.add_argument(
        "--memory-mode",
        default="Dedicated_Sram_384KB",
        help="Vela memory mode.",
    )
    parser.add_argument(
        "--extra-vela-flag",
        action="append",
        default=[],
        help="Additional Vela flag. Can be provided multiple times.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory for intermediate TOSA/Vela artifacts.",
    )
    parser.add_argument(
        "--debug-output-dir",
        default=None,
        help="Optional directory for masks, overlays, and validation summaries.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_vela_version() -> str:
    try:
        installed_version = version("ethos-u-vela")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "MobileSAM export requires ethos-u-vela 5.1.0 or newer. "
            "Run examples/arm/setup.sh and retry."
        ) from error

    if Version(installed_version) < MINIMUM_VELA_VERSION:
        raise RuntimeError(
            "MobileSAM A16W8 attention requires ethos-u-vela 5.1.0 or newer; "
            f"found {installed_version}. Run examples/arm/setup.sh and retry."
        )
    return installed_version


def import_mobile_sam_module(mobile_sam_source: str | None) -> Any:
    if mobile_sam_source is not None:
        sys.path.insert(0, str(Path(mobile_sam_source).expanduser().resolve()))
    try:
        return importlib.import_module("mobile_sam")
    except ImportError as error:
        raise ImportError(
            "Could not import the patched mobile_sam package. Run "
            "model_export/prepare_mobilesam.py and pass its checkout with "
            "--mobile-sam-source."
        ) from error


def find_module_type(module: torch.nn.Module, class_name: str) -> type[torch.nn.Module]:
    for child in module.modules():
        if child.__class__.__name__ == class_name:
            return child.__class__
    raise ValueError(f"Could not find module type {class_name} in {module.__class__}.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_checkpoint_cache_dir() -> Path:
    return (
        Path.home() / ".cache" / "executorch" / "mobilesam" / MOBILE_SAM_SOURCE_REVISION
    )


def verify_checkpoint(path: Path, expected_sha256: str) -> None:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}."
        )


def download_checkpoint(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with (
            urllib.request.urlopen(url, timeout=60) as response,  # nosec B310
            temp_path.open("wb") as file,
        ):
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                file.write(chunk)
        temp_path.replace(path)
    except (OSError, urllib.error.URLError) as error:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download MobileSAM checkpoint from {url}."
        ) from error


def resolve_checkpoint(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None]:
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path), None, None

    checkpoint_path = default_checkpoint_cache_dir() / DEFAULT_CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        if args.local_files_only:
            raise FileNotFoundError(
                f"Checkpoint not found in local cache: {checkpoint_path}"
            )
        download_checkpoint(DEFAULT_CHECKPOINT_URL, checkpoint_path)

    verify_checkpoint(checkpoint_path, DEFAULT_CHECKPOINT_SHA256)
    return (
        str(checkpoint_path.resolve()),
        DEFAULT_CHECKPOINT_URL,
        DEFAULT_CHECKPOINT_SHA256,
    )


def preprocess_image(
    image: Image.Image,
    segmentation_map: Image.Image | None,
    *,
    name: str,
    input_size: int,
    pixel_mean: np.ndarray,
    pixel_std: np.ndarray,
    output_mask_size: tuple[int, int] | None,
) -> PreparedSample:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    scale = input_size / max(height, width)
    resized_size = (round(width * scale), round(height * scale))
    resized_image = rgb_image.resize(resized_size, Image.Resampling.BILINEAR)

    padded_image = Image.new("RGB", (input_size, input_size))
    padded_image.paste(resized_image, (0, 0))

    image_np = np.asarray(resized_image, dtype=np.float32)
    image_np = (image_np - pixel_mean) / pixel_std
    pixel_values = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    pixel_values = torch.nn.functional.pad(
        pixel_values,
        (
            0,
            input_size - resized_size[0],
            0,
            input_size - resized_size[1],
        ),
    )
    pixel_values = pixel_values.contiguous()

    labels = None
    if segmentation_map is not None:
        if output_mask_size is None:
            raise ValueError("An output mask size is required for reference masks.")
        resized_mask = segmentation_map.convert("L").resize(
            resized_size,
            Image.Resampling.NEAREST,
        )
        padded_mask = Image.new("L", (input_size, input_size))
        padded_mask.paste(resized_mask, (0, 0))
        resized_mask = padded_mask.resize(output_mask_size, Image.Resampling.NEAREST)
        mask_np = np.asarray(resized_mask, dtype=np.uint8)
        labels = torch.from_numpy((mask_np > 0).astype(np.uint8)).to(torch.long)

    return PreparedSample(
        name=name,
        image=padded_image,
        pixel_values=pixel_values,
        labels=labels,
    )


def load_local_samples(
    image_paths: list[str],
    mask_paths: list[str],
    limit: int,
    *,
    include_labels: bool,
    input_size: int,
    pixel_mean: np.ndarray,
    pixel_std: np.ndarray,
    output_mask_size: tuple[int, int] | None = None,
) -> list[PreparedSample]:
    if include_labels and len(mask_paths) not in (0, len(image_paths)):
        raise ValueError("--eval-mask must be omitted or match --eval-image count.")

    samples: list[PreparedSample] = []
    for index, image_path in enumerate(image_paths):
        if len(samples) >= limit:
            break
        image = Image.open(image_path)
        mask = None
        if include_labels and len(mask_paths) > 0:
            mask = Image.open(mask_paths[index])
        samples.append(
            preprocess_image(
                image,
                mask,
                name=Path(image_path).stem or f"local_sample_{index:04d}",
                input_size=input_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                output_mask_size=output_mask_size,
            )
        )
    if len(samples) == 0:
        raise ValueError("No local samples were loaded.")
    return samples


def run_mask_logits(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    output = model(input_tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor mask logits, got {type(output)}")
    return output


def predict_mask(logits: torch.Tensor, threshold: float) -> np.ndarray:
    mask = logits.detach().cpu().squeeze(0).squeeze(0).numpy() > threshold
    return mask.astype(np.uint8)


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def save_binary_mask(path: Path, mask: np.ndarray) -> Image.Image:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(path)
    return image.convert("RGB")


def save_mask_overlay(
    path: Path,
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    rgb_image = image.convert("RGB")
    resized_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        rgb_image.size,
        Image.Resampling.NEAREST,
    )
    mask_np = np.asarray(resized_mask, dtype=np.uint8) > 0
    overlay_np = np.asarray(rgb_image, dtype=np.float32)
    color_np = np.asarray(color, dtype=np.float32)
    overlay_np[mask_np] = overlay_np[mask_np] * 0.55 + color_np * 0.45
    Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8), mode="RGB").save(path)


def write_debug_artifacts(
    debug_dir: Path,
    sample: PreparedSample,
    fp32_mask: np.ndarray,
    quantized_mask: np.ndarray,
) -> None:
    sample_dir = debug_dir / sample.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample.image.save(sample_dir / "input.png")
    save_binary_mask(sample_dir / "fp32_mask.png", fp32_mask)
    save_binary_mask(sample_dir / "quantized_mask.png", quantized_mask)
    save_mask_overlay(
        sample_dir / "fp32_overlay.png",
        sample.image,
        fp32_mask,
        (0, 220, 120),
    )
    save_mask_overlay(
        sample_dir / "quantized_overlay.png",
        sample.image,
        quantized_mask,
        (0, 170, 255),
    )
    if sample.labels is not None:
        save_binary_mask(
            sample_dir / "reference_mask.png",
            sample.labels.detach().cpu().numpy().astype(np.uint8),
        )

    mismatch = fp32_mask != quantized_mask
    heatmap = np.zeros((*fp32_mask.shape, 3), dtype=np.uint8)
    heatmap[~mismatch] = [0, 128, 0]
    heatmap[mismatch] = [255, 0, 0]
    Image.fromarray(heatmap, mode="RGB").save(sample_dir / "mismatch_heatmap.png")

    write_json(
        sample_dir / "mask_summary.json",
        {
            "foreground_pixels": int(quantized_mask.sum()),
            "background_pixels": int(quantized_mask.size - quantized_mask.sum()),
            "fp32_quantized_iou": binary_iou(fp32_mask, quantized_mask),
        },
    )


def evaluate_and_debug(
    fp32_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    eval_samples: list[PreparedSample],
    debug_dir: Path | None,
    num_debug_samples: int,
    threshold: float,
) -> dict[str, float]:
    fp32_quantized_ious: list[float] = []
    fp32_quantized_pixel_agreements: list[float] = []
    reference_ious: list[float] = []

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print("\nEvaluating quantized MobileSAM on validation samples...")
    for index, sample in enumerate(tqdm.tqdm(eval_samples)):
        fp32_logits = run_mask_logits(fp32_model, sample.pixel_values)
        quantized_logits = run_mask_logits(quantized_model, sample.pixel_values)
        fp32_mask = predict_mask(fp32_logits, threshold)
        quantized_mask = predict_mask(quantized_logits, threshold)

        fp32_quantized_ious.append(binary_iou(fp32_mask, quantized_mask))
        fp32_quantized_pixel_agreements.append(
            float(np.mean(fp32_mask == quantized_mask))
        )
        if sample.labels is not None:
            labels = sample.labels.detach().cpu().numpy().astype(np.uint8)
            reference_ious.append(binary_iou(quantized_mask, labels))

        if debug_dir is not None and index < num_debug_samples:
            write_debug_artifacts(debug_dir, sample, fp32_mask, quantized_mask)

    metrics = {
        "num_samples": float(len(eval_samples)),
        "fp32_quantized_mean_iou": float(np.mean(fp32_quantized_ious)),
        "fp32_quantized_pixel_agreement": float(
            np.mean(fp32_quantized_pixel_agreements)
        ),
    }
    if len(reference_ious) > 0:
        metrics["reference_mean_iou"] = float(np.mean(reference_ious))
    return metrics


def quantize_model(
    model: torch.nn.Module,
    quantizer: EthosUQuantizer,
    example_inputs: tuple[torch.Tensor],
    calibration_samples: list[PreparedSample],
) -> torch.export.ExportedProgram:
    exported = torch.export.export(model, example_inputs)
    prepared = prepare_pt2e(exported.module(), quantizer)

    print("\nCalibrating MobileSAM...")
    for sample in tqdm.tqdm(calibration_samples):
        prepared(sample.pixel_values)

    quantized = convert_pt2e(prepared)
    return torch.export.export(quantized, example_inputs)


def has_quantized_out_variants() -> bool:
    try:
        _ = torch.ops.quantized_decomposed.quantize_per_tensor.out
        _ = torch.ops.quantized_decomposed.dequantize_per_tensor.out
        return True
    except AttributeError:
        return False


def load_quantized_ops_library(library_path: Path) -> Path:
    if not library_path.is_file():
        raise FileNotFoundError(f"Quantized ops library not found: {library_path}")
    torch.ops.load_library(str(library_path))
    if has_quantized_out_variants():
        return library_path
    raise RuntimeError(
        f"Quantized ops library did not register required out variants: {library_path}"
    )


def ensure_quantized_ops_loaded() -> Path | None:
    if has_quantized_out_variants():
        return None

    quantized_ops_library = os.environ.get("EXECUTORCH_QUANTIZED_OPS_AOT_LIBRARY")
    if quantized_ops_library:
        return load_quantized_ops_library(
            Path(quantized_ops_library).expanduser().resolve()
        )

    try:
        import executorch.kernels.quantized  # noqa: F401
    except ImportError:
        pass
    else:
        if has_quantized_out_variants():
            return None

    repo_root = Path(__file__).resolve().parents[4]
    search_patterns = (
        "cmake-out/kernels/quantized/libquantized_ops_aot_lib.*",
        "arm_test/*/kernels/quantized/libquantized_ops_aot_lib.*",
        "arm_test/**/kernels/quantized/libquantized_ops_aot_lib.*",
    )
    for pattern in search_patterns:
        for candidate in sorted(repo_root.glob(pattern)):
            if not candidate.is_file():
                continue
            return load_quantized_ops_library(candidate)

    raise RuntimeError(
        "MobileSAM int8 export requires the quantized ops out-variant library. "
        "Build or install ExecuTorch quantized kernels so that "
        "`quantized_decomposed::quantize_per_tensor.out` and "
        "`quantized_decomposed::dequantize_per_tensor.out` are available."
    )


def write_delegation_report(edge_program_manager: Any, report_path: Path) -> None:
    delegation_info = get_delegation_info(
        edge_program_manager.exported_program().graph_module
    )
    report_path.write_text(delegation_info.get_summary() + "\n")


def resolve_point_prompts(args: argparse.Namespace) -> list[tuple[float, float]]:
    if len(args.point) > 0:
        point_prompts = [(float(x), float(y)) for x, y in args.point]
    else:
        point_prompts = [(args.input_size / 2, args.input_size / 2)]

    for point_x, point_y in point_prompts:
        if not (0 <= point_x <= args.input_size and 0 <= point_y <= args.input_size):
            raise ValueError("Point prompts must be inside the square input.")
    return point_prompts


def validate_export_args(args: argparse.Namespace) -> None:
    if args.input_size < 224 or args.input_size % MOBILE_SAM_INPUT_ALIGNMENT != 0:
        raise ValueError("--input-size must be at least 224 and divisible by 16.")
    if args.num_calibration_samples <= 0:
        raise ValueError("--num-calibration-samples must be positive.")
    if args.num_eval_samples <= 0:
        raise ValueError("--num-eval-samples must be positive.")
    if args.minimum_fp32_quantized_iou is not None and not (
        0.0 <= args.minimum_fp32_quantized_iou <= 1.0
    ):
        raise ValueError("--minimum-fp32-quantized-iou must be between 0 and 1.")
    if len(args.calibration_image) == 0:
        raise ValueError("At least one --calibration-image is required.")
    if len(args.eval_image) == 0:
        args.eval_image = list(args.calibration_image)
    if len(args.eval_mask) not in (0, len(args.eval_image)):
        raise ValueError("--eval-mask must be omitted or match --eval-image count.")


def main() -> None:
    args = parse_args()
    validate_export_args(args)
    vela_version = validate_vela_version()
    point_prompts = resolve_point_prompts(args)
    quantized_ops_library = ensure_quantized_ops_loaded()
    if quantized_ops_library is not None:
        print(f"Loaded quantized ops library from {quantized_ops_library}")

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    delegation_path = output_path.with_name(f"{output_path.stem}_delegation.txt")
    debug_dir = Path(args.debug_output_dir).resolve() if args.debug_output_dir else None

    checkpoint_path, checkpoint_url, checkpoint_sha256 = resolve_checkpoint(args)
    mobile_sam = load_mobile_sam(
        checkpoint_path,
        args.mobile_sam_source,
        args.input_size,
    )
    pixel_mean = cast(Any, mobile_sam).pixel_mean.detach().cpu().reshape(-1).numpy()
    pixel_std = cast(Any, mobile_sam).pixel_std.detach().cpu().reshape(-1).numpy()
    if pixel_mean.shape != (3,) or pixel_std.shape != (3,):
        raise ValueError("MobileSAM preprocessing must provide three RGB values.")
    wrapped_model = MobileSAMFixedPrompt(mobile_sam, point_prompts).eval()

    calibration_samples = load_local_samples(
        args.calibration_image,
        [],
        args.num_calibration_samples,
        include_labels=False,
        input_size=args.input_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    example_inputs = (calibration_samples[0].pixel_values,)
    with torch.no_grad():
        output_shape = list(run_mask_logits(wrapped_model, example_inputs[0]).shape)
    if len(output_shape) != 4 or output_shape[:2] != [1, 1]:
        raise ValueError(
            f"Expected MobileSAM output shape [1, 1, height, width], got {output_shape}."
        )
    output_mask_size = (output_shape[3], output_shape[2])

    eval_samples = load_local_samples(
        args.eval_image,
        args.eval_mask,
        args.num_eval_samples,
        include_labels=True,
        input_size=args.input_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        output_mask_size=output_mask_size,
    )

    compile_spec = EthosUCompileSpec(
        target=args.target,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        extra_flags=args.extra_vela_flag,
    )
    compile_spec.set_pass_pipeline_config(
        ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE)
    )
    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        compile_spec.dump_intermediate_artifacts_to(str(artifact_dir))

    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    attention_module_type = find_module_type(wrapped_model.image_encoder, "Attention")
    quantizer.set_module_type(
        attention_module_type,
        get_symmetric_a16w8_quantization_config(),
    )

    with torch.no_grad():
        quantized_program = quantize_model(
            wrapped_model,
            quantizer,
            example_inputs,
            calibration_samples,
        )
        quantized_module = quantized_program.module()
        metrics = evaluate_and_debug(
            wrapped_model,
            quantized_module,
            eval_samples,
            debug_dir,
            args.num_debug_samples,
            args.mask_threshold,
        )
        write_json(metrics_path, metrics)
        print(
            "Validation metrics: "
            f"fp32_quantized_mean_iou={metrics['fp32_quantized_mean_iou']:.4f} "
            "fp32_quantized_pixel_agreement="
            f"{metrics['fp32_quantized_pixel_agreement']:.4f}"
        )
        if (
            args.minimum_fp32_quantized_iou is not None
            and metrics["fp32_quantized_mean_iou"] < args.minimum_fp32_quantized_iou
        ):
            raise RuntimeError(
                "Host quantized/FP32 mask IoU "
                f"{metrics['fp32_quantized_mean_iou']:.4f} is below "
                f"{args.minimum_fp32_quantized_iou:.4f}."
            )

    edge_program_manager = to_edge_transform_and_lower(
        programs=quantized_program,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    write_delegation_report(edge_program_manager, delegation_path)

    executorch_program_manager = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(
        executorch_program_manager,
        str(output_path),
        output_dir=str(output_path.parent),
    )

    write_json(
        metadata_path,
        {
            "model_name": "MobileSAM vit_t",
            "checkpoint_filename": DEFAULT_CHECKPOINT_FILENAME,
            "checkpoint_path": checkpoint_path,
            "checkpoint_url": checkpoint_url,
            "checkpoint_sha256": checkpoint_sha256,
            "mobile_sam_source_license": MOBILE_SAM_SOURCE_LICENSE,
            "mobile_sam_source_url": MOBILE_SAM_SOURCE_URL,
            "mobile_sam_source_revision": MOBILE_SAM_SOURCE_REVISION,
            "mobile_sam_patch": MOBILE_SAM_PATCH,
            "input_shape": list(example_inputs[0].shape),
            "output_shape": output_shape,
            "input_size": args.input_size,
            "preprocessing": {
                "pixel_mean": pixel_mean.tolist(),
                "pixel_std": pixel_std.tolist(),
                "resize": "longest_side_then_zero_pad",
            },
            "point_prompts_xy": point_prompts,
            "mask_threshold": args.mask_threshold,
            "target": args.target,
            "vela_version": vela_version,
            "system_config": args.system_config,
            "memory_mode": args.memory_mode,
            "extra_vela_flags": args.extra_vela_flag,
            "quantization": {
                "global": "int8 activations and int8 weights",
                "tinyvit_attention": "int16 activations and int8 weights",
            },
            "num_calibration_samples": len(calibration_samples),
            "num_eval_samples": len(eval_samples),
            "calibration_images": args.calibration_image,
            "eval_images": args.eval_image,
            "eval_masks": args.eval_mask,
            "output_path": str(output_path),
            "metrics_path": str(metrics_path),
            "delegation_path": str(delegation_path),
            "debug_output_dir": str(debug_dir) if debug_dir is not None else None,
        },
    )

    print(f"\nExported model saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Delegation summary saved to {delegation_path}")
    if debug_dir is not None:
        print(f"Debug artifacts saved to {debug_dir}")


if __name__ == "__main__":
    main()
