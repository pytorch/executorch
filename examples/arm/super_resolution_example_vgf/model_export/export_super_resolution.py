# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from torch.utils.data import DataLoader
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Keep script-compatible imports without requiring package execution.
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (  # type: ignore[import-not-found, no-redef]
        create_model_bundle,
        evaluate_super_resolution_model,
        load_calibration_inputs,
        PairedSuperResolutionDataset,
        SUPPORTED_MODELS,
        write_json,
    )
else:
    from .common import (
        create_model_bundle,
        evaluate_super_resolution_model,
        load_calibration_inputs,
        PairedSuperResolutionDataset,
        SUPPORTED_MODELS,
        write_json,
    )

CALIBRATION_MAX_SAMPLES = 1000
EVAL_MAX_SAMPLES = 1000


def has_quantized_out_variants() -> bool:
    try:
        _ = torch.ops.quantized_decomposed.quantize_per_tensor.out
        _ = torch.ops.quantized_decomposed.dequantize_per_tensor.out
        return True
    except AttributeError:
        return False


def ensure_quantized_ops_loaded() -> Path | None:
    if has_quantized_out_variants():
        return None

    quantized_kernels_available = False
    try:
        import executorch.kernels.quantized  # noqa: F401
    except ImportError:
        quantized_kernels_available = False
    else:
        quantized_kernels_available = True

    if quantized_kernels_available and has_quantized_out_variants():
        return None

    repo_root = Path(__file__).resolve().parents[4]
    search_patterns = (
        "cmake-out/kernels/quantized/libquantized_ops_aot_lib.*",
        "arm_test/*/kernels/quantized/libquantized_ops_aot_lib.*",
    )
    for pattern in search_patterns:
        for candidate in sorted(repo_root.glob(pattern)):
            if not candidate.is_file():
                continue
            torch.ops.load_library(str(candidate))
            if has_quantized_out_variants():
                return candidate

    raise RuntimeError(
        "INT8 export requires the quantized ops out-variant library. "
        "Build or install ExecuTorch quantized kernels so that "
        "`quantized_decomposed::quantize_per_tensor.out` and "
        "`quantized_decomposed::dequantize_per_tensor.out` are available."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Swin2SR model for VGF.")
    parser.add_argument(
        "--model-name",
        choices=SUPPORTED_MODELS,
        default="swin2sr",
        help="Model profile to export.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve Hugging Face assets from local cache only.",
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=None,
        help=(
            "Pinned Hugging Face revision to use when --checkpoint points to a model id. "
            "Required for remote checkpoints; ignored for local checkpoint paths."
        ),
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=64,
        help="Static low-resolution input height used for export.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=64,
        help="Static low-resolution input width used for export.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Destination .pte path.",
    )
    parser.add_argument(
        "--quantization-mode",
        choices=("none", "int8"),
        default="none",
        help="Quantization mode used before lowering to VGF.",
    )
    parser.add_argument(
        "--calibration-lr-dir",
        default=None,
        help=(
            "Directory of low-resolution images used for PTQ calibration. "
            "Required when --quantization-mode=int8."
        ),
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=32,
        help="Maximum number of calibration images to use.",
    )
    parser.add_argument(
        "--eval-lr-dir",
        default=None,
        help="Optional directory of low-resolution evaluation images.",
    )
    parser.add_argument(
        "--eval-hr-dir",
        default=None,
        help="Optional directory of high-resolution evaluation targets.",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=100,
        help="Maximum number of evaluation image pairs to use.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory for intermediate VGF/TOSA artifacts.",
    )
    return parser.parse_args()


def quantize_model(
    model: torch.nn.Module,
    quantizer: VgfQuantizer,
    example_inputs: tuple[torch.Tensor],
    calibration_samples: list[tuple[torch.Tensor]],
) -> torch.export.ExportedProgram:
    exported_program = torch.export.export(model, example_inputs)
    graph_module = exported_program.module(check_guards=False)

    prepared = prepare_pt2e(graph_module, quantizer)
    for sample in calibration_samples:
        prepared(*sample)

    quantized_model = convert_pt2e(prepared)
    return torch.export.export(quantized_model, example_inputs)


def write_delegation_report(edge_program_manager, report_path: Path) -> None:
    delegation_info = get_delegation_info(
        edge_program_manager.exported_program().graph_module
    )
    report_path.write_text(delegation_info.get_summary() + "\n")


def maybe_make_eval_loader(
    eval_lr_dir: str | None,
    eval_hr_dir: str | None,
    input_height: int,
    input_width: int,
    upscale: int,
    num_eval_samples: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]] | None:
    if eval_lr_dir is None and eval_hr_dir is None:
        return None
    if (eval_lr_dir is None) != (eval_hr_dir is None):
        raise ValueError("--eval-lr-dir and --eval-hr-dir must be provided together.")

    eval_dataset = PairedSuperResolutionDataset(
        eval_lr_dir,
        eval_hr_dir,
        input_height,
        input_width,
        upscale,
        crop_mode="center",
        max_samples=min(num_eval_samples, EVAL_MAX_SAMPLES),
    )
    return DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    delegation_path = output_path.with_name(f"{output_path.stem}_delegation.txt")
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")

    bundle = create_model_bundle(
        model_name=args.model_name,
        input_height=args.input_height,
        input_width=args.input_width,
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        local_files_only=args.local_files_only,
    )

    quantize = args.quantization_mode != "none"
    compile_spec = VgfCompileSpec("TOSA-1.0+INT" if quantize else "TOSA-1.0+FP")
    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        compile_spec.dump_intermediate_artifacts_to(str(artifact_dir))

    calibration_samples: list[tuple[torch.Tensor]] = []
    if quantize:
        quantized_ops_library = ensure_quantized_ops_loaded()
        if quantized_ops_library is not None:
            print(f"Loaded quantized ops library from {quantized_ops_library}")
        if args.calibration_lr_dir is None:
            raise ValueError(
                "--calibration-lr-dir is required when --quantization-mode=int8."
            )
        calibration_samples = load_calibration_inputs(
            args.calibration_lr_dir,
            args.input_height,
            args.input_width,
            min(args.num_calibration_samples, CALIBRATION_MAX_SAMPLES),
        )

    exported_program: torch.export.ExportedProgram
    if quantize:
        quantizer = VgfQuantizer(compile_spec)
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        exported_program = quantize_model(
            bundle.model,
            quantizer,
            bundle.example_inputs,
            calibration_samples,
        )
    else:
        exported_program = torch.export.export(bundle.model, bundle.example_inputs)

    eval_loader = maybe_make_eval_loader(
        args.eval_lr_dir,
        args.eval_hr_dir,
        args.input_height,
        args.input_width,
        bundle.upscale,
        args.num_eval_samples,
    )
    evaluation_metrics: dict[str, float] | None = None
    if eval_loader is not None:
        eval_module = exported_program.module(check_guards=False)
        metrics = evaluate_super_resolution_model(
            eval_module,
            eval_loader,
            torch.device("cpu"),
        )
        evaluation_metrics = metrics
        write_json(metrics_path, metrics)
        print(
            "Evaluation metrics: "
            f"l1={metrics['l1']:.6f} "
            f"psnr={metrics['psnr']:.4f} "
            f"ssim={metrics['ssim']:.4f}"
        )

    partitioner = VgfPartitioner(compile_spec)
    edge_program_manager = to_edge_transform_and_lower(
        programs=exported_program,
        partitioner=[partitioner],
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
    if not output_path.is_file():
        raise RuntimeError(f"Expected exported model at {output_path}")

    write_json(
        metadata_path,
        {
            "model_name": bundle.model_name,
            "checkpoint": args.checkpoint,
            "checkpoint_revision": args.checkpoint_revision,
            "input_shape": list(bundle.input_shape),
            "output_shape": list(bundle.output_shape),
            "input_dtype": bundle.input_dtype,
            "output_dtype": bundle.output_dtype,
            "num_outputs": 1,
            "upscale": bundle.upscale,
            "window_size": bundle.window_size,
            "quantization_mode": args.quantization_mode,
            "num_calibration_samples": len(calibration_samples),
            "num_eval_samples": (
                int(evaluation_metrics["num_samples"])
                if evaluation_metrics is not None
                else 0
            ),
        },
    )

    print(f"Exported model saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Delegation summary saved to {delegation_path}")
    if evaluation_metrics is not None:
        print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
