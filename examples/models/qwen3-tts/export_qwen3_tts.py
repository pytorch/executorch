import argparse
import json
import sys
from pathlib import Path

import torch
from torch.export import export

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import DecoderExportMetadata, make_decode_export_module, make_sample_codes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qwen3-TTS speech tokenizer decoder to ExecuTorch."
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        required=True,
        help="Directory produced by convert_weights.py (contains decoder_metadata.json).",
    )
    parser.add_argument(
        "--backend",
        choices=["portable", "xnnpack"],
        default="xnnpack",
        help="Backend to target for decoder export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./qwen3_tts_exports"),
        help="Output directory for model.pte.",
    )
    parser.add_argument(
        "--fixed-codes-len",
        type=int,
        default=1200,
        help="Static codec sequence length used for export.",
    )
    parser.add_argument(
        "--bucket-sizes",
        type=str,
        default=None,
        help="Comma-separated list of bucket sizes to export (e.g. '75,150,300,600,1200'). "
        "Each bucket produces a separate model_{size}.pte file. Overrides --fixed-codes-len.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Decoder weight dtype for export.",
    )
    parser.add_argument(
        "--qlinear",
        choices=["4w", "8w", "8da4w", "8da8w", "fpa4w"],
        default=None,
        help="Optional quantization mode for linear layers.",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=32,
        help="Group size for linear quantization.",
    )
    parser.add_argument(
        "--qlinear-packing-format",
        choices=["tile_packed_to_4d"],
        default=None,
        help="Optional packing format for 4w quantization.",
    )
    return parser.parse_args()


def lower_to_executorch(programs, constant_methods: dict, backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        partitioner = {
            "decode_codes": [
                XnnpackDynamicallyQuantizedPartitioner(),
                XnnpackPartitioner(),
            ]
        }
    else:
        partitioner = []

    edge_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    return edge_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )


def export_single_bucket(
    module,
    metadata: DecoderExportMetadata,
    codes_len: int,
    backend: str,
    output_dir: Path,
    model_filename: str = "model.pte",
) -> Path:
    sample_codes = make_sample_codes(
        codebook_size=metadata.codebook_size,
        num_quantizers=metadata.num_quantizers,
        code_len=codes_len,
    )
    programs = {
        "decode_codes": export(
            module,
            (sample_codes,),
            strict=True,
        )
    }

    constant_methods = metadata.to_constant_methods()
    constant_methods["fixed_codes_len"] = int(codes_len)

    et_prog = lower_to_executorch(
        programs, constant_methods=constant_methods, backend=backend
    )
    model_path = output_dir / model_filename
    with model_path.open("wb") as f:
        et_prog.write_to_file(f)
    return model_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_dir = args.converted_dir.resolve()
    metadata_path = converted_dir / "decoder_metadata.json"
    metadata = DecoderExportMetadata.from_json(metadata_path)
    checkpoint_path = converted_dir / metadata.decoder_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    module = make_decode_export_module(
        metadata=metadata, checkpoint_path=checkpoint_path, dtype=dtype
    )

    if args.qlinear is not None:
        from executorch.extension.llm.export.quantize import quantize_model_

        quantize_model_(
            module,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qlinear_packing_format=args.qlinear_packing_format,
        )

    bucket_sizes = None
    if args.bucket_sizes is not None:
        bucket_sizes = sorted(int(s.strip()) for s in args.bucket_sizes.split(","))

    if bucket_sizes is not None:
        bucket_manifest = {
            "backend": args.backend,
            "dtype": args.dtype,
            "qlinear": args.qlinear,
            "qlinear_group_size": args.qlinear_group_size,
            "qlinear_packing_format": args.qlinear_packing_format,
            "source_converted_dir": str(converted_dir),
            "buckets": [],
        }
        for size in bucket_sizes:
            print(f"\n--- Exporting bucket: codes_len={size} ---")
            filename = f"model_{size}.pte"
            model_path = export_single_bucket(
                module, metadata, size, args.backend, output_dir, filename
            )
            bucket_manifest["buckets"].append({
                "codes_len": size,
                "model_path": str(model_path),
                "model_filename": filename,
            })
            print(f"Saved model: {model_path}")

        manifest_path = output_dir / "export_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(bucket_manifest, f, indent=2, sort_keys=True)
        print(f"\nSaved manifest: {manifest_path}")
        print(f"Exported {len(bucket_sizes)} buckets: {bucket_sizes}")
    else:
        model_path = export_single_bucket(
            module, metadata, args.fixed_codes_len, args.backend, output_dir
        )

        export_manifest = {
            "backend": args.backend,
            "dtype": args.dtype,
            "qlinear": args.qlinear,
            "qlinear_group_size": args.qlinear_group_size,
            "qlinear_packing_format": args.qlinear_packing_format,
            "fixed_codes_len": args.fixed_codes_len,
            "source_converted_dir": str(converted_dir),
            "model_path": str(model_path),
            "constant_methods": metadata.to_constant_methods(),
        }
        manifest_path = output_dir / "export_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(export_manifest, f, indent=2, sort_keys=True)

        print(f"Saved model: {model_path}")
        print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
