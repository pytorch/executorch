# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Pocket TTS b6369a24 to one XNNPACK .pte with three methods."""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from executorch.examples.models.phonon.export_phonon import (
    delegation_summary,
    partitioners_for,
    quantize_scheme,
)
from executorch.examples.models.phonon.model import (
    dynamic_shapes,
    example_inputs,
    io_contract,
    mimi_state_spec,
    program_metadata,
)
from executorch.examples.models.phonon.source_transformations import (
    conv_transpose1d_as_conv1d,
    depthwise_conv_transpose_as_mul,
)
from executorch.examples.models.pocket_tts.model import (
    build_programs,
    load_voice,
    PocketTTS,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.llm.export.quantize import quantize_model_
from huggingface_hub import snapshot_download
from torch.export import export

logger: logging.Logger = logging.getLogger(__name__)

HF_MODEL = "kyutai/pocket-tts-without-voice-cloning"
HF_REVISION = "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
CHECKPOINT = "tts_b6369a24.safetensors"
VOICE = "embeddings/alba.safetensors"
BACKENDS = ("xnnpack",)


def lower_pocket_tts(
    model: PocketTTS,
    backend: str = "xnnpack",
    quantize: str | None = None,
    quantize_group_size: int | None = None,
):
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported Pocket TTS backend: {backend}")
    qlinear, _ = quantize_scheme(backend, quantize, None)
    cfg = model.cfg
    metadata = program_metadata(model)
    programs = build_programs(model)
    depthwise_conv_transpose_as_mul(programs)
    conv_transpose1d_as_conv1d(programs)
    if qlinear:
        flow_lm = programs["flow_lm"]
        for module in (flow_lm.layers, flow_lm.flow_net):
            quantize_model_(
                module,
                qlinear_config=qlinear,
                qlinear_group_size=quantize_group_size,
                skip_incompatible_shapes=True,
            )
    inputs, shapes = example_inputs(cfg), dynamic_shapes(cfg)
    with torch.no_grad():
        exported = {
            name: export(
                module, inputs[name], dynamic_shapes=shapes[name], strict=False
            )
            for name, module in programs.items()
        }
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=partitioners_for(backend, exported, quantized=bool(qlinear))[0],
        constant_methods=metadata,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    delegation = delegation_summary(edge)
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    return program, delegation, metadata


def export_pocket_tts(
    checkpoint_dir: Path,
    output_dir: Path,
    backend: str = "xnnpack",
    quantize: str | None = None,
    quantize_group_size: int | None = None,
) -> Path:
    quantize_scheme(backend, quantize, None)
    model = PocketTTS.from_checkpoint(checkpoint_dir / CHECKPOINT)
    cfg = model.cfg
    load_voice(checkpoint_dir / VOICE, cfg.dim)
    if not (checkpoint_dir / "tokenizer.model").is_file():
        raise FileNotFoundError(checkpoint_dir / "tokenizer.model")
    program, delegation, metadata = lower_pocket_tts(
        model, backend, quantize, quantize_group_size
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pocket_tts.pte"
    with path.open("wb") as output:
        program.write_to_file(output)
    shutil.copyfile(checkpoint_dir / "tokenizer.model", output_dir / "tokenizer.model")
    shutil.copyfile(checkpoint_dir / VOICE, output_dir / "alba.safetensors")
    contract = {
        "format_version": metadata["format_version"],
        "model": "pocket-tts-b6369a24",
        "backend": "xnnpack",
        "dtype": "fp32",
        "quantize": quantize,
        "qlinear": quantize_scheme(backend, quantize, None)[0],
        "quantize_group_size": quantize_group_size,
        "files": {
            "program": path.name,
            "tokenizer": "tokenizer.model",
            "default_voice": "alba.safetensors",
        },
        "constants": {k: v for k, v in metadata.items() if not torch.is_tensor(v)},
        "constant_tensors": {"bos_embeds": [1, 1, cfg.dim]},
        "methods": {k: v for k, v in io_contract(cfg).items() if k != "voice_embed"},
        "mimi_state": [[name, list(shape)] for name, shape in mimi_state_spec(cfg)],
        "delegation": delegation,
    }
    (output_dir / "pocket_tts.json").write_text(json.dumps(contract, indent=2) + "\n")
    logger.info(
        "Saved %s (%.1f MiB)\n%s", path, path.stat().st_size / 2**20, delegation
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="xnnpack")
    parser.add_argument(
        "--quantize",
        choices=("4bit", "8bit"),
        help="Quantize FlowLM linear layers (default: none)",
    )
    parser.add_argument(
        "--quantize-group-size",
        type=int,
        help="Weight group size (default: 32 for 4bit, 0/per-channel for 8bit)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pocket_tts_exports"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Local b6369a24 checkpoint, tokenizer.model and embeddings/alba.safetensors",
    )
    args = parser.parse_args()
    try:
        quantize_scheme(args.backend, args.quantize, None)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO)
    checkpoint_dir = args.checkpoint_dir or Path(
        snapshot_download(
            HF_MODEL,
            revision=HF_REVISION,
            allow_patterns=[CHECKPOINT, "tokenizer.model", VOICE],
        )
    )
    export_pocket_tts(
        checkpoint_dir,
        args.output_dir,
        args.backend,
        args.quantize,
        args.quantize_group_size,
    )


if __name__ == "__main__":
    main()
