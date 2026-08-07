#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import argparse
import operator
import tempfile
from pathlib import Path
from typing import Any

import torch

from executorch.examples.models.gemma4.eagle_webgpu_round import (
    compose_k2_round_program,
)
from executorch.examples.models.gemma4.export_assistant_webgpu_artifacts import (
    load_qat_assistant,
    validate_assistant_checkpoint,
)
from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    finalize_mtp_export,
    validate_export_identity,
    WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES,
)
from executorch.examples.models.gemma4.webgpu_partitioner import (
    build_webgpu_partitioner,
    build_webgpu_transform_passes,
    rewrite_certified_unique_scatter,
)


def _load_target(
    checkpoint: Path,
    *,
    max_seq_len: int,
    text_quantize: str,
    group_size: int,
) -> Any:
    from executorch.examples.models.gemma4.quant_utils import (
        apply_embedding_quantization,
        apply_linear_quantization,
        parse_quantize,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
        Gemma4Config,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_model import Gemma4Model

    config = Gemma4Config.from_config("e2b")
    config.use_kv_cache = True
    config.max_seq_len = max_seq_len
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = True
    model = Gemma4Model(
        config=config,
        checkpoint_path=str(checkpoint.resolve()),
        dtype=torch.float32,
    ).get_eager_model()
    linear_quant, embedding_quant = parse_quantize(text_quantize)
    if embedding_quant:
        model = apply_embedding_quantization(model, embedding_quant).eval()
    if linear_quant:
        model = apply_linear_quantization(
            model, linear_quant, group_size=group_size
        ).eval()
    return model.eval()


def build_k2_round_program(
    target_checkpoint: Path,
    assistant_checkpoint: Path,
    *,
    max_seq_len: int = 8960,
    max_input_len: int = 512,
    text_quantize: str = "8da4w+emb4",
    assistant_quantize: str = "8da4w",
    assistant_lm_head_bits: int = 4,
    group_size: int = 128,
) -> torch.export.ExportedProgram:
    if max_seq_len < 514:
        raise ValueError("Gemma 4 MTP export requires max_seq_len >= 514")
    if max_input_len < 3 or max_input_len > max_seq_len:
        raise ValueError("invalid Gemma 4 MTP max_input_len")
    if text_quantize != "8da4w+emb4":
        raise ValueError("Gemma 4 MTP target requires 8da4w+emb4 quantization")
    if assistant_quantize != "8da4w" or assistant_lm_head_bits != 4:
        raise ValueError("Gemma 4 MTP assistant requires 8da4w with a 4-bit LM head")
    if group_size != 128:
        raise ValueError("Gemma 4 MTP export requires quantization group size 128")
    target_checkpoint_evidence = validate_export_identity(target_checkpoint)
    validate_assistant_checkpoint(assistant_checkpoint)
    target: Any = _load_target(
        target_checkpoint / "model.safetensors",
        max_seq_len=max_seq_len,
        text_quantize=text_quantize,
        group_size=group_size,
    )
    assistant: Any = load_qat_assistant(
        assistant_checkpoint,
        max_donor_len=max_seq_len,
        lm_head_bits=assistant_lm_head_bits,
        quantize_backbone=assistant_quantize,
    )
    text_model: Any = target.model
    program = compose_k2_round_program(
        text_model=text_model,
        embed_tokens=text_model.self_decoder.embed_tokens,
        assistant=assistant,
        hidden_size=text_model.config.hidden_size,
        max_seq_len=max_seq_len,
        embed_scale=text_model.self_decoder.embed_scale,
        max_input_len=max_input_len,
        max_donor_len=max_seq_len,
    )
    qat_selection_evidence = assistant._webgpu_qat_selection_evidence
    rewrites = rewrite_certified_unique_scatter(
        program,
        assistant.assistant.masked_embedding.token_ordering,
        expected_chains=2,
    )
    if rewrites != 2:
        raise ValueError(f"Gemma 4 MTP expected two scatter rewrites, found {rewrites}")
    k2_abi_evidence = program.graph_module.meta.get("gemma4K2Abi")
    if not isinstance(k2_abi_evidence, dict):
        raise ValueError("K=2 composition lacks exact ABI evidence")

    from executorch.exir.program._program import _transform
    from executorch.extension.llm.export.export_passes import (
        ReplaceSDPAWithCustomSDPAPass,
    )

    transformed = _transform(program, ReplaceSDPAWithCustomSDPAPass())
    transformed.graph_module.meta["gemma4QATSelectionEvidence"] = qat_selection_evidence
    transformed.graph_module.meta["gemma4K2Abi"] = k2_abi_evidence
    transformed.graph_module.meta["gemma4TargetCheckpointEvidence"] = dict(
        target_checkpoint_evidence
    )
    return transformed


def _lower_k2_round(
    program: torch.export.ExportedProgram,
    *,
    external_constants_max_data_bytes: int,
    text_quantize: str,
) -> tuple[Any, dict[str, object]]:
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import (
        ConstraintBasedSymShapeEvalPass,
    )

    compile_options = {
        "alias_buffer_mutations": True,
        "external_constants_max_data_bytes": external_constants_max_data_bytes,
        "require_dynamic_shapes": True,
    }
    mtp_transform_passes = build_webgpu_transform_passes(mode="mtp")
    edge = to_edge_transform_and_lower(
        {"k2_round": program.run_decompositions({})},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner={
            "k2_round": [
                build_webgpu_partitioner(
                    text_quantize=text_quantize,
                    mode="mtp",
                    compile_options=compile_options,
                )
            ]
        },
        transform_passes={"k2_round": mtp_transform_passes},
    )
    edge_census = getattr(mtp_transform_passes[0], "census", None)
    if not isinstance(edge_census, dict):
        raise ValueError("K=2 lowering did not record the MTP edge census")
    lowered = edge.exported_program("k2_round")
    delegate_count = 0
    portable_nodes: list[str] = []
    for node in lowered.graph.nodes:
        if node.op != "call_function" or node.target is operator.getitem:
            continue
        target = str(node.target)
        if "executorch_call_delegate" in target:
            delegate_count += 1
        else:
            portable_nodes.append(target)
    if delegate_count != 1 or portable_nodes:
        raise ValueError(
            "K=2 lowering requires one delegate and no portable operators: "
            f"delegates={delegate_count}, portable={portable_nodes}"
        )
    executorch_program = edge.to_executorch(
        ExecutorchBackendConfig(
            external_constants=True,
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
    return executorch_program, {
        "delegate_count": delegate_count,
        "edge": edge_census,
        "portable_operator_count": len(portable_nodes),
    }


def _validate_output_paths(output_path: Path, receipt_path: Path) -> None:
    if output_path.suffix != ".pte":
        raise ValueError("Gemma 4 MTP output path must end in .pte")
    sealed_root = output_path.parent.resolve()
    resolved_receipt = receipt_path.resolve()
    try:
        resolved_receipt.relative_to(sealed_root)
    except ValueError:
        pass
    else:
        raise ValueError("Gemma 4 MTP receipt must be outside the sealed artifact root")


def export_speculative(  # noqa: C901
    target_checkpoint: Path,
    assistant_checkpoint: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    source_receipt_path: Path | None = None,
    max_seq_len: int = 8960,
    max_input_len: int = 512,
    text_quantize: str = "8da4w+emb4",
    assistant_quantize: str = "8da4w",
    assistant_lm_head_bits: int = 4,
    group_size: int = 128,
    external_constants_max_data_bytes: int = (WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES),
) -> Path:
    _validate_output_paths(output_path, receipt_path)
    if (max_seq_len, max_input_len) != (8960, 512):
        raise ValueError("production Gemma 4 MTP export requires P512/ctx8960")
    if external_constants_max_data_bytes != WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES:
        raise ValueError(
            "production Gemma 4 MTP export requires the reviewed PTD split"
        )
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"refusing to overwrite existing artifact: {output_path}")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError(f"refusing to overwrite existing artifact: {receipt_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    assistant_checkpoint_evidence = validate_assistant_checkpoint(assistant_checkpoint)
    program = build_k2_round_program(
        target_checkpoint,
        assistant_checkpoint,
        max_seq_len=max_seq_len,
        max_input_len=max_input_len,
        text_quantize=text_quantize,
        assistant_quantize=assistant_quantize,
        assistant_lm_head_bits=assistant_lm_head_bits,
        group_size=group_size,
    )
    qat_selection_evidence = program.graph_module.meta.get("gemma4QATSelectionEvidence")
    if not isinstance(qat_selection_evidence, dict):
        raise ValueError("K=2 export lacks QAT selection evidence")
    target_checkpoint_evidence = program.graph_module.meta.get(
        "gemma4TargetCheckpointEvidence"
    )
    if not isinstance(target_checkpoint_evidence, dict):
        raise ValueError("K=2 export lacks target checkpoint evidence")
    k2_abi_evidence = program.graph_module.meta.get("gemma4K2Abi")
    if not isinstance(k2_abi_evidence, dict):
        raise ValueError("K=2 export lacks exact ABI evidence")
    executorch_program, lowering_evidence = _lower_k2_round(
        program,
        external_constants_max_data_bytes=external_constants_max_data_bytes,
        text_quantize=text_quantize,
    )
    tensor_tags = sorted(executorch_program._tensor_data)
    if not tensor_tags:
        raise ValueError("K=2 export did not produce external tensor data")
    for tag in tensor_tags:
        if not tag or tag in {".", ".."} or "/" in tag or "\\" in tag:
            raise ValueError(f"invalid external tensor-data tag: {tag!r}")
        destination = output_path.parent / f"{tag}.ptd"
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"refusing to overwrite existing artifact: {destination}")

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}.", dir=output_path.parent.parent
    ) as staging_directory:
        staging = Path(staging_directory)
        staged_pte = staging / output_path.name
        with staged_pte.open("xb") as output:
            executorch_program.write_to_file(output)
        executorch_program.write_tensor_data_to_file(str(staging))

        staged_tensor_paths: list[Path] = []
        for tag in tensor_tags:
            path = staging / f"{tag}.ptd"
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing external tensor data: {path.name}")
            staged_tensor_paths.append(path)
        if staged_pte.stat().st_size == 0:
            raise ValueError("K=2 export produced an empty PTE")

        evidence = {
            "assistant_checkpoint": assistant_checkpoint_evidence,
            "k2_abi": k2_abi_evidence,
            "lowering": lowering_evidence,
            "qat_selection": qat_selection_evidence,
            "target_checkpoint": target_checkpoint_evidence,
        }
        return finalize_mtp_export(
            staging,
            output_path,
            receipt_path,
            staged_pte,
            staged_tensor_paths,
            source_receipt_path,
            evidence,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Gemma 4 K=2 for WebGPU")
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--assistant-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=8960)
    parser.add_argument("--max-input-len", type=int, default=512)
    args = parser.parse_args()
    export_speculative(
        args.target_checkpoint,
        args.assistant_checkpoint,
        args.output,
        args.receipt,
        source_receipt_path=args.source_receipt,
        max_seq_len=args.max_seq_len,
        max_input_len=args.max_input_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
