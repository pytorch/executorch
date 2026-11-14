# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Gradium's Phonon text-to-speech model to one ExecuTorch program.

The .pte holds four methods (text_embed, voice_embed, flow_lm, mimi_decode)
and constant metadata methods. The I/O contract is the same for every backend
and is also written to phonon.json next to the program.

  python -m executorch.examples.models.phonon.export_phonon \\
      --backend xnnpack --output-dir phonon_exports/xnnpack
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from executorch.examples.models.phonon.model import (
    build_programs,
    dynamic_shapes,
    example_inputs,
    io_contract,
    mimi_state_spec,
    Phonon,
    program_metadata,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export

HF_MODEL = "gradium/phonon-31m-en"
HF_REVISION = "52433c46975bfa13144c56204c9f4ee1c5b148aa"
CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "default-voice.safetensors",
)
BACKENDS = ("portable", "xnnpack", "webgpu", "vulkan", "cuda", "mlx")
# Methods with a dynamic sequence length.
DYNAMIC_METHODS = ("text_embed", "voice_embed", "flow_lm")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# Backends that support --dtype bf16.
BF16_BACKENDS = ("portable", "cuda", "mlx")
# The linear quantization scheme of each --quantize level, per backend: 8da is
# dynamically quantized 8-bit activations, 4w and 8w quantize the weights only.
# CUDA 4w is the packed int4 kernel, which takes BF16 weights; MLX 4w is
# asymmetric (see quantize_asymmetric_int4).
QUANTIZE_SCHEMES = {
    "xnnpack": {"4bit": "8da4w", "8bit": "8da8w"},
    "cuda": {"4bit": "4w", "8bit": "8w"},
    "vulkan": {"4bit": "4w"},
    "webgpu": {"4bit": "4w"},
    "mlx": {"4bit": "4w", "8bit": "8w"},
}


def attention_for(backend: str) -> str:
    """KV cache / attention implementation of the flow LM per backend (see
    FlowLMStep): the ExecuTorch custom op where a kernel exists for it, plain
    SDPA with an explicit mask for the compiler backends."""
    return "sdpa" if backend in ("cuda", "mlx") else "custom_sdpa"


def source_transforms_for(backend: str, webgpu_compat_ops: bool):
    """Module swaps applied before export (see source_transformations.py)."""
    from executorch.examples.models.phonon import source_transformations as st

    if webgpu_compat_ops:
        if backend != "webgpu":
            raise ValueError("--webgpu-compat-ops only applies to --backend webgpu")
        return [st.use_webgpu_compat_ops]
    return {
        "xnnpack": [st.depthwise_conv_transpose_as_mul, st.conv_transpose1d_as_conv1d],
        "mlx": [st.decompose_elu],
        "vulkan": [
            st.decompose_elu,
            st.depthwise_conv_transpose_as_mul,
            st.conv_transpose1d_as_2d,
        ],
    }.get(backend, [])


def quantize_scheme(
    backend: str, quantize: str | None, dtype: str | None
) -> tuple[str | None, str]:
    """The linear quantization scheme and the dtype for `--quantize` and
    `--dtype` on `backend`. Raises ValueError with what to run instead when the
    backend has no kernel for them."""
    if dtype == "bf16" and backend not in BF16_BACKENDS:
        raise ValueError(
            f"--dtype bf16 is available for --backend {', '.join(BF16_BACKENDS)}. "
            f"Export --backend {backend} without --dtype (FP32)."
        )
    if quantize is None:
        return None, dtype or "fp32"
    schemes = QUANTIZE_SCHEMES.get(backend, {})
    if quantize not in schemes:
        options = [f"with --quantize {q}" for q in schemes]
        options.append("without --quantize (full precision)")
        raise ValueError(
            f"--backend {backend} has no {quantize} linear kernel. Export "
            f"{' or '.join(options)} instead."
        )
    scheme = schemes[quantize]
    if backend == "cuda" and scheme == "4w":
        if dtype == "fp32":
            raise ValueError(
                "--backend cuda --quantize 4bit needs BF16 weights (the int4 "
                "kernel takes BF16). Drop --dtype fp32 and it uses BF16, or "
                "export without --quantize for FP32."
            )
        if dtype is None:
            print("--backend cuda --quantize 4bit: using --dtype bf16")
        dtype = "bf16"
    return scheme, dtype or "fp32"


def quantize_asymmetric_int4(module: torch.nn.Module, group_size: int) -> None:
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
    from torchao.quantization.quant_primitives import MappingType

    quantize_(
        module,
        IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(group_size),
            mapping_type=MappingType.ASYMMETRIC,
        ),
        filter_fn=lambda m, _: isinstance(m, torch.nn.Linear)
        and m.in_features % group_size == 0,
    )


def partitioners_for(backend: str, methods, quantized: bool):
    """Per-method partitioners, transform passes and graph decompositions for
    one backend. Returns (partitioners, transform_passes, decompositions)."""
    if backend == "portable":
        return {m: [] for m in methods}, None, None
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        parts = {m: [XnnpackPartitioner()] for m in methods}
        if quantized:
            # --quantize quantizes the flow LM only.
            parts["flow_lm"].insert(0, XnnpackDynamicallyQuantizedPartitioner())
        return parts, None, None
    if backend in ("vulkan", "webgpu"):
        if backend == "vulkan":
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
                VulkanPartitioner as Partitioner,
            )
        else:
            from executorch.backends.webgpu.partitioner import (
                WebGPUPartitioner as Partitioner,
            )
        parts = {}
        for m in methods:
            options = {"force_fp16": False}
            if m in DYNAMIC_METHODS:
                options["require_dynamic_shapes"] = True
            parts[m] = [Partitioner(compile_options=options)]
        if backend == "webgpu":
            # The WebGPU runtime does not resize an embedding output to a
            # dynamic token count yet; the lookup runs on the CPU kernels.
            parts["text_embed"] = []
        return parts, None, None
    if backend == "cuda":
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from torch._inductor.decomposition import conv1d_to_conv2d

        parts = {
            m: [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(m)])]
            for m in methods
        }
        return parts, None, {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
    if backend == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes

        return {m: [MLXPartitioner()] for m in methods}, get_default_passes(), None
    raise ValueError(f"Unknown backend: {backend}")


def resolve_checkpoint(args) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            args.hf_model, revision=args.revision, allow_patterns=list(CHECKPOINT_FILES)
        )
    )


def delegation_summary(edge) -> dict[str, Any]:
    from executorch.devtools.backend_debug import get_delegation_info

    summary = {}
    for name in edge.methods:
        info = get_delegation_info(edge.exported_program(name).graph_module)
        cpu_ops = sorted(
            op
            for op, b in info.delegation_by_operator.items()
            if b.non_delegated
            and op not in ("getitem", "sym_size", "_local_scalar_dense")
        )
        summary[name] = {
            "delegates": info.num_delegated_subgraphs,
            "delegated_nodes": info.num_delegated_nodes,
            "non_delegated_ops": cpu_ops,
        }
    return summary


def lower_phonon(
    model: Phonon,
    backend: str = "xnnpack",
    quantize: str | None = None,
    quantize_group_size: int | None = None,
    webgpu_compat_ops: bool = False,
    dtype: str | None = None,
):
    """Export and lower the four programs of `model` for `backend`.
    Returns (ExecutorchProgramManager, delegation summary, metadata)."""
    from executorch.examples.models.phonon.source_transformations import (
        compute_weights_in,
    )

    cfg = model.cfg
    qlinear, dtype = quantize_scheme(backend, quantize, dtype)
    packing = "tile_packed_to_4d" if backend == "cuda" and qlinear == "4w" else None
    # Before the module swaps, which change the model's modules in place.
    metadata = program_metadata(model)
    programs = build_programs(model, attention_for(backend))
    for transform in source_transforms_for(backend, webgpu_compat_ops):
        print(f"{transform.__name__}: {transform(programs)}")
    if dtype == "bf16":
        print(
            f"compute_weights_in(bf16): {compute_weights_in(programs, DTYPES[dtype])}"
        )
    if qlinear:
        from executorch.extension.llm.export.quantize import quantize_model_

        # Weight quantization applies to the transformer and flow net of the
        # flow LM. The codec, the input projection and the one-output EOS head
        # are not quantized. The int4 packing for CUDA runs on the GPU.
        flow_lm = programs["flow_lm"].to("cuda" if packing else "cpu")
        for module in (flow_lm.layers, flow_lm.flow_net):
            if backend == "mlx" and qlinear == "4w":
                # Symmetric int4 delays the end-of-speech decision (long
                # pauses); asymmetric int4 keeps the timing of FP32.
                quantize_asymmetric_int4(module, quantize_group_size or 32)
                continue
            quantize_model_(
                module,
                qlinear_config=qlinear,
                qlinear_group_size=quantize_group_size,
                qlinear_packing_format=packing,
                skip_incompatible_shapes=True,
            )

    inputs, shapes = example_inputs(cfg), dynamic_shapes(cfg)
    if packing:
        inputs["flow_lm"] = tuple(t.to("cuda") for t in inputs["flow_lm"])
    partitioners, passes, decompositions = partitioners_for(
        backend, list(programs), quantized=bool(qlinear)
    )
    eps = {}
    with torch.no_grad():
        for name, program in programs.items():
            ep = export(
                program, inputs[name], dynamic_shapes=shapes[name], strict=False
            )
            if decompositions:
                ep = ep.run_decompositions(decompositions)
            eps[name] = ep
    edge = to_edge_transform_and_lower(
        eps,
        partitioner=partitioners,
        transform_passes={m: passes for m in eps} if passes else None,
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


def export_phonon(
    checkpoint_dir: Path,
    output_dir: Path,
    backend: str = "xnnpack",
    quantize: str | None = None,
    quantize_group_size: int | None = None,
    webgpu_compat_ops: bool = False,
    dtype: str | None = None,
) -> Path:
    """Export a checkpoint directory to output_dir/phonon.pte (plus
    aoti_cuda_blob.ptd for CUDA), phonon.json, the tokenizer and the default
    voice."""
    qlinear, dtype = quantize_scheme(backend, quantize, dtype)
    model = Phonon.from_checkpoint(checkpoint_dir / "model.safetensors")
    cfg = model.cfg
    program, delegation, metadata = lower_phonon(
        model, backend, quantize, quantize_group_size, webgpu_compat_ops, dtype
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pte = output_dir / "phonon.pte"
    with pte.open("wb") as f:
        program.write_to_file(f)
    data_file = None
    if backend == "cuda":
        program.write_tensor_data_to_file(str(output_dir))
        data_file = "aoti_cuda_blob.ptd"
    for name in ("tokenizer.json", "tokenizer.model", "default-voice.safetensors"):
        if (checkpoint_dir / name).exists():
            shutil.copyfile(checkpoint_dir / name, output_dir / name)

    contract = {
        "format_version": metadata["format_version"],
        "backend": backend,
        "dtype": dtype,
        "quantize": quantize,
        "qlinear": qlinear,
        "webgpu_compat_ops": webgpu_compat_ops,
        "files": {
            "program": pte.name,
            "data": data_file,
            "tokenizer": "tokenizer.json",
            "default_voice": "default-voice.safetensors",
        },
        "constants": {k: v for k, v in metadata.items() if not torch.is_tensor(v)},
        "constant_tensors": {"bos_embeds": [1, 1, cfg.dim]},
        "dynamic": {"S": [1, cfg.max_prefill]},
        "methods": io_contract(cfg),
        "mimi_state": [[n, list(s)] for n, s in mimi_state_spec(cfg)],
        "delegation": delegation,
    }
    (output_dir / "phonon.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(delegation, indent=2))
    print(f"Saved {pte} ({pte.stat().st_size / 2**20:.1f} MiB)")
    return pte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="xnnpack")
    parser.add_argument("--output-dir", type=Path, default=Path("phonon_exports"))
    parser.add_argument(
        "--checkpoint-dir",
        help="Local directory with model.safetensors, config.json, tokenizer.json "
        "(default: download --hf-model at --revision)",
    )
    parser.add_argument("--hf-model", default=HF_MODEL)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument(
        "--quantize",
        choices=("4bit", "8bit"),
        help="Quantize the flow LM's linear layers with the scheme of the backend "
        "(see QUANTIZE_SCHEMES; default: none)",
    )
    parser.add_argument("--quantize-group-size", type=int)
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        help="Weight and matmul dtype (default: fp32, or bf16 for --backend cuda "
        "--quantize 4bit); inputs, outputs and state stay float32",
    )
    parser.add_argument(
        "--webgpu-compat-ops",
        action="store_true",
        help="Replace layer_norm, elu, silu and conv_transpose1d with equivalent "
        "ops for WebGPU runtimes built without those kernels",
    )
    args = parser.parse_args()
    # Before the download, so that an unsupported combination fails first.
    try:
        _, dtype = quantize_scheme(args.backend, args.quantize, args.dtype)
    except ValueError as e:
        parser.error(str(e))
    export_phonon(
        resolve_checkpoint(args),
        args.output_dir,
        backend=args.backend,
        quantize=args.quantize,
        quantize_group_size=args.quantize_group_size,
        webgpu_compat_ops=args.webgpu_compat_ops,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
