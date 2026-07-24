# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LFM2.5-VL as a multi-method PTE for ExecuTorch with CUDA/AOTI backend.

All three methods are delegated to the CUDA backend.  Conv layer state is
threaded through attn_options as explicit IO; KV cache uses mark_static_address
so AOTI can trace through in-place mutations.

Methods (D = text hidden dim):
  vision_encoder  : [1, 3, 512, 512] f32 -> [1, 256, D] f32
  token_embedding : [1, seq_len] i64     -> [1, seq_len, D] f32
  text_decoder    : ([1, seq_len, D], [seq_len] i64) -> [1, vocab] f32

Usage:
    python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \\
        --model_dir LiquidAI/LFM2.5-VL-450M --dtype bf16
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
from torch.export import Dim, ExportedProgram
from torch.nn.attention import SDPBackend

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.examples.models.lfm2_5_vl.model import (
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    Lfm2p5VlModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Blackwell (sm_103) workaround: torch._inductor maps arch 103 -> "100f" but
# Triton generates PTX targeting sm_103a.  Patch to match.
# TODO: Remove once PyTorch bump includes the upstream fix in
# torch/_inductor/codegen/cuda/compile_utils.py
# ---------------------------------------------------------------------------
try:
    from torch._inductor.codecache import cuda_compile_utils

    _orig_nvcc_arch = cuda_compile_utils._nvcc_arch_as_compile_option

    def _patched_nvcc_arch() -> str:
        arch = cuda_compile_utils.cuda_env.get_cuda_arch()
        return "103a" if arch == "103" else _orig_nvcc_arch()

    cuda_compile_utils._nvcc_arch_as_compile_option = _patched_nvcc_arch
except (ImportError, AttributeError):
    pass

_CONFIG_DIR = Path(__file__).parent / "config"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _resolve_params_path(model_dir: str, params: str | None) -> str | None:
    if params is not None:
        return params
    name = model_dir.lower()
    if "450m" in name:
        return str(_CONFIG_DIR / "lfm2_5_vl_450m_config.json")
    if "1.6b" in name or "1_6b" in name:
        return str(_CONFIG_DIR / "lfm2_5_vl_1_6b_config.json")
    return None


# ---------------------------------------------------------------------------
# Per-method export
# ---------------------------------------------------------------------------


def _export_image_encoder(lfm2: torch.nn.Module, *, device: str) -> ExportedProgram:
    class _Encoder(torch.nn.Module):
        def __init__(self, lfm2: torch.nn.Module) -> None:
            super().__init__()
            self.lfm2 = lfm2

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.lfm2.image_embedding(images)

    example = torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        return torch.export.export(_Encoder(lfm2), (example,), strict=False)


def _export_text_decoder(lfm2: torch.nn.Module, *, dtype: torch.dtype, device: str) -> ExportedProgram:
    dim = lfm2.text_model_args.dim

    class _Decoder(torch.nn.Module):
        def __init__(self, text_model: torch.nn.Module) -> None:
            super().__init__()
            self.text_model = text_model

        def forward(self, embeddings: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
            out = self.text_model(None, {"input_pos": input_pos}, embeddings)
            if isinstance(out, tuple):
                out = out[0]
            return out.contiguous()

    seq = 8
    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN - 1)
    example_emb = torch.randn(1, seq, dim, dtype=dtype, device=device)
    example_pos = torch.arange(seq, dtype=torch.int64, device=device)

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        return torch.export._trace._export(
            _Decoder(lfm2.text_model),
            (example_emb, example_pos),
            dynamic_shapes=({1: token_dim}, {0: token_dim}),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def _export_token_embedding(lfm2: torch.nn.Module, *, device: str) -> ExportedProgram:
    embed = lfm2.model_.model.language_model.get_input_embeddings()
    token_dim = Dim("token_dim_1", min=1, max=MAX_SEQ_LEN)
    example = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64, device=device)
    with torch.no_grad():
        return torch.export.export(embed, (example,), dynamic_shapes=[{1: token_dim}], strict=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def export_all(
    model_dir: str,
    output: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = MAX_SEQ_LEN,
    params_path: str | None = None,
) -> None:
    logging.info("Loading %s...", model_dir)
    lfm2_model = Lfm2p5VlModel(
        model_dir=model_dir,
        max_seq_len=max_seq_len,
        max_context_len=max_seq_len,
        params_path=params_path,
        use_sdpa_with_kv_cache_op=False,
    )
    lfm2 = lfm2_model.get_eager_model().to(dtype=dtype, device="cuda")

    # Mark KV cache and conv state buffers as static addresses so AOTI can
    # trace through in-place mutations. Must be after .to("cuda") because
    # marking a CPU buffer that later gets replaced is a no-op.
    for module in lfm2.text_model.modules():
        for name, buf in module.named_buffers(recurse=False):
            if name in ("k_cache", "v_cache", "conv_state"):
                torch._dynamo.mark_static_address(buf)

    logging.info("[1/3] Vision encoder")
    vision_ep = _export_image_encoder(lfm2, device="cuda")
    logging.info("[2/3] Text decoder")
    decoder_ep = _export_text_decoder(lfm2, dtype=dtype, device="cuda")
    logging.info("[3/3] Token embedding")
    token_ep = _export_token_embedding(lfm2, device="cuda")

    programs = {"vision_encoder": vision_ep, "token_embedding": token_ep, "text_decoder": decoder_ep}
    partitioners = {
        k: [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(k)])]
        for k in programs
    }
    metadata = {
        "get_max_seq_len": lfm2.text_model_args.max_seq_len,
        "get_vocab_size": lfm2.text_model_args.vocab_size,
        "use_kv_cache": lfm2.text_model_args.use_kv_cache,
        "get_eos_ids": [7],
    }

    logging.info("Lowering to Edge IR + CUDA")
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=metadata,
    )

    logging.info("Finalizing ExecuTorch program")
    et_program = et_prog.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={k: ConstraintBasedSymShapeEvalPass() for k in programs},
        )
    )

    output_path = Path(output)
    output_dir = output_path.parent or Path(".")
    logging.info("Saving %s", output_path)
    with open(output_path, "wb") as f:
        et_program.write_to_file(f)
    et_program.write_tensor_data_to_file(str(output_dir))
    logging.info("Done — methods: %s", et_program.methods)


def main() -> None:
    parser = ArgumentParser(description="Export LFM2.5-VL to ExecuTorch (CUDA)")
    parser.add_argument("--model_dir", default="LiquidAI/LFM2.5-VL-450M")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--params", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    params_path = _resolve_params_path(args.model_dir, args.params)
    output = args.output or f"lfm2_5_vl_{args.dtype}_cuda.pte"

    export_all(args.model_dir, output, dtype=dtype, max_seq_len=args.max_seq_len, params_path=params_path)


if __name__ == "__main__":
    main()
