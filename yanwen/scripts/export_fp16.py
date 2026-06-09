#!/usr/bin/env python3
"""
FP16 Llama 3.1 8B -> Vulkan .pte, memory-frugal model.half() path.

Why a custom script (not export/export.py): the export_llm CLI upcasts the bf16
checkpoint to fp32 (16->32 GB) and torch.export peaks ~44.6 GB > 45 GB box RAM ->
global OOM. The meta-device construct + mmap + .half() path here keeps peak ~16 GB.

Storage (texture vs buffer) is chosen by the ET_VK_FORCE_BUFFER env (Plan A / A2):
  unset            -> texture PTE (default ExecuTorch; coopmat can't fire) -> *_fp16_texture.pte
  ET_VK_FORCE_BUFFER=1 -> buffer PTE (coopmat-eligible)                    -> *_fp16_buffer.pte

No op_registry edit needed — VulkanPartitioner reads the env and sets
storage_type_override=BUFFER, and the (fixed) TagMemoryMetaPass honors it graph-wide.

NOTE: full fp16 (16 GB) does NOT fit the phone (11.4 GB). Run fp16 on a discrete GPU,
or set N_LAYERS to a small subset (env N_LAYERS=8) just to observe shader dispatch.

Env knobs:
  ET_VK_FORCE_BUFFER=1   buffer PTE (else texture)
  ET_VK_DISABLE_COOPMAT  (runtime only; irrelevant at export — set it when RUNNING the buffer PTE)
  N_LAYERS=<int>         layer subset (default 32 = full)
  SEQ_LEN=<int>          export seq len (default 128)
"""

import gc
import json
import os
import time
from pathlib import Path

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

WEIGHTS_DIR = Path("/local/yanwen.xu/models/llama3_1_8b/original")
CKPT = WEIGHTS_DIR / "consolidated.00.pth"
PARAMS = WEIGHTS_DIR / "params.json"
PTE_OUT = Path("/local/yanwen.xu/workspace/pte_out")  # single source of truth for PTEs

N_LAYERS = int(os.environ.get("N_LAYERS", "32"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))
STORAGE = "buffer" if os.environ.get("ET_VK_FORCE_BUFFER") else "texture"


def main():
    suffix = "" if N_LAYERS == 32 else f"_{N_LAYERS}L"
    out = PTE_OUT / f"llama3_1_8b_fp16_{STORAGE}{suffix}.pte"
    print(f"[export] storage={STORAGE} n_layers={N_LAYERS} seq_len={SEQ_LEN} -> {out}")

    with open(PARAMS) as f:
        params = json.load(f)
    if N_LAYERS != 32:
        params["n_layers"] = N_LAYERS

    model_args = ModelArgs(max_seq_len=SEQ_LEN + 16, max_context_len=SEQ_LEN + 16, **params)

    print("[export] constructing transformer on meta device")
    with torch.device("meta"):
        model = construct_transformer(model_args)

    print(f"[export] mmap-loading checkpoint {CKPT}")
    t0 = time.perf_counter()
    checkpoint = torch.load(CKPT, map_location="cpu", mmap=True)  # noqa: TOR102
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    print(f"[export] checkpoint open in {time.perf_counter()-t0:.1f}s")

    model.load_state_dict(checkpoint, strict=False, assign=True)
    model = model.half().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] params: {n_params/1e9:.2f}B fp16 ({n_params*2/1e9:.1f} GiB)")

    example_inputs = (torch.randint(0, model_args.vocab_size, (1, SEQ_LEN), dtype=torch.int64),)
    print("[export] torch.export(strict=False)")
    t0 = time.perf_counter()
    with torch.no_grad():
        prog = export(model, example_inputs, strict=False)
    print(f"[export] torch.export done in {time.perf_counter()-t0:.1f}s")

    del model, checkpoint
    gc.collect()

    print("[export] to_edge_transform_and_lower")
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        prog,
        compile_config=EdgeCompileConfig(_skip_dim_order=False),
        partitioner=[VulkanPartitioner({})],  # honors ET_VK_FORCE_BUFFER
    )
    et = edge.to_executorch()
    print(f"[export] lowered in {time.perf_counter()-t0:.1f}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(et.buffer)
    print(f"[export] DONE. {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
