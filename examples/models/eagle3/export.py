# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export an EAGLE-3 speculator (a registered target + draft head) to one .pte.

Three methods are lowered together so they share mutable state:
  - "prefill":       target prompt prefill (T in [get_min_prefill_chunk,
                     get_max_prefill_chunk]) -> next token + fused feature.
  - "target_verify": target forward over the candidate chain (static T=chain+1)
                     -> per-position greedy ids + fused feature.
  - "draft_decode":  draft proposal over its KV cache (T>=1; seed with T>1, step
                     with T=1) -> proposed target ids + recurrent feature.

prefill and target_verify share the target's KV cache; draft_decode uses the
draft's KV cache. ``share_mutable_buffers`` deduplicates each by FQN, so a single
allocation backs each cache across the methods that touch it.

A standalone single-token target ``decode`` is intentionally not exported. Under
the shifted (vLLM-EAGLE) runner scheme the draft pairs target hidden_state_t with
token_{t+1}, so after verification the next draft chain reseeds from the
``feature`` ``target_verify`` already produced for the accepted positions — the
corrected/bonus token never needs its own target forward. So prefill +
target_verify + draft_decode are sufficient for multi-round decoding
(``test_shifted_speculative_decode_is_lossless`` drives the full loop through only
these three methods).

Export runs with the model on the host (CPU); AOTInductor streams weights to the
GPU per kernel during compilation, so peak GPU memory stays low even for the INT4
31B target. The target is loaded from a prequantized (INT4) directory and the
draft from a vLLM-speculator checkpoint; only the CUDA (AOTI) backend is
supported.

Scope (this is a fixed-shape ExecuTorch artifact, not a generic EAGLE runtime):
chain length, the chain_len+1 verify window, the prefill/draft dynamic ranges,
the CUDA backend, and the small-M INT4 dispatch policy are all baked at export —
vary the target, chain length, or backend by re-exporting. The caller is
responsible for pairing a target, draft, and tokenizer that were trained
together: only target/draft hidden size is checked here; tokenizer identity,
target vocab size, the d2t/t2d mapping, the tap-layer convention, and the draft's
training target are NOT validated, and a mismatch can pass export yet silently
degrade acceptance or correctness. A versioned target/draft/tokenizer manifest +
runtime validation is left as future work.
"""

import argparse
import gc
import os

import torch
import torch.nn as nn

from executorch.examples.models.eagle3.draft import Eagle3Draft
from executorch.examples.models.eagle3.speculator import Eagle3Speculator
from executorch.examples.models.eagle3.target import TARGETS

# Route the static chain_len+1 verify forward to the small-M INT4 GEMM. Must be
# <= the shim's GEMM_MAX_M (8 in int4_plain_mm.cuh) and >= the largest chain+1.
# Set locally on int4_dispatch (not the global default) so other models' exports
# keep MATVEC_MAX_M=4 and their dynamic prefill ranges are unaffected.
_MATVEC_MAX_M = 8


# Thin per-method modules: torch.export traces ``forward``, so each method of the
# shared speculator is exposed as its own module. All wrap the *same* spec
# instance, so their captured buffers share FQNs and are deduplicated on lower.


class _Prefill(nn.Module):
    def __init__(self, spec: Eagle3Speculator):
        super().__init__()
        self.spec = spec

    def forward(self, tokens, input_pos):
        return self.spec.prefill(tokens, input_pos)


class _TargetVerify(nn.Module):
    def __init__(self, spec: Eagle3Speculator):
        super().__init__()
        self.spec = spec

    def forward(self, tokens, input_pos, kv_window):
        # kv_window length = number of valid KV positions; its dynamic dim is a
        # backed SymInt that bounds the mid-M SDPA key loop (ignored if mid-M is
        # off). Only its shape matters, not its contents.
        return self.spec.target_verify(tokens, input_pos, kv_window)


class _DraftDecode(nn.Module):
    def __init__(self, spec: Eagle3Speculator):
        super().__init__()
        self.spec = spec

    def forward(self, tokens, feature, input_pos):
        return self.spec.draft_decode(tokens, feature, input_pos)


def _export_cuda(
    spec: Eagle3Speculator,
    output_dir: str,
    max_prefill: int,
    chain_len: int,
    prefill_min: int,
) -> None:
    import torch._inductor.config as inductor_config

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"

    import time

    _t = [time.time()]

    def _lap(msg: str) -> None:
        now = time.time()
        print(f"[export +{now - _t[0]:6.1f}s] {msg}", flush=True)
        _t[0] = now

    # Register Int4Tensor dispatch -> executorch_cuda::int4_plain_mm for the
    # target. main() sets MATVEC_MAX_M (and restores it) around this call.
    import executorch.backends.cuda.int4_dispatch as int4_dispatch

    target_config = spec.target.config
    hidden = spec.draft.config.hidden_size
    draft_vocab_size = spec.draft.config.draft_vocab_size
    # Verify re-feeds the last confirmed token (its logits are the folded bonus)
    # plus the K proposals: a fixed chain_len+1 window in one target forward. With
    # chain_len+1 <= MATVEC_MAX_M the verify forward stays on the small-M GEMM
    # rather than the dequant path.
    verify_len = chain_len + 1
    # prefill's dynamic length must take a single INT4 dispatch branch over its
    # whole range: the target may specialize a lower bound (prefill_min), and the
    # dispatch branches at M = MATVEC_MAX_M, so a long-prefill (dequant) export
    # needs min > MATVEC_MAX_M.
    target_min = max(prefill_min, int4_dispatch.MATVEC_MAX_M + 1)

    # Export on the host: weights stay on CPU, AOTI streams them to the GPU per
    # kernel, so the INT4 target's dequant during codegen never piles up on-device.
    print(f"Exporting prefill (T in [{target_min}, {max_prefill}])...")
    prefill_dim = Dim("prefill_len", min=target_min, max=max_prefill)
    with torch.no_grad():
        prefill_ep = export(
            _Prefill(spec),
            (
                torch.zeros((1, max_prefill), dtype=torch.long),
                torch.arange(max_prefill, dtype=torch.long),
            ),
            dynamic_shapes=({1: prefill_dim}, {0: prefill_dim}),
            strict=True,
        )
    _lap("export prefill")

    print(f"Exporting target_verify (T = {verify_len})...")
    # The mid-M SDPA key bound is the dynamic length of kv_window: valid KV
    # positions = anchor_pos + chain + 1, in [verify_len, max_seq_len].
    kv_dim = Dim("kv_len", min=verify_len, max=target_config.max_seq_len)
    with torch.no_grad():
        verify_ep = export(
            _TargetVerify(spec),
            (
                torch.zeros((1, verify_len), dtype=torch.long),
                torch.arange(verify_len, dtype=torch.long),
                torch.zeros((8 * verify_len,), dtype=torch.int32),
            ),
            dynamic_shapes=({}, {}, {0: kv_dim}),
            strict=True,
        )
    _lap("export target_verify")

    # draft_decode: T>1 seeds the draft KV (prompt / newly confirmed tokens), T=1
    # steps the chain. The feature is hidden-size for both (fused target feature
    # or recurrent g).
    # The draft seeds with up to max_prefill tokens (prompt) and reseeds with up
    # to chain_len+1 confirmed tokens per round, so the dynamic max must cover both.
    draft_max = max(max_prefill, verify_len)
    print(f"Exporting draft_decode (T in [1, {draft_max}])...")
    draft_dim = Dim("draft_len", min=1, max=draft_max)
    with torch.no_grad():
        draft_ep = export(
            _DraftDecode(spec),
            (
                torch.zeros((1, draft_max), dtype=torch.long),
                torch.zeros((1, draft_max, hidden), dtype=torch.bfloat16),
                torch.arange(draft_max, dtype=torch.long),
            ),
            dynamic_shapes=({1: draft_dim}, {1: draft_dim}, {0: draft_dim}),
            strict=True,
        )
    _lap("export draft_decode")

    del spec
    gc.collect()

    def _partitioner(name: str):
        return [
            CudaPartitioner(
                [
                    CudaBackend.generate_method_name_compile_spec(name),
                    CompileSpec("low_memory_mode", b"ON"),
                ]
            )
        ]

    print("Lowering to ExecuTorch with CUDA backend...")
    et_prog = to_edge_transform_and_lower(
        {
            "prefill": prefill_ep,
            "target_verify": verify_ep,
            "draft_decode": draft_ep,
        },
        partitioner={
            "prefill": _partitioner("prefill"),
            "target_verify": _partitioner("target_verify"),
            "draft_decode": _partitioner("draft_decode"),
        },
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods={
            "get_max_seq_len": target_config.max_seq_len,
            "get_vocab_size": target_config.vocab_size,
            "get_n_layers": target_config.num_hidden_layers,
            "get_max_prefill_chunk": max_prefill,
            "get_min_prefill_chunk": target_min,
            "get_chain_len": chain_len,
            "get_draft_vocab_size": draft_vocab_size,
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": False,
            "enable_dynamic_shape": True,
        },
    )
    del prefill_ep, verify_ep, draft_ep
    gc.collect()
    _lap("to_edge_transform_and_lower (AOTI compile)")

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            emit_mutable_buffer_names=True,
        ),
    )
    del et_prog
    gc.collect()
    _lap("to_executorch")

    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    print(f"  {os.path.getsize(pte_path) / 1024**2:.1f} MB")
    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  Saved tensor data (.ptd) to {output_dir}/")
    _lap("write .pte + .ptd")
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Export an EAGLE-3 speculator to .pte.")
    p.add_argument(
        "--target-model",
        default="gemma4_31b",
        choices=list(TARGETS),
        help="Registered target model (see eagle3/target.py).",
    )
    p.add_argument(
        "--target", required=True, help="Prequantized (INT4) target directory."
    )
    p.add_argument("--draft", required=True, help="EAGLE-3 draft head directory.")
    p.add_argument("--output-dir", default="./eagle3_exports")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument(
        "--max-prefill",
        type=int,
        default=512,
        help="Max prefill length: AOTI compiles prefill kernels for up to this T "
        "and the whole prompt must fit in one prefill (the runner does not chunk). "
        "Smaller compiles faster.",
    )
    p.add_argument(
        "--chain", type=int, default=4, help="Draft chain length K (verify K+1)."
    )
    p.add_argument(
        "--no-midm-sdpa",
        action="store_true",
        help="Disable the length-bounded mid-M SDPA kernel for target_verify "
        "(it accelerates full-attention layers at long context).",
    )
    args = p.parse_args()

    spec_t = TARGETS[args.target_model]
    if not torch.cuda.is_available():
        p.error("CUDA is required to compile the EAGLE-3 export.")

    print(f"Loading {args.target_model} target from {args.target}...")
    target = spec_t.load(args.target, args.max_seq_len)

    # Route the target's full-attention layers' verify SDPA (M=chain+1) through
    # the length-bounded mid-M Triton kernel. Only affects target_verify (prefill
    # M is out of range, decode isn't exported); huge win at long context.
    if not args.no_midm_sdpa and hasattr(target, "set_midm_sdpa"):
        target.set_midm_sdpa(True)
        print("Enabled mid-M SDPA for target_verify.")

    print(f"Loading draft head from {args.draft}...")
    draft, _ = Eagle3Draft.from_checkpoint(
        args.draft, device="cpu", dtype=torch.bfloat16, max_seq_len=args.max_seq_len
    )
    if target.config.hidden_size != draft.config.target_hidden_size:
        p.error(
            f"target hidden_size {target.config.hidden_size} != draft "
            f"target_hidden_size {draft.config.target_hidden_size}"
        )
    # Cheap matched-pair guard: every draft id must map (target_id = draft_id +
    # d2t[draft_id]) into the target vocab. A wrong d2t / mismatched pair would
    # otherwise emit target ids outside the embedding range at runtime. This does
    # not validate tokenizer identity or tap convention (see the scope note above).
    target_ids = torch.arange(draft.d2t.numel(), device=draft.d2t.device) + draft.d2t
    if int(target_ids.min()) < 0 or int(target_ids.max()) >= target.config.vocab_size:
        p.error(
            f"draft d2t maps draft ids outside the target vocab "
            f"[0, {target.config.vocab_size}): got target id range "
            f"[{int(target_ids.min())}, {int(target_ids.max())}]; the draft and "
            f"target are likely not a matched pair"
        )

    spec = Eagle3Speculator(target, draft).eval()

    # A single target forward accepts min_forward_len .. max_forward_len tokens.
    max_forward = spec_t.max_forward_len(target.config)
    max_prefill = min(args.max_prefill, args.max_seq_len - 1, max_forward)
    # prefill's dynamic min (see _export_cuda target_min): the target's own
    # specialization (min_forward_len) and the INT4 dispatch (> MATVEC_MAX_M).
    prefill_min = max(spec_t.min_forward_len, _MATVEC_MAX_M + 1)
    if max_prefill < prefill_min:
        p.error(
            f"computed max_prefill={max_prefill} < {prefill_min}; raise "
            f"--max-prefill (got {args.max_prefill}) or --max-seq-len (got "
            f"{args.max_seq_len})"
        )
    # target_verify is a single static forward of chain+1 tokens: it must fit the
    # small-M GEMM (chain+1 <= _MATVEC_MAX_M) and the target's per-forward bounds
    # [min_forward_len, max_forward].
    verify_len = args.chain + 1
    if verify_len > _MATVEC_MAX_M:
        p.error(
            f"--chain {args.chain} (verify window {verify_len}) exceeds the "
            f"INT4 small-M GEMM limit {_MATVEC_MAX_M}"
        )
    if verify_len < spec_t.min_forward_len:
        p.error(
            f"--chain {args.chain} (verify window {verify_len}) is below the "
            f"target's minimum forward length {spec_t.min_forward_len}"
        )
    if verify_len > min(args.max_seq_len - 1, max_forward):
        p.error(
            f"--chain {args.chain} (verify window {verify_len}) exceeds the "
            f"target's per-forward limit {min(args.max_seq_len - 1, max_forward)}"
        )
    # Route the static chain_len+1 verify forward to the small-M INT4 GEMM by
    # raising the dispatch threshold for this export only; restore it so the
    # process-global default (4) is unchanged for any later use.
    import executorch.backends.cuda.int4_dispatch as int4_dispatch

    saved_threshold = int4_dispatch.MATVEC_MAX_M
    int4_dispatch.MATVEC_MAX_M = _MATVEC_MAX_M
    try:
        _export_cuda(
            spec,
            args.output_dir,
            max_prefill=max_prefill,
            chain_len=args.chain,
            prefill_min=spec_t.min_forward_len,
        )
    finally:
        int4_dispatch.MATVEC_MAX_M = saved_threshold


if __name__ == "__main__":
    main()
