"""Six-stage validation for Gemma4 export pipeline.

Stages:
  1. Eager mode forward pass
  2. torch.export(decode, T=1)
  3. torch.export(prefill, T>=2 dynamic)
  4. Replay prefill_ep -> compare logits vs eager
  5. to_edge_transform_and_lower (CUDA backend)
  6. Load .pte + .ptd, run "prefill" -> compare logits vs eager

On any error: prints the full traceback and stops.

Usage:
  python inference.py --tiny-test
  python inference.py --model-dir /path/to/gemma-4-31B --output-dir /tmp/gemma4
"""

import argparse
import copy
import os
import sys
import tempfile
import traceback

import torch

from executorch.examples.models.gemma4.export import (
    build_tiny_model,
    export_two_methods,
    load_full_model,
    load_prequantized_model,
    lower_and_save,
    materialize_buffers,
)


# Test prompt from the spec: "The future of artificial intelligence is"
DEFAULT_PROMPT_IDS = [2, 1596, 2003, 576, 12500, 12175, 603]


def _reset_kv(model) -> None:
    """Zero out all KV-cache buffers in the model (between runs)."""
    for layer in model.layers:
        attn = layer.self_attn
        with torch.no_grad():
            attn.kv_cache.k_cache.zero_()
            attn.kv_cache.v_cache.zero_()


def _stage(idx: int, name: str):
    print(f"[STAGE {idx}/6] {name}...", end=" ", flush=True)


def _ok(extra: str = "") -> None:
    print("OK" + (f" ({extra})" if extra else ""))


def _fail_and_exit(idx: int, name: str, exc: BaseException) -> None:
    print("FAILED")
    print(f"--- Traceback for stage {idx}/6 ({name}) ---")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    print("--- End traceback ---")
    print("STOPPING. Please review the error above and advise on next steps.")
    sys.exit(1)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32).cpu()
    b32 = b.detach().to(torch.float32).cpu()
    return (a32 - b32).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description="Validate Gemma4 export pipeline")
    parser.add_argument("--tiny-test", action="store_true",
                        help="Use a small random-weight model (no checkpoint).")
    parser.add_argument("--model-dir", default=None,
                        help="HuggingFace gemma-4 checkpoint directory.")
    parser.add_argument("--prequantized", default=None,
                        help="Prequant bundle dir from quantize_and_save.py.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save .pte/.ptd. Tempdir if omitted.")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="KV cache length (full model).")
    args = parser.parse_args()

    if not args.tiny_test and not args.model_dir and not args.prequantized:
        parser.error("--tiny-test, --model-dir, or --prequantized is required")

    if args.tiny_test:
        model, config = build_tiny_model()
        # Tiny vocab=256, so use ids inside that range.
        prompt_ids = [2, 17, 53, 91, 128, 200, 7]
    elif args.prequantized:
        model, config = load_prequantized_model(args.prequantized, args.max_seq_len)
        prompt_ids = DEFAULT_PROMPT_IDS
    else:
        model, config = load_full_model(args.model_dir, args.max_seq_len)
        prompt_ids = DEFAULT_PROMPT_IDS

    materialize_buffers(model, config)

    # Prequantized models use CUDA-only kernels (Int4TilePackedTo4dTensor).
    # The INT4 31B model is ~16 GB, so the whole pipeline fits in a single
    # 80 GB GPU without the bf16 OOM that plain --model-dir hits at lowering.
    if args.prequantized and torch.cuda.is_available():
        print("Moving prequantized model to CUDA...")
        model.to("cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    T = len(prompt_ids)
    if T < 2:
        raise ValueError("Prompt must have at least 2 tokens for prefill.")
    if T > config.max_seq_len - 1:
        raise ValueError(f"Prompt length {T} exceeds max_seq_len-1={config.max_seq_len - 1}")

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    input_pos = torch.arange(T, dtype=torch.long, device=device)

    # ---- Stage 1: eager forward ----
    _stage(1, "Eager mode forward pass")
    try:
        _reset_kv(model)
        with torch.no_grad():
            eager_logits = model(tokens, input_pos)
        top5 = eager_logits[0, -1, :].float().topk(5).indices.tolist()
        # Move to CPU so we don't pin a GPU copy of the vocab-wide logits through
        # the rest of the pipeline (only needed for numerical diff comparisons).
        eager_logits = eager_logits.detach().cpu()
        _ok(f"logits shape: {tuple(eager_logits.shape)}, top-5 last: {top5}")
    except Exception as e:
        _fail_and_exit(1, "Eager mode forward pass", e)

    # ---- Stage 2 + 3: torch.export decode + prefill ----
    decode_ep = None
    prefill_ep = None
    _stage(2, "torch.export (decode)")
    try:
        from torch.export import export
        decode_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
        decode_pos = torch.tensor([0], dtype=torch.long, device=device)
        _reset_kv(model)
        with torch.no_grad():
            decode_ep = export(model, (decode_tokens, decode_pos), strict=True)
        _ok()
    except Exception as e:
        _fail_and_exit(2, "torch.export (decode)", e)

    _stage(3, "torch.export (prefill)")
    try:
        from torch.export import Dim, export
        max_prefill = config.max_seq_len - 1
        ex_tokens = torch.zeros((1, max_prefill), dtype=torch.long, device=device)
        ex_pos = torch.arange(max_prefill, dtype=torch.long, device=device)
        seq_dim = Dim("seq_len", min=2, max=max_prefill)
        dynamic_shapes = ({1: seq_dim}, {0: seq_dim})
        _reset_kv(model)
        with torch.no_grad():
            prefill_ep = export(
                model, (ex_tokens, ex_pos), dynamic_shapes=dynamic_shapes, strict=True,
            )
        _ok()
    except Exception as e:
        _fail_and_exit(3, "torch.export (prefill)", e)

    # ---- Stage 4: numerical check on exported (replayed) prefill ----
    _stage(4, "Exported model numerical check")
    try:
        _reset_kv(model)
        ep_module = prefill_ep.module()
        with torch.no_grad():
            ep_logits = ep_module(tokens, input_pos)
        diff = _max_abs_diff(eager_logits, ep_logits)
        _ok(f"max_abs_diff: {diff:.4g}")
    except Exception as e:
        _fail_and_exit(4, "Exported model numerical check", e)

    # Free the stage-2/3/4 EPs and intermediates before re-exporting + lowering.
    # On the 31B model the GPU is near-full; lingering graph metadata + logits
    # tensors can push move_to_device_pass over the edge.
    del decode_ep, prefill_ep, ep_module, ep_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 5: to_edge_transform_and_lower ----
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="gemma4_export_")
    _stage(5, "to_edge_transform_and_lower")
    try:
        # The lowered program will mutate buffers under-the-hood; export again
        # from a freshly-reset model so the EPs we lower haven't already been
        # consumed by the stage-4 replay.
        _reset_kv(model)
        decode_ep_l, prefill_ep_l = export_two_methods(model, config)
        lower_and_save(decode_ep_l, prefill_ep_l, config, output_dir)
        _ok(f"saved to {output_dir}")
    except Exception as e:
        _fail_and_exit(5, "to_edge_transform_and_lower", e)

    # Free the original model + EPs so the .pte runtime has room to load its
    # own copy of the weights into CUDA. With INT4 31B the model+blobs are
    # ~22 GB each; both can't coexist on a single 80 GB GPU.
    del model, decode_ep_l, prefill_ep_l
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 6: load .pte and compare lowered logits ----
    _stage(6, "Lowered model numerical check")
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        pte_path = os.path.join(output_dir, "model.pte")
        ptd_path = None
        for fname in os.listdir(output_dir):
            if fname.endswith(".ptd"):
                ptd_path = os.path.join(output_dir, fname)
                break
        loaded = _load_for_executorch(pte_path, data_path=ptd_path)
        outputs = loaded.run_method("prefill", [tokens.cpu(), input_pos.cpu()])
        lowered_logits = outputs[0]
        diff = _max_abs_diff(eager_logits, lowered_logits)
        _ok(f"max_abs_diff: {diff:.4g}")
    except Exception as e:
        _fail_and_exit(6, "Lowered model numerical check", e)

    print("\nAll 6 stages passed.")


if __name__ == "__main__":
    main()
