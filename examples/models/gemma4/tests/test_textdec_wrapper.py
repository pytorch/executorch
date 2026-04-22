"""Test the TextDecoderExport wrapper used in multimodal export.

Compares it to direct tokens= path to see if the h= + pli_token_ids path
produces the same logits.
"""
from __future__ import annotations
import os, sys, torch
from pathlib import Path

HF_DIR = os.environ.get("GEMMA4_HF_DIR", str(Path.home() / "models/gemma-4-E2B-it"))
ET_CKPT = os.environ.get("GEMMA4_ET_CHECKPOINT", str(Path(HF_DIR) / "model_et.pth"))
ET_PARAMS = str(Path(__file__).resolve().parents[1] / "config" / "e2b_config.json")
PROMPT = "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"


def main():
    from transformers import AutoTokenizer
    from executorch.examples.models.llama.export_llama_lib import _prepare_for_llama_export
    from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType
    from executorch.examples.models.gemma4.export_gemma4_multimodal import (
        TokenEmbeddingExport, TextDecoderExport
    )

    tok = AutoTokenizer.from_pretrained(HF_DIR)
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    S = ids.shape[1]
    print(f"prompt has {S} tokens")

    cfg = LlmConfig()
    cfg.base.model_class = ModelType.gemma4
    cfg.base.params = ET_PARAMS
    cfg.base.checkpoint = ET_CKPT
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.enable_dynamic_shape = True
    cfg.export.max_seq_length = 128
    cfg.export.max_context_length = 128

    print("Building model...")
    builder = _prepare_for_llama_export(cfg)
    transformer = builder.model.eval()

    tok_emb = TokenEmbeddingExport(transformer).eval()
    txt_dec = TextDecoderExport(transformer).eval()

    # ---- Reference: tokens= batched prefill ----
    print("\nRef: tokens= batched prefill")
    with torch.no_grad():
        out = transformer(tokens=ids, attn_options={"input_pos": torch.tensor([0])})
        if isinstance(out, tuple): out = out[0]
        ref_logits = out[0, -1] if out.dim() == 3 else out[0]
    print(f"  top-1: {torch.argmax(ref_logits).item()} {tok.decode([torch.argmax(ref_logits).item()])!r}")

    # ---- Wrapper: h= path with PLI token IDs ----
    print("\nWrapper: TextDecoderExport(embeds, [0], pli_token_ids=ids)")
    transformer2 = _prepare_for_llama_export(cfg).model.eval()  # fresh KV cache
    tok_emb2 = TokenEmbeddingExport(transformer2).eval()
    txt_dec2 = TextDecoderExport(transformer2).eval()
    with torch.no_grad():
        embeds = tok_emb2(ids)
        out2 = txt_dec2(embeds, torch.tensor([0]), ids)
        if isinstance(out2, tuple): out2 = out2[0]
        wr_logits = out2[0] if out2.dim() == 2 else out2[0, -1]
    print(f"  top-1: {torch.argmax(wr_logits).item()} {tok.decode([torch.argmax(wr_logits).item()])!r}")
    print(f"  max_diff vs ref: {float((ref_logits.float() - wr_logits.float()).abs().max()):.4e}")

    # ---- Sequential decode after batched prefill ----
    print("\nSequential decode 15 tokens via wrapper:")
    transformer3 = _prepare_for_llama_export(cfg).model.eval()
    tok_emb3 = TokenEmbeddingExport(transformer3).eval()
    txt_dec3 = TextDecoderExport(transformer3).eval()
    with torch.no_grad():
        # Batched prefill
        embeds = tok_emb3(ids)
        out_p = txt_dec3(embeds, torch.tensor([0]), ids)
        if isinstance(out_p, tuple): out_p = out_p[0]
        nxt_logits = out_p[0] if out_p.dim() == 2 else out_p[0, -1]
        nxt = int(torch.argmax(nxt_logits).item())
        gen = [nxt]
        pos = S
        for _ in range(15):
            t = torch.tensor([[nxt]], dtype=torch.long)
            e = tok_emb3(t)
            o = txt_dec3(e, torch.tensor([pos]), t)
            if isinstance(o, tuple): o = o[0]
            l = o[0] if o.dim() == 2 else o[0, -1]
            nxt = int(torch.argmax(l).item())
            gen.append(nxt)
            pos += 1
    print(f"  generated: {tok.decode(gen)!r}")


if __name__ == "__main__":
    main()
