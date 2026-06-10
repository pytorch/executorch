# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an exported MLX EAGLE-3 ``.pte`` (gemma4-31B target + draft) on device.

The MLX export exposes two methods that each own a persistent KV cache within
their handle (MLX has no cross-method sharing) and return *distributions* so the
host applies temperature + (rejection) sampling:
  - ``target_forward(tokens, input_pos) -> (logits[1,T,V], fused_feature[1,T,H])``
  - ``draft_decode(tokens, feature, input_pos) -> (draft_logits[1,T,Vd], g[1,T,H])``

The shifted (vLLM-EAGLE) loop from ``eagle3/test_speculator.py`` runs over these
two methods, mapping the test's ``prefill``/``target_verify`` onto the merged
``target_forward``.

``--temperature 0`` is greedy: proposals/verify use argmax and the output equals
the target's greedy decode (lossless, token-exact). ``--temperature > 0`` uses
modified rejection sampling (Leviathan/Chen), which reproduces the target's
sampling distribution. The draft's reduced vocab is bridged to the target vocab
via ``d2t`` (target_id = draft_id + d2t[draft_id]); the rejection residual is
taken over the full target vocab so tokens the draft cannot propose are still
reachable.

Each run loads a fresh program so the KV caches start zeroed.

Usage:
    python -m executorch.examples.models.eagle3.run_mlx \\
        --pte ./eagle3_mlx/model.pte \\
        --tokenizer-path ./gemma-4-31B-it-HQQ-INT4/tokenizer.json \\
        --draft "$SPEC" \\
        --prompt "Write a short joke about saving RAM." \\
        --chain 3 --num-gen 128 --temperature 0.8 --seed 0
"""

import argparse
import os
import time

import torch

from executorch.examples.models.gemma4_31b.inference import apply_chat_template

EOS_TOKEN_IDS = {1, 50, 106}
BOS_TOKEN_ID = 2


def _toks(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long)


def _target_forward(method, token_ids, positions):
    logits, feat = method.execute(
        [_toks(token_ids), torch.tensor(positions, dtype=torch.long)]
    )
    return logits, feat


def _pick(row_logits, temp, gen):
    """Greedy argmax (temp==0) or a temperature sample. Returns (index, probs)."""
    if temp == 0.0:
        return int(row_logits.argmax()), None
    probs = torch.softmax(row_logits.float() / temp, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen)), probs


def _propose_chain(
    draft_method, seed_tokens, seed_feat, seed_pos, chain_len, temp, d2t, gen
):
    """Seed the draft (last slot predicts proposal 0), then chain_len-1 steps.

    Returns (proposals, draft_ids, q_list) where ``proposals`` are target-vocab
    ids; ``q_list`` holds the draft distribution per proposal (None when greedy).
    """
    dl, g = draft_method.execute([seed_tokens, seed_feat, seed_pos])
    proposals, draft_ids, qs = [], [], []
    d, q = _pick(dl[0, -1], temp, gen)
    proposals.append(d + int(d2t[d]))
    draft_ids.append(d)
    qs.append(q)
    last = int(seed_pos[-1])
    tok, feat = _toks([proposals[-1]]), g[:, -1:]
    for k in range(1, chain_len):
        dl, g = draft_method.execute(
            [tok, feat, torch.tensor([last + k], dtype=torch.long)]
        )
        d, q = _pick(dl[0, 0], temp, gen)
        proposals.append(d + int(d2t[d]))
        draft_ids.append(d)
        qs.append(q)
        tok, feat = _toks([proposals[-1]]), g
    return proposals, draft_ids, qs


def _accept(p_logits, proposal_x, draft_d, q, temp, target_ids_all, gen):
    """Accept one proposal (greedy or modified rejection sampling).

    Returns (accepted, fallback_token): on rejection ``fallback_token`` is the
    corrected token (greedy argmax, or a residual sample over the target vocab).
    """
    if temp == 0.0:
        greedy = int(p_logits.argmax())
        return (proposal_x == greedy), greedy
    p = torch.softmax(p_logits.float() / temp, dim=-1)
    qx = q[draft_d]
    ratio = (p[proposal_x] / qx).clamp(max=1.0)
    if torch.rand((), generator=gen) <= ratio:
        return True, None
    # Resample from the residual (p - q)_+ over the full target vocab; the draft
    # contributes q only on its reachable target ids.
    q_target = torch.zeros_like(p)
    q_target[target_ids_all] = q
    resid = (p - q_target).clamp(min=0)
    resid = resid / resid.sum()
    return False, int(torch.multinomial(resid, 1, generator=gen))


def speculative_decode(
    target_method, draft_method, prompt, chain_len, num_gen, temp, d2t, gen
):
    """Shifted one-target-forward-per-round speculative decode (greedy or sampling).

    Returns (generated, num_steps, num_accepted, prefill_s) where prefill_s is the
    time of the initial prompt forward, so the caller can report decode-only tok/s.
    """
    target_ids_all = torch.arange(d2t.numel(), dtype=torch.long) + d2t
    L = len(prompt)
    _t = time.perf_counter()
    logits, feat_prompt = _target_forward(target_method, prompt, list(range(L)))
    prefill_s = time.perf_counter() - _t
    anchor, _ = _pick(logits[0, -1], temp, gen)
    anchor_pos = L
    emitted = [anchor]
    num_steps = num_accepted = 0
    if anchor in EOS_TOKEN_IDS:
        return emitted, num_steps, num_accepted, prefill_s

    proposals, draft_ids, qs = _propose_chain(
        draft_method,
        _toks(prompt[1:] + [anchor]),
        feat_prompt,
        torch.arange(L, dtype=torch.long),
        chain_len,
        temp,
        d2t,
        gen,
    )

    while len(emitted) < num_gen:
        num_steps += 1
        vlogits, vfeat = _target_forward(
            target_method,
            [anchor] + proposals,
            list(range(anchor_pos, anchor_pos + chain_len + 1)),
        )
        a = 0
        corrected = None
        for j in range(chain_len):
            accepted, fallback = _accept(
                vlogits[0, j], proposals[j], draft_ids[j], qs[j], temp, target_ids_all, gen
            )
            if accepted:
                a += 1
            else:
                corrected = fallback
                break
        if corrected is None:  # whole chain accepted -> bonus from the next dist
            corrected, _ = _pick(vlogits[0, chain_len], temp, gen)
        num_accepted += a

        new = proposals[:a] + [corrected]
        eos_pos = next((i for i, t in enumerate(new) if t in EOS_TOKEN_IDS), None)
        if eos_pos is not None:
            new = new[: eos_pos + 1]
        new = new[: num_gen - len(emitted)]
        emitted += new
        if eos_pos is not None or len(emitted) >= num_gen:
            break

        proposals, draft_ids, qs = _propose_chain(
            draft_method,
            _toks(proposals[:a] + [corrected]),
            vfeat[:, : a + 1],
            torch.arange(anchor_pos, anchor_pos + a + 1, dtype=torch.long),
            chain_len,
            temp,
            d2t,
            gen,
        )
        anchor, anchor_pos = corrected, anchor_pos + 1 + a

    return emitted[:num_gen], num_steps, num_accepted, prefill_s


def greedy_decode(target_method, prompt, num_gen):
    """Greedy baseline using only ``target_forward`` (the lossless reference).

    Returns (generated, prefill_s).
    """
    L = len(prompt)
    _t = time.perf_counter()
    logits, _ = _target_forward(target_method, prompt, list(range(L)))
    prefill_s = time.perf_counter() - _t
    tok, pos, out = int(logits[0, -1].argmax()), L, []
    out.append(tok)
    while len(out) < num_gen and tok not in EOS_TOKEN_IDS:
        logits, _ = _target_forward(target_method, [tok], [pos])
        tok, pos = int(logits[0, -1].argmax()), pos + 1
        out.append(tok)
    return out[:num_gen], prefill_s


def _load_d2t(draft_dir: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(os.path.join(draft_dir, "model.safetensors"))["d2t"].to(torch.long)


def _encode(tokenizer, prompt, raw_prompt):
    text = prompt if raw_prompt else apply_chat_template(prompt)
    ids = tokenizer.encode(text).ids
    if not ids or ids[0] != BOS_TOKEN_ID:
        ids = [BOS_TOKEN_ID] + ids
    return ids


def _program_const(program, name, default):
    """Read a scalar constant_method from the program; fall back to default."""
    try:
        return int(program.load_method(name).execute([])[0])
    except Exception:
        return default


def main() -> None:
    p = argparse.ArgumentParser(description="Run an MLX EAGLE-3 .pte (gemma4-31B).")
    p.add_argument("--pte", required=True, help="Path to the exported model.pte.")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument(
        "--draft",
        default=None,
        help="EAGLE-3 draft dir (for d2t); required for --mode speculative.",
    )
    p.add_argument("--prompt", default="Write a short joke about saving RAM.")
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--num-gen", type=int, default=128)
    p.add_argument(
        "--chain",
        type=int,
        default=None,
        help="Draft chain length K (default: the export-time get_chain_len).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (lossless); >0 = modified rejection sampling.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    p.add_argument(
        "--mode",
        default="speculative",
        choices=["speculative", "greedy"],
        help="speculative = draft+verify; greedy = target-only baseline.",
    )
    args = p.parse_args()
    if args.mode == "speculative" and not args.draft:
        p.error("--mode speculative requires --draft (for the d2t vocab map).")

    from executorch.runtime import Runtime, Verification
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    prompt_ids = _encode(tokenizer, args.prompt, args.raw_prompt)

    program = Runtime.get().load_program(args.pte, verification=Verification.Minimal)
    target_method = program.load_method("target_forward")
    chain = args.chain if args.chain is not None else _program_const(program, "get_chain_len", 3)

    print(f"\nPrompt: {args.prompt}\n" + "-" * 40)
    t0 = time.perf_counter()
    if args.mode == "greedy":
        generated, prefill_s = greedy_decode(target_method, prompt_ids, args.num_gen)
    else:
        gen = torch.Generator().manual_seed(args.seed)
        d2t = _load_d2t(args.draft)
        draft_method = program.load_method("draft_decode")
        generated, num_steps, num_accepted, prefill_s = speculative_decode(
            target_method,
            draft_method,
            prompt_ids,
            chain,
            args.num_gen,
            args.temperature,
            d2t,
            gen,
        )
        if num_steps:
            accept_len = (num_accepted + num_steps) / num_steps
            print(
                f"  speculative (chain={chain}, temp={args.temperature}): "
                f"{num_steps} target steps, {num_accepted} draft tokens accepted, "
                f"mean accept length {accept_len:.2f}"
            )
    elapsed = time.perf_counter() - t0
    # Decode tok/s excludes the prompt prefill (matches the CUDA runner).
    decode_s = max(elapsed - prefill_s, 1e-9)

    print(tokenizer.decode(generated))
    print("-" * 40)
    print(f"token ids: {generated}")
    print(
        f"prefill: {len(prompt_ids)} tokens in {prefill_s:.2f}s | "
        f"decode: {len(generated) / decode_s:.2f} tok/s "
        f"({len(generated)} tokens in {decode_s:.2f}s)"
    )
    print(f"Generated in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
