#!/usr/bin/env python3
"""
Run the correctness eval against an OpenAI-compatible /v1/chat/completions
endpoint. Framework-agnostic — works for vLLM on NVIDIA today and every
SiliconBench framework on Mac later.

Reads prompts.jsonl (one JSON object per line; each with `id`, `messages`,
`max_tokens`). Writes responses.jsonl with one record per prompt:
  {id, content, prompt_tokens, completion_tokens, finish_reason, latency_s, error}

Greedy decoding (temperature=0, top_p=1). No streaming — we only care about
the final text for scoring.

Resume: if --output already exists and has N records with ids, those ids are
skipped on a re-run unless --overwrite is set.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from jsonl_io import iter_jsonl


async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    record: dict,
    temperature: float,
    top_p: float,
    timeout: int,
    max_retries: int,
) -> dict:
    payload = {
        "model": model,
        "messages": record["messages"],
        "max_tokens": record["max_tokens"],
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        # Disable thinking for templates that opt into it (Qwen3, Gemma4,
        # ...). Forwarded to the Jinja chat template by vllm/llamacpp/
        # mlx-vlm; harmless on templates that don't read the kwarg. Without
        # this, Gemma4-E4B-it consumed the entire 256-token max_tokens
        # budget on reasoning and never emitted JSON, scoring F1≈0.14.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_err = None
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            latency = time.perf_counter() - t0

            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
            # Optional — not all backends split reasoning from content
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            # mlx-vlm reports input_tokens/output_tokens; OpenAI standard is
            # prompt_tokens/completion_tokens. Tolerate either.
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

            return {
                "id": record["id"],
                "content": content,
                "reasoning": reasoning,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": choice.get("finish_reason"),
                "latency_s": latency,
                "error": None,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue

    return {
        "id": record["id"],
        "content": "",
        "reasoning": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "latency_s": time.perf_counter() - t0,
        "error": last_err,
    }


def load_done_ids(path: Path) -> set[str]:
    """Return ids from an existing responses.jsonl (for resume)."""
    try:
        return {
            str(rec["id"]) for rec in iter_jsonl(path)
            if rec["error"] is None
        }
    except FileNotFoundError:
        return set()


async def run(args):
    prompts_path = Path(args.prompts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        output_path.unlink(missing_ok=True)

    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"Resuming — {len(done_ids)} ids already completed in {output_path}")

    prompts = [rec for rec in iter_jsonl(prompts_path)
               if str(rec["id"]) not in done_ids]

    if not prompts:
        print("Nothing to do — all prompts already completed.")
        return

    total = len(prompts)
    print(f"Sending {total} prompts to {args.base_url} (model={args.model}, "
          f"concurrency={args.concurrency}, temp={args.temperature}, top_p={args.top_p})")

    sem = asyncio.Semaphore(args.concurrency)
    completed = 0
    t_start = time.perf_counter()

    with open(output_path, "a") as out_f:
        async def bounded(rec, session):
            nonlocal completed
            async with sem:
                result = await one_request(
                    session, args.base_url, args.model, rec,
                    args.temperature, args.top_p, args.timeout, args.max_retries,
                )
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                if completed % 25 == 0 or completed == total:
                    elapsed = time.perf_counter() - t_start
                    rps = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{total}] {rps:.1f} req/s")
                return result

        connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await asyncio.gather(
                *[bounded(rec, session) for rec in prompts],
                return_exceptions=False,
            )

    errors = [r for r in results if r["error"]]
    print(f"\nDone. {len(results) - len(errors)} succeeded, {len(errors)} errored.")
    if errors:
        unique = list({r["error"] for r in errors})[:3]
        for e in unique:
            print(f"  sample error: {e}")
    print(f"Responses written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible server base URL, e.g. http://localhost:8000")
    parser.add_argument("--model", required=True,
                        help="Model name as the server expects it, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts", required=True, help="Path to prompts.jsonl")
    parser.add_argument("--output", required=True, help="Path to responses.jsonl")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing responses.jsonl before running")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
