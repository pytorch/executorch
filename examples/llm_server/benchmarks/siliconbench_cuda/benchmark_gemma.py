#!/usr/bin/env python3
"""
SiliconBench — Benchmark an OpenAI-compatible LLM endpoint.

Measures TTFT, throughput (tok/s), inter-token latency, and total latency
at specified concurrency levels.
"""

import argparse
import asyncio
import json
import os
import platform
import ssl
import subprocess
import time
import sys
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict, field

import aiohttp

# The prompt set is frozen and lives on the Hub. Pinning a commit makes the URL
# immutable, so there is nothing to cache or verify.
PROMPTS_REVISION = "3e9fa20c6fc6de333c5cb40e5da46d001194443c"
PROMPTS_URL = ("https://huggingface.co/datasets/windchimeran/SiliconBench/resolve/"
               f"{PROMPTS_REVISION}/{{split}}_benchmark_prompts.json")
# Offline runs (APPLEBENCH_OFFLINE=1) read a copy of the same pinned revision, saved
# once with --save-prompts. The directory is named after the revision, so a copy of
# an older revision is never picked up after PROMPTS_REVISION changes.
LOCAL_PROMPTS_DIR = Path(__file__).resolve().parent.parent / ".prompts" / PROMPTS_REVISION


def _hub_ssl_context():
    """TLS context for the Hub download. Some Python builds (e.g. a non-system
    framework Python on macOS) ship no CA bundle and fail certificate checks unless
    SSL_CERT_FILE is set; fall back to the system bundle when it exists."""
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile and os.path.exists("/etc/ssl/cert.pem"):
        cafile = "/etc/ssl/cert.pem"
    return ssl.create_default_context(cafile=cafile)


def save_prompts():
    """Download both splits of the pinned prompt set for offline runs."""
    LOCAL_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("chat", "agent"):
        with urllib.request.urlopen(PROMPTS_URL.format(split=split), timeout=60,
                                    context=_hub_ssl_context()) as r:
            data = json.load(r)
        (LOCAL_PROMPTS_DIR / f"{split}_benchmark_prompts.json").write_text(json.dumps(data))
        print(f"saved {len(data)} {split} prompts to {LOCAL_PROMPTS_DIR}")


def machine_info():
    """Identify the host that produced a result.

    Results are otherwise addressed only by path, so nothing downstream can
    tell one machine's numbers from another's. That is not hypothetical: a run
    on a second Apple machine wrote into the same results/<MODEL>/<split>/ tree
    the first machine used, and a stale file from the older host later won
    collect_results.py's latest-by-mtime selection — reporting success for a
    cell that in fact crashed on the current box.

    Best-effort and never fatal: a benchmark must not fail because a probe did.
    """
    info = {"host": "", "chip": "", "os": ""}
    try:
        info["host"] = platform.node()
        info["os"] = f"{platform.system()} {platform.release()}"
        if platform.system() == "Darwin":
            info["chip"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        else:
            # Linux/CUDA hosts (e.g. DGX Spark): prefer the GPU name, which is
            # what actually distinguishes those boxes, and fall back to the CPU.
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            info["chip"] = (gpu.stdout.strip().splitlines() or [""])[0] \
                if gpu.returncode == 0 else platform.processor()
    except Exception:
        pass
    return info


@dataclass
class RequestResult:
    prompt_name: str
    ttft: float          # seconds
    total_time: float    # seconds
    tokens_generated: int  # SSE content chunks received (client-side count)
    throughput: float    # tokens/sec, decode only (excludes TTFT); server-token denominator
    inter_token_latency: float  # mean ms per token; server-token denominator
    decode_span: float = 0.0    # seconds between first and last content chunk
    decode_tokens: int = 0      # server tokens attributable to decode_span (n_decoded - 1)
    prompt_tokens: int | None = None      # server-reported, from usage chunk
    completion_tokens: int | None = None  # server-reported, from usage chunk
    content: str = ""    # concatenated assistant content deltas
    reasoning: str = ""  # concatenated reasoning_content / reasoning deltas
    tool_call_deltas: list = field(default_factory=list)
    request_idx: int = -1  # submission index within the concurrency level


# End-to-end limit per request. At 300 s a 12B model's c16 agent level timed out
# up to 65 of 100 requests while still queued, so the level measured the timeout
# rather than the engine. This stays only as a guard against a hung server.
REQUEST_TIMEOUT_S = 1800

# The stock SGLang chat endpoint rewrites final assistant turns. Its official
# completions endpoint accepts token IDs and preserves the audited input.
_native_input_path = os.environ.get("SB_NATIVE_COMPLETION_PROMPTS")
_native_inputs = json.loads(Path(_native_input_path).read_text()) if _native_input_path else None


async def benchmark_single(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: dict,
    max_tokens: int,
) -> RequestResult:
    """Send a single streaming request and measure timing."""

    payload = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Forward enable_thinking=False to the Jinja chat template. Required
        # for Gemma4-E4B-it (its template defaults thinking on); harmless on
        # templates that don't read the kwarg. Vllm/llamacpp/mlx-vlm all
        # honor this OpenAI extension.
        "chat_template_kwargs": {"enable_thinking": False, "current_date": "2026-10-21", "reasoning_strength": "high"},
    }

    endpoint = "/v1/chat/completions"
    if _native_inputs is not None:
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": _native_inputs["prompts"][prompt["name"]],
            "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "stop_token_ids": _native_inputs["stop_token_ids"],
            "skip_special_tokens": False,
        }

    if _native_inputs is not None and _native_inputs.get("logit_bias"):
        payload["logit_bias"] = _native_inputs["logit_bias"]

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    tokens_generated = 0
    prompt_tokens = None
    completion_tokens = None
    content_chunks = []
    reasoning_chunks = []
    tool_call_deltas = []
    saw_done = False
    finish_reason = None

    async with session.post(
        f"{base_url}{endpoint}",
        json=payload,
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} for {prompt['name']}: {body[:2000]}")
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                saw_done = True
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            if usage := data.get("usage"):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            choices = data.get("choices", [])
            if not choices:
                continue
            if fr := choices[0].get("finish_reason"):
                finish_reason = fr
            delta = choices[0].get("delta", {})
            content_delta = choices[0].get("text", "") if _native_inputs is not None else delta.get("content", "")
            reasoning_delta = delta.get("reasoning_content", "") or delta.get("reasoning", "")
            tools_delta = delta.get("tool_calls") or []
            tool_call_deltas.extend(tools_delta)
            has_tool_text = any(t.get("function", {}).get("name") or t.get("function", {}).get("arguments") for t in tools_delta)
            if content_delta or reasoning_delta or has_tool_text:
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                token_times.append(now)
                tokens_generated += 1
                if content_delta:
                    content_chunks.append(content_delta)
                if reasoning_delta:
                    reasoning_chunks.append(reasoning_delta)

    t_end = time.perf_counter()

    if not saw_done or completion_tokens is None or prompt_tokens is None:
        raise RuntimeError("incomplete stream or missing server token usage; exclude from token throughput")

    if t_first_token is None:
        t_first_token = t_end

    ttft = t_first_token - t_start
    total_time = t_end - t_start

    # Decode throughput: tokens after first / time after first.
    # Denominator is the server's completion_tokens, not the client-side SSE
    # chunk count: frameworks may pack several tokens into one delta and they
    # differ (llamacpp/vllm_metal ~1.0 per chunk, omlx 2.0-4.0). Dividing by
    # chunks scaled omlx's throughput down and its ITL up by exactly its
    # bundling factor. summarize() reports tokens_per_chunk so it stays visible.
    decode_time = t_end - t_first_token if t_first_token else total_time
    n_decoded = completion_tokens if (completion_tokens or 0) > 0 else tokens_generated
    throughput = (n_decoded - 1) / decode_time if decode_time > 0 and n_decoded > 1 else 0.0

    # Inter-token latency (mean). Exact once the denominator is the real token
    # count. The distribution is not recoverable when tokens arrive bundled —
    # fewer timestamps than tokens — so itl_p50_ms stays chunk-derived and is
    # not comparable across frameworks that bundle differently.
    if len(token_times) > 1 and n_decoded > 1:
        itl = (token_times[-1] - token_times[0]) / (n_decoded - 1) * 1000  # ms
    else:
        itl = 0.0

    # Detect silent failures: server returned 200 but produced no usable output.
    # A short completion that stopped naturally (finish_reason="stop", or no
    # finish_reason reported by the server) is a LEGITIMATE answer — e.g. a
    # binary-classification prompt correctly answered in a single token ("No").
    # Capable models answer such prompts concisely, so the old "<=1 token =
    # failure" rule mis-flagged them. The real silent-failure signature is the
    # server returning OK with zero tokens, or truncating abnormally
    # (finish_reason like "length"/"error") on a <=1-token reply.
    abnormal_stop = finish_reason not in (None, "stop")
    if max_tokens > 1 and (tokens_generated == 0 or (tokens_generated <= 1 and abnormal_stop)):
        raise RuntimeError(
            f"silent failure: server returned OK but generated only "
            f"{tokens_generated} token(s) for prompt '{prompt['name']}' "
            f"(requested max_tokens={max_tokens}, finish_reason={finish_reason})"
        )

    return RequestResult(
        prompt_name=prompt["name"],
        ttft=ttft,
        total_time=total_time,
        tokens_generated=tokens_generated,
        throughput=throughput,
        inter_token_latency=itl,
        decode_span=(token_times[-1] - token_times[0]) if len(token_times) > 1 else 0.0,
        decode_tokens=max(n_decoded - 1, 0) if len(token_times) > 1 else 0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        content="".join(content_chunks),
        reasoning="".join(reasoning_chunks),
        tool_call_deltas=tool_call_deltas,
    )


async def run_concurrent(
    base_url: str,
    model: str,
    prompts: list,
    concurrency: int,
    num_requests: int,
    warmup: int,
) -> list[RequestResult]:
    """Run benchmark at a given concurrency level."""

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warmup
        if warmup > 0:
            print(f"  Warming up ({warmup} requests)...")
            for i in range(warmup):
                prompt = prompts[i % len(prompts)]
                try:
                    await benchmark_single(session, base_url, model, prompt, prompt["max_tokens"])
                except Exception as e:
                    print(f"  Warmup error: {e}")

        # Benchmark
        print(f"  Running {num_requests} requests at concurrency {concurrency}...")
        t_wall_start = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)
        results = []

        async def bounded_request(prompt, idx):
            async with sem:
                r = await benchmark_single(session, base_url, model, prompt, prompt["max_tokens"])
                r.request_idx = idx
                return r

        tasks = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            tasks.append(bounded_request(prompt, i))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.perf_counter() - t_wall_start

        # Separate successes and failures — both are reported
        good = []
        errors = []
        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
            else:
                good.append(r)

        if errors:
            print(f"  {len(errors)}/{len(results)} requests FAILED")
            # Print first unique error
            unique = list(dict.fromkeys(errors))
            for e in unique[:3]:
                print(f"    > {e[:200]}")

        return good, errors, wall_time


def summarize(results: list[RequestResult], errors: list[str], concurrency: int, num_requests: int, wall_time: float) -> dict:
    """Compute summary statistics."""
    if not results:
        unique_errors = list(dict.fromkeys(errors))[:5]
        return {"concurrency": concurrency, "num_requests": num_requests,
                "successful": 0, "failed": len(errors), "wall_time_s": wall_time,
                "error": "no successful requests", "error_samples": unique_errors}

    ttfts = [r.ttft for r in results]
    throughputs = [r.throughput for r in results]
    total_times = [r.total_time for r in results]
    itls = [r.inter_token_latency for r in results if r.inter_token_latency > 0]
    tokens = [r.tokens_generated for r in results]

    def percentile(data, p):
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    total_chunks = sum(tokens)
    agg_decode_span = sum(r.decode_span for r in results)
    agg_decode_tokens = sum(r.decode_tokens for r in results)

    # Server-reported token counts from the usage chunk (None if any request missing it)
    completion_list = [r.completion_tokens for r in results]
    prompt_list = [r.prompt_tokens for r in results]
    has_usage = all(t is not None for t in completion_list)
    has_input = all(t is not None for t in prompt_list)

    # Prefer server counts; fall back to chunk count (validate_results raises a warning)
    total_output_tokens = sum(completion_list) if has_usage else total_chunks
    total_input_tokens = sum(prompt_list) if has_input else None

    output_throughput = total_output_tokens / wall_time if wall_time > 0 else 0.0
    input_throughput = (total_input_tokens / wall_time) if (has_input and wall_time > 0) else None
    total_throughput = ((total_input_tokens + total_output_tokens) / wall_time) if (has_input and wall_time > 0) else None

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successful": len(results),
        "failed": len(errors),
        "total_tokens_generated": total_chunks,
        "total_output_tokens": total_output_tokens,
        # Server tokens per SSE chunk; >1 means the framework bundles.
        "tokens_per_chunk": (total_output_tokens / total_chunks) if total_chunks else None,
        "total_input_tokens": total_input_tokens,
        "server_usage_available": has_usage,
        "ttft_avg_ms": sum(ttfts) / len(ttfts) * 1000,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000,
        "ttft_p99_ms": percentile(ttfts, 99) * 1000,
        # Token-weighted aggregates, NOT means of per-request rates. A request
        # that arrives in very few SSE chunks has a near-zero span, so its
        # individual rate explodes and poisons a plain mean: omlx on Qwen3-0.6B
        # (47 tokens in 3 chunks) produced 8016 tok/s for a model whose
        # bandwidth ceiling is ~330. Aggregating also keeps throughput_avg_tps
        # and itl_avg_ms consistent with each other by construction.
        "throughput_avg_tps": (agg_decode_tokens / agg_decode_span) if agg_decode_span > 0 else 0.0,
        "throughput_p50_tps": percentile(throughputs, 50),
        "itl_avg_ms": (agg_decode_span / agg_decode_tokens * 1000) if agg_decode_tokens else 0.0,
        "itl_p50_ms": percentile(itls, 50) if itls else 0.0,
        "latency_avg_s": sum(total_times) / len(total_times),
        "latency_p50_s": percentile(total_times, 50),
        "latency_p99_s": percentile(total_times, 99),
        "output_throughput_tps": output_throughput,
        "input_throughput_tps": input_throughput,
        "total_token_throughput_tps": total_throughput,
        "wall_time_s": wall_time,
        "error_samples": list(dict.fromkeys(errors))[:5] if errors else [],
    }


def validate_results(results: list[RequestResult], summary: dict, prompts_used: list[dict]):
    """Warn about suspicious results that may indicate framework issues."""
    warnings = []

    successful = summary.get("successful", 0)
    if successful == 0:
        return warnings

    # Check 1: average tokens per request vs expected (prefer server-reported count)
    total_out = summary.get("total_output_tokens") or summary.get("total_tokens_generated", 0)
    avg_tokens = total_out / successful
    expected_max_tokens = sum(p["max_tokens"] for p in prompts_used[:successful]) / successful
    if avg_tokens < expected_max_tokens * 0.1:
        warnings.append(
            f"avg tokens/request ({avg_tokens:.0f}) is far below expected "
            f"(max_tokens avg={expected_max_tokens:.0f}). Possible silent failures."
        )

    # Check 2: throughput vs ITL consistency
    throughput_avg = summary.get("throughput_avg_tps", 0)
    itl_avg = summary.get("itl_avg_ms", 0)
    if throughput_avg > 0 and itl_avg > 0:
        implied_throughput = 1000.0 / itl_avg
        ratio = throughput_avg / implied_throughput if implied_throughput > 0 else 0
        if ratio < 0.25 or ratio > 4.0:
            warnings.append(
                f"throughput ({throughput_avg:.1f} tok/s) inconsistent with ITL "
                f"({itl_avg:.1f}ms, implies ~{implied_throughput:.1f} tok/s). "
                f"Ratio: {ratio:.2f}x."
            )

    # Check 3: many zero-throughput requests (p50 = 0 means >50% had 0 throughput)
    if summary.get("throughput_p50_tps", -1) == 0 and throughput_avg > 0:
        zero_count = sum(1 for r in results if r.throughput == 0)
        warnings.append(
            f"{zero_count}/{successful} requests had 0 throughput "
            f"(generated <=1 token each)."
        )

    # Check 4: server did not report token usage — fell back to SSE chunk count
    if not summary.get("server_usage_available", False):
        missing = sum(1 for r in results if r.completion_tokens is None)
        warnings.append(
            f"server did not report token usage for {missing}/{successful} requests "
            f"(missing `usage.completion_tokens`). Output counts fell back to SSE chunk "
            f"count, which may undercount if the framework bundles multiple tokens per "
            f"chunk. Framework should honor `stream_options.include_usage=true`."
        )

    # Check 5: server completion_tokens diverges from client-side chunk count
    if summary.get("server_usage_available", False):
        divergent = []
        for r in results:
            if r.completion_tokens is None or r.tokens_generated == 0:
                continue
            ratio = r.completion_tokens / r.tokens_generated
            if ratio < 0.95 or ratio > 1.05:
                divergent.append(ratio)
        if divergent:
            avg_ratio = sum(divergent) / len(divergent)
            warnings.append(
                f"{len(divergent)}/{successful} requests had server completion_tokens "
                f"diverging >5% from SSE chunk count (avg ratio {avg_ratio:.2f}x): the "
                f"framework bundles multiple tokens per SSE delta. throughput_avg_tps and "
                f"itl_avg_ms use the server token count and are unaffected; itl_p50_ms is "
                f"still chunk-derived and is NOT comparable against a non-bundling framework."
            )

    for w in warnings:
        print(f"  \u26a0 WARNING: {w}")

    return warnings


def get_model_name(base_url: str) -> str:
    """Query the server for the model name."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return "unknown"





# Engine knobs are captured by prefix, not by an allowlist: the variable that
# silently broke a comparison last time (LLAMACPP_CTX_SIZE) would not have been
# on a list written the week before.
_SERVE_ENV_PREFIXES = ("VLLM_METAL_", "OMLX_", "LLAMACPP_", "OLLAMA_", "EXECUTORCH_",
                       "LLAMA_ARG_", "MLX_LM_", "VLLM_MLX_", "MISTRALRS_")


def _run_config(args, concurrency_levels):
    """Record how this run was measured, not just what it measured.

    A single-level arm and a level drawn from a multi-level sweep are not
    comparable: the sweep reaches later levels with caches warmed by the
    earlier ones. Serve flags that change the memory budget or context size do
    the same thing more quietly. Nothing used to record either.
    """
    import os

    return {
        "concurrency_levels": list(concurrency_levels),
        # levels after the first reuse the server, and therefore its caches
        "levels_share_server": len(concurrency_levels) > 1,
        "request_api": "/v1/completions" if _native_inputs is not None else "/v1/chat/completions",
        "pretokenized_input": _native_inputs is not None,
        "requests_per_level": args.requests,
        "warmup": args.warmup,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "split": args.split,
        "prompts": args.prompts or (f"local-copy@{PROMPTS_REVISION}"
                                    if os.environ.get("APPLEBENCH_OFFLINE") == "1"
                                    else f"hub@{PROMPTS_REVISION}"),
        "serve_env":{k: v for k, v in sorted(os.environ.items())
                      if k.startswith(_SERVE_ENV_PREFIXES)},
    }


def _framework_version(label):
    """Stamp the build that produced this run. Best-effort: a version lookup
    must never fail a benchmark, and an unknown version is reported as null
    rather than omitted, so a reader cannot assume two runs matched.

    Variant labels (vllm_metal_mtp, omlx_bounded) probe their base engine.
    """
    try:
        from framework_version import describe
    except Exception:
        return None
    import re as _re
    base = _re.sub(r"_n\d+$", "", label)
    for suffix in ("_mtp", "_bounded", "_ssd"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    try:
        d = describe(base)
    except Exception:
        return None
    if base != label:
        d = dict(d, variant_of=base)
    return d

def main():
    if "--save-prompts" in sys.argv:
        save_prompts()
        return
    parser = argparse.ArgumentParser(description="SiliconBench — LLM inference benchmark")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--concurrency", default="16,8,1", help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests")
    parser.add_argument("--framework", required=True, help="Framework name (for labeling results)")
    parser.add_argument("--model", default=None, help="Model name to use in API calls (auto-detected if not set)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--outputs", default=None,
                        help="Optional path for per-request outputs JSONL sidecar (content + reasoning per request)")
    parser.add_argument("--results-dir", default=None, help="Directory for result files (overrides default)")
    parser.add_argument("--prompts", default=None, help="Path to prompts JSON file")
    parser.add_argument("--split", default="chat", choices=["chat", "agent"],
                        help="Prompt split to use: 'chat' or 'agent' (default: chat)")
    parser.add_argument("--max-wall-time", type=int, default=3600,
                        help="Max wall time (seconds) per concurrency level before skipping remaining levels (default: 3600 = 60 min)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Load prompts
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    elif os.environ.get("APPLEBENCH_OFFLINE") == "1":
        local = LOCAL_PROMPTS_DIR / f"{args.split}_benchmark_prompts.json"
        if not local.exists():
            sys.exit(f"APPLEBENCH_OFFLINE=1 but {local} is missing; "
                     "run `python scripts/benchmark.py --save-prompts` while online")
        prompts = json.loads(local.read_text())
    else:
        with urllib.request.urlopen(PROMPTS_URL.format(split=args.split), timeout=60,
                                    context=_hub_ssl_context()) as r:
            prompts = json.load(r)

    # Get model name from server or use provided
    model = args.model if args.model else get_model_name(base_url)
    print(f"Framework: {args.framework}")
    print(f"Model: {model}")
    print(f"Endpoint: {base_url}")
    print()

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    all_results = {
        "framework": args.framework,
        "framework_version": _framework_version(args.framework),
        "run_config": _run_config(args, concurrency_levels),
        "model": model,
        "endpoint": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": machine_info(),
        "concurrency_results": [],
    }

    # Per-request outputs accumulator. Only populated if --outputs is set; one
    # record per successful request across all concurrency levels.
    all_outputs = []

    t_total_start = time.perf_counter()

    skip_remaining = False
    for conc in concurrency_levels:
        if skip_remaining:
            print(f"--- Concurrency: {conc} --- SKIPPED (previous level exceeded {args.max_wall_time}s)")
            all_results["concurrency_results"].append(
                {"concurrency": conc, "num_requests": args.requests,
                 "successful": 0, "failed": args.requests,
                 "wall_time_s": 0, "error": "skipped (previous level too slow)"}
            )
            print()
            continue

        print(f"--- Concurrency: {conc} ---")
        try:
            good, errors, wall_time = asyncio.run(run_concurrent(
                base_url, model, prompts, conc, args.requests, args.warmup
            ))
        except Exception as e:
            print(f"  SERVER CRASHED: {e}")
            all_results["concurrency_results"].append(
                {"concurrency": conc, "num_requests": args.requests,
                 "successful": 0, "failed": args.requests,
                 "wall_time_s": 0, "error": f"server crashed: {e}"}
            )
            print()
            continue
        summary = summarize(good, errors, conc, args.requests, wall_time)

        # Sanity checks
        prompts_used = [prompts[i % len(prompts)] for i in range(args.requests)]
        sanity_warnings = validate_results(good, summary, prompts_used)
        if sanity_warnings:
            summary["sanity_warnings"] = sanity_warnings

        all_results["concurrency_results"].append(summary)

        if args.outputs:
            for r in good:
                all_outputs.append({
                    "concurrency": conc,
                    "request_idx": r.request_idx,
                    "prompt_name": r.prompt_name,
                    "content": r.content,
                    "reasoning": r.reasoning,
                    "tool_call_deltas": r.tool_call_deltas,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "ttft_s": r.ttft,
                    "e2e_s": r.total_time,
                    "tokens_generated": r.tokens_generated,
                })

        # Print summary
        failed = summary.get('failed', 0)
        fail_str = f" | FAILED: {failed}" if failed else ""
        print(f"  TTFT avg: {summary.get('ttft_avg_ms', 0):.1f}ms | "
              f"Throughput avg: {summary.get('throughput_avg_tps', 0):.1f} tok/s | "
              f"Output: {summary.get('output_throughput_tps', 0):.1f} tok/s | "
              f"ITL avg: {summary.get('itl_avg_ms', 0):.1f}ms | "
              f"Wall: {wall_time:.1f}s{fail_str}")
        if wall_time > args.max_wall_time:
            print(f"  ⚠ Wall time ({wall_time:.0f}s) exceeded limit ({args.max_wall_time}s). Skipping remaining levels.")
            skip_remaining = True
        print()

    total_duration = time.perf_counter() - t_total_start
    all_results["total_duration_s"] = total_duration
    print(f"Total benchmark duration: {total_duration:.1f}s")

    # Save results
    if args.output:
        output_path = Path(args.output)
    elif args.results_dir:
        output_path = Path(args.results_dir) / f"{args.framework}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_path = Path(__file__).parent.parent / "results" / f"{args.framework}_{time.strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {output_path}")

    if args.outputs:
        outputs_path = Path(args.outputs)
        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(outputs_path, "w") as f:
            for record in all_outputs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Outputs saved to: {outputs_path} ({len(all_outputs)} records)")

    return all_results


if __name__ == "__main__":
    main()
