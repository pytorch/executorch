# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline-authored HTTP benchmark; never executes fixture or generated code."""

import argparse
import base64
import concurrent.futures
import copy
import hashlib
import http.client
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=True, allow_nan=False, indent=2)
        f.write("\n")
        name = f.name
    os.replace(name, path)


def load_fixtures(path):
    raw = path.read_bytes()
    fixture = json.loads(raw)
    tasks = fixture["tasks"]
    if fixture["schema_version"] != 1 or len(tasks) != 4:
        raise ValueError("Expected schema 1 and exactly four tasks")
    if len({t["id"] for t in tasks}) != 4:
        raise ValueError("Task IDs must be unique")
    # Validate every raw source before returning any payload for use.
    for task in tasks:
        source = path.parent / task["source_file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != task["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {task['id']}")
    for task in tasks:
        p = task["payload"]
        if p["stream"] is not True or p["max_tokens"] != 512 or p["temperature"] != 0:
            raise ValueError("Expected stream=true, max_tokens=512, temperature=0")
        if not isinstance(p["model"], str) or not isinstance(p["messages"], list):
            raise ValueError("Invalid model/messages")
    return tasks, hashlib.sha256(raw).hexdigest()


class SSE:
    """Incremental byte parser; timestamps refer to receipt of event delimiter."""

    def __init__(self):
        self.buffer = b""
        self.data = []
        self.skip_lf = False

    def feed(self, chunk):
        if self.skip_lf and chunk:
            if chunk.startswith(b"\n"):
                chunk = chunk[1:]
            self.skip_lf = False
        self.buffer += chunk
        while match := re.search(rb"\r\n|\r|\n", self.buffer):
            delimiter = self.buffer[match.start() : match.end()]
            line, self.buffer = self.buffer[: match.start()], self.buffer[match.end() :]
            # A trailing CR is complete; absorb its optional LF in the next chunk.
            self.skip_lf = delimiter == b"\r" and not self.buffer
            if not line:
                if self.data:
                    data, self.data = b"\n".join(self.data), []
                    yield data.decode("utf-8")
            elif not line.startswith(b":"):
                field, sep, value = line.partition(b":")
                if field == b"data":
                    self.data.append(value[1:] if value.startswith(b" ") else value)


def consume(chunks, result):
    parser = SSE()
    done = False
    for stamp, chunk in chunks:
        result["raw_chunks"].append(
            {"received_ns": stamp, "base64": base64.b64encode(chunk).decode("ascii")}
        )
        for data in parser.feed(chunk):
            event = {"received_ns": stamp, "data": data}
            result["events"].append(event)
            if done:
                raise ValueError("Event after [DONE]")
            if data == "[DONE]":
                done = True
                result["finish_ns"] = stamp
                continue
            obj = json.loads(data)
            event["decoded"] = obj
            if "error" in obj:
                raise ValueError(f"Stream error: {obj['error']}")
            choices = obj["choices"]
            if not isinstance(choices, list) or len(choices) > 1:
                raise ValueError("Expected zero or one choice")
            if obj.get("usage") is not None:
                n = obj["usage"]["completion_tokens"]
                if type(n) is not int or n < 0:
                    raise ValueError("Invalid API completion_tokens")
                result["usage"] = obj["usage"]
            for choice in choices:
                if choice.get("index", 0) != 0:
                    raise ValueError("Unexpected choice index")
                for field in ("reasoning_content", "content"):
                    text = choice.get("delta", {}).get(field)
                    if text is not None and not isinstance(text, str):
                        raise ValueError(f"Invalid delta {field}")
                    if text:
                        result[field] += text
                        key = (
                            "first_reasoning_ns"
                            if field == "reasoning_content"
                            else "first_content_ns"
                        )
                        if result[key] is None:
                            result[key] = stamp
                        if result["first_visible_delta_ns"] is None:
                            result["first_visible_delta_ns"] = stamp
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = choice["finish_reason"]
    if not done or parser.buffer or parser.data:
        raise ValueError("Missing [DONE] or unterminated SSE event")
    if result["usage"] is None:
        raise ValueError("Missing API usage; no token estimates permitted")
    if not isinstance(result["finish_reason"], str) or not result["finish_reason"]:
        raise ValueError("Missing finish_reason")


def metrics(result):
    first, finish = result["first_visible_delta_ns"], result["finish_ns"]
    n = result["usage"]["completion_tokens"]
    return {
        "client_ttft_s_not_gpu_first_token": (
            None if first is None else (first - result["start_ns"]) / 1e9
        ),
        "client_avg_tpot_s": (
            None if first is None or n <= 1 else (finish - first) / 1e9 / (n - 1)
        ),
        "total_finish_s": (finish - result["start_ns"]) / 1e9,
        "completion_tokens_api_only": n,
    }


def request_one(task, fixture_sha, args, barrier=None):
    payload = copy.deepcopy(task["payload"])
    if args.include_usage:
        payload.setdefault("stream_options", {})["include_usage"] = True
    wire = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
    result = {
        "task": task,
        "fixture_sha256": fixture_sha,
        "mode": args.mode,
        "wire_payload": payload,
        "wire_body_utf8": wire.decode("utf-8"),
        "usage_instrumentation": args.include_usage,
        "raw_chunks": [],
        "events": [],
        "reasoning_content": "",
        "content": "",
        "usage": None,
        "finish_reason": None,
        "first_visible_delta_ns": None,
        "first_reasoning_ns": None,
        "first_content_ns": None,
        "finish_ns": None,
        "success": False,
        "error": None,
        "metrics": None,
    }
    url = urlsplit(args.base_url)
    conn_type = (
        http.client.HTTPSConnection
        if url.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_type(url.hostname, url.port, timeout=args.timeout)
    if barrier is not None:
        barrier.wait()
    result["start_ns"] = time.perf_counter_ns()
    try:
        conn.request(
            "POST",
            url.path.rstrip("/") + "/v1/chat/completions",
            body=wire,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = conn.getresponse()
        result["http_status"] = response.status
        result["http_headers"] = response.getheaders()

        def chunks():
            while chunk := response.read1(65536):
                yield time.perf_counter_ns(), chunk

        if response.status != 200 or not response.getheader(
            "Content-Type", ""
        ).lower().startswith("text/event-stream"):
            for stamp, chunk in chunks():
                result["raw_chunks"].append(
                    {
                        "received_ns": stamp,
                        "base64": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            raise ValueError(
                "Expected HTTP 200 text/event-stream (redirects not followed)"
            )
        consume(chunks(), result)
        result["metrics"] = metrics(result)
        result["success"] = True
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["finish_ns"] = time.perf_counter_ns()
    finally:
        conn.close()
    return result


def group_metrics(results):
    makespan = (
        max(r["finish_ns"] for r in results) - min(r["start_ns"] for r in results)
    ) / 1e9
    success = all(r["success"] for r in results)
    tokens = sum(r["usage"]["completion_tokens"] for r in results) if success else None
    return {
        "success": success,
        "makespan_s": makespan,
        "completion_tokens_api_only": tokens,
        "aggregate_completion_tokens_per_s": (
            tokens / makespan if success and makespan > 0 else None
        ),
    }


def run_group(tasks, fixture_sha, args):
    if args.concurrency == 1:
        return [request_one(t, fixture_sha, args) for t in tasks]
    barrier = threading.Barrier(4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(request_one, t, fixture_sha, args, barrier) for t in tasks
        ]
        return [f.result() for f in futures]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=1)
    parser.add_argument(
        "--mode",
        required=True,
        help="Descriptive label for externally configured server mode",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--include-usage",
        action="store_true",
        help="Add stream_options.include_usage=true instrumentation",
    )
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.query
        or url.fragment
        or url.username
    ):
        parser.error(
            "base-url must be an HTTP(S) server URL without credentials/query/fragment"
        )
    if not args.mode.strip() or not 0 < args.timeout < float("inf"):
        parser.error("mode must be nonempty and timeout finite and positive")
    tasks, fixture_sha = load_fixtures(args.fixtures)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = run_group(tasks, fixture_sha, args)
    for index, result in enumerate(results, 1):
        atomic_json(args.output_dir / f"request_{index}.json", result)
    summary = {
        "mode": args.mode,
        "concurrency": args.concurrency,
        "fixture_sha256": fixture_sha,
        **group_metrics(results),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
