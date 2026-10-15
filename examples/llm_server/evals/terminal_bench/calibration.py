# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure isolated and overlapping requests before admitting agent children."""

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

import httpx

from .inference_cost import fit_cost, ModelCost, validation_error


def calibration_engine(initial):
    engine = initial.get("engine") or {}
    if (
        initial["submitted_generations"]
        or (engine.get("max_decode_sequences") or 0) < 2
    ):
        raise ValueError(
            "Calibration requires an idle server with at least two decode slots"
        )
    if "recent_steps" not in engine:
        raise ValueError("Rebuild the worker with completed-step telemetry")
    return engine


async def calibrate(base_url, model_id, model_cost, output, max_context=32768):
    samples = {}
    sessions = set()
    phase = "warmup"
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/", timeout=180, trust_env=False
    ) as client:
        response = await client.get("runtime/stats")
        response.raise_for_status()
        initial = response.json()
        engine = calibration_engine(initial)
        first_id = max((s["id"] for s in engine["recent_steps"]), default=0)
        stopped = asyncio.Event()

        async def collect_once():
            response = await client.get("runtime/stats")
            response.raise_for_status()
            for sample in response.json()["engine"]["recent_steps"]:
                if sample["id"] > first_id and sample["id"] not in samples:
                    samples[sample["id"]] = sample | {"phase": phase}

        async def collect():
            while True:
                await collect_once()
                if stopped.is_set():
                    return
                await asyncio.sleep(0.2)

        async def generate(words, cap):
            session = "calibration-" + uuid.uuid4().hex
            sessions.add(session)
            prompt = (
                session
                + "\nRead the following reference words, then write a numbered list "
                "of 200 distinct observations about testing software. Continue until the output limit.\n"
                + " reference" * words
            )
            response = await client.post(
                "chat/completions",
                json={
                    "model": model_id,
                    "session_id": session,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": cap,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            result = response.json()
            if result["usage"]["completion_tokens"] < 8:
                raise ValueError(
                    "Calibration response stopped too early to measure decode"
                )
            response = await client.delete("sessions/" + session)
            response.raise_for_status()
            sessions.remove(session)

        async def pair(words, cap):
            async with asyncio.TaskGroup() as group:
                group.create_task(generate(words, cap))
                group.create_task(generate(words, cap))

        sampler = asyncio.create_task(collect())
        try:
            longest = min(8192, max_context // 2)
            print("Calibration: warming solo and paired execution.", flush=True)
            await pair(min(256, longest // 4), 16)
            await collect_once()
            phase = "fit"
            print(
                "Calibration: measuring prefill, solo decode and paired decode.",
                flush=True,
            )
            await generate(min(1024, longest // 4), 64)
            await pair(longest // 2, 96)
            await collect_once()
            phase = "validate"
            print("Calibration: validating at different prompt lengths.", flush=True)
            await generate(min(2048, longest // 2), 64)
            await pair(longest, 96)
        finally:
            stopped.set()
            try:
                await sampler
            finally:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.with_suffix(".steps.json").write_text(
                    json.dumps(
                        sorted(samples.values(), key=lambda s: s["id"]), indent=2
                    )
                    + "\n"
                )
                for session in sessions:
                    response = await client.delete("sessions/" + session)
                    response.raise_for_status()

    ordered = sorted(samples.values(), key=lambda s: s["id"])
    cost = fit_cost(model_cost, [s for s in ordered if s["phase"] == "fit"])
    error = validation_error(cost, [s for s in ordered if s["phase"] == "validate"])
    if error > 0.30:
        raise ValueError(
            f"Calibration validation error {error:.1%} exceeds 30%; see {output.with_suffix('.steps.json')}"
        )
    cost = replace(cost, relative_error=error)
    result = {
        "model_id": model_id,
        "created_at": time.time(),
        "cost": cost.to_dict(),
        "validation_p90_relative_error": error,
        "max_measured_context": max(
            i["position"] + i["tokens"] for s in ordered for i in s["inputs"]
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Inference calibration passed: p90 timing error {error:.1%}. Profile: {output}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(
        calibrate(
            args.base_url,
            args.model,
            ModelCost.from_dict(json.loads(args.profile.read_text())),
            args.output,
            args.max_context,
        )
    )


if __name__ == "__main__":
    main()
