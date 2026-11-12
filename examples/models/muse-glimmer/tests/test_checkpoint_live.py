# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real model restart/continuation checks for an explicitly configured worker.

Set MUSE_GLIMMER_CHECKPOINT_COMMAND to a JSON argv array for the batching worker,
and MUSE_GLIMMER_HF_DIR to the tokenizer directory. Use greedy generation, a
context capacity of at least 8192, at least three sessions, and disable the shared
prefix cache. Run once with DFlash and once without it.
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from executorch.examples.llm_server.python.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)


@pytest.mark.parametrize("prefix_length", [512, 4096])
def test_restart_matches_cold_continuation(tmp_path, prefix_length):
    command = os.environ.get("MUSE_GLIMMER_CHECKPOINT_COMMAND")
    hf_dir = os.environ.get("MUSE_GLIMMER_HF_DIR")
    if not command or not hf_dir:
        pytest.skip("requires MUSE_GLIMMER_CHECKPOINT_COMMAND and MUSE_GLIMMER_HF_DIR")
    tokenizer = pytest.importorskip("tokenizers").Tokenizer.from_file(
        str(Path(hf_dir) / "tokenizer.json")
    )
    prefix = tokenizer.encode(
        "Review this code for correctness:\n"
        + "def add(a, b):\n    return a - b\n\n" * 600,
        add_special_tokens=False,
    ).ids[:prefix_length]
    suffix = tokenizer.encode("\nExplain the fix:\n", add_special_tokens=False).ids

    async def generate(worker, key, tokens, count):
        stream = worker.generate(
            "",
            SimpleNamespace(
                session_id=key,
                prompt_segments=[{"ids": tokens}],
                max_new_tokens=count,
                temperature=0,
                ignore_eos=True,
            ),
        )
        async for _ in stream:
            pass
        return await stream.wait()

    async def run(log):
        worker = await spawn_multiplexed_worker(json.loads(command), stderr=log)
        try:
            stats = await generate(worker, "seed", prefix, 24)
            position = stats.num_prompt_tokens + stats.num_generated_tokens - 1
            checkpoint = await worker.export_checkpoint_to_cpu(
                "seed", position=position, identity="same-worker-test"
            )
            save_checkpoint(checkpoint, tmp_path / "state.etcp")
        finally:
            await worker.close()

        checkpoint = load_checkpoint(
            tmp_path / "state.etcp", expected_identity="same-worker-test"
        )
        worker = await spawn_multiplexed_worker(json.loads(command), stderr=log)
        try:
            await worker.restore_checkpoint_from_cpu(
                checkpoint,
                "restored",
                position=position,
                expected_identity="same-worker-test",
            )
            prompt = list(checkpoint.tokens) + suffix
            actual = await generate(worker, "restored", prompt, 16)
            expected = await generate(worker, "cold", prompt, 16)
            assert actual.generated_token_ids == expected.generated_token_ids
            assert actual.reused_prompt_tokens == position
            assert actual.prefilled_prompt_tokens == len(suffix)
            await worker.close_session("restored")
            await worker.close_session("cold")

            retained = any(
                layer["begin"] > 0
                for cache in checkpoint.manifest["caches"]
                for layer in cache["layers"]
            )
            if retained:
                with pytest.raises(ValueError, match="no longer retains"):
                    checkpoint.validate_restore_position(1)
            else:
                partial = position // 2
                await worker.restore_checkpoint_from_cpu(
                    checkpoint,
                    "partial",
                    position=partial,
                    expected_identity="same-worker-test",
                )
                prompt = list(checkpoint.tokens[:partial]) + suffix
                actual = await generate(worker, "partial", prompt, 16)
                expected = await generate(worker, "partial-cold", prompt, 16)
                assert actual.generated_token_ids == expected.generated_token_ids
                assert actual.reused_prompt_tokens == partial
        finally:
            await worker.close()

    with (tmp_path / "worker.log").open("w") as log:
        asyncio.run(asyncio.wait_for(run(log), 600))
