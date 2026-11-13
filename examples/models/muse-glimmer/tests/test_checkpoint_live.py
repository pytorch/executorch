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

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from executorch.examples.llm_server.python.checkpoint import (
    checkpoint_identity,
    load_checkpoint,
    save_checkpoint,
)
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.session_store import SessionStore
from executorch.examples.llm_server.python.worker_client import WorkerError


def _checkpoint_artifacts(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--pte", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data_path", default="")
    parser.add_argument("--pos_embed_path", default="")
    paths, settings = parser.parse_known_args(argv[1:])
    tokenizer = Path(paths.tokenizer)
    tokenizer_files = (
        [tokenizer / "tokenizer.json"]
        + [
            tokenizer / name
            for name in ("tokenizer_config.json", "special_tokens_map.json")
            if (tokenizer / name).exists()
        ]
        if tokenizer.is_dir()
        else [tokenizer]
    )
    files = [paths.pte, *(str(path) for path in tokenizer_files)]
    files.extend(path for path in (paths.data_path, paths.pos_embed_path) if path)
    return files, {"worker_flags": settings}


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("tokenizer_directory", [False, True])
def test_live_identity_includes_all_worker_artifacts(
    tmp_path, equals, tokenizer_directory
):
    command = ["worker"]
    paths = []
    for name in ("pte", "tokenizer", "data_path", "pos_embed_path"):
        path = tmp_path / name
        if name == "tokenizer" and tokenizer_directory:
            path.mkdir()
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ):
                artifact = path / filename
                artifact.write_bytes(filename.encode())
                paths.append(artifact)
        else:
            path.write_bytes(name.encode())
            paths.append(path)
        command.extend([f"--{name}={path}"] if equals else [f"--{name}", str(path)])
    command.append("--dflash=true")
    files, configuration = _checkpoint_artifacts(command)
    assert files == [str(path) for path in paths]
    assert configuration == {"worker_flags": ["--dflash=true"]}
    identity = checkpoint_identity(files, configuration)
    for path in paths:
        before = path.read_bytes()
        path.write_bytes(before + b"changed")
        assert checkpoint_identity(files, configuration) != identity
        path.write_bytes(before)


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
        argv = json.loads(command)
        model_files, configuration = _checkpoint_artifacts(argv)
        store = await SessionStore.create(
            tmp_path / "sessions",
            model_files=model_files,
            configuration=configuration,
        )
        worker = await spawn_multiplexed_worker(argv, stderr=log, session_store=store)
        try:
            stats = await generate(worker, "restored", prefix, 24)
            await worker.save_session("restored")
            checkpoint = await asyncio.to_thread(store.load, "restored")
            position = checkpoint.manifest["position"]
            committed = position - stats.num_prompt_tokens
            assert (
                stats.num_generated_tokens - 1
                <= committed
                <= stats.num_generated_tokens
            )
            assert (
                list(checkpoint.tokens[stats.num_prompt_tokens :])
                == stats.generated_token_ids[:committed]
            )
            save_checkpoint(checkpoint, tmp_path / "state.etcp")
        finally:
            await worker.close()

        checkpoint = load_checkpoint(
            tmp_path / "state.etcp", expected_identity=store.identity
        )
        worker = await spawn_multiplexed_worker(argv, stderr=log, session_store=store)
        try:
            assert await worker.resume_session("restored") == "restored"
            restored_state = await worker.export_checkpoint_to_cpu(
                "restored", position=position, identity=store.identity
            )
            assert restored_state == checkpoint
            del restored_state
            prompt = list(checkpoint.tokens) + suffix
            actual = await generate(worker, "restored", prompt, 16)
            expected = await generate(worker, "cold", prompt, 16)
            assert actual.generated_token_ids == expected.generated_token_ids
            assert actual.reused_prompt_tokens == position
            assert actual.prefilled_prompt_tokens == len(suffix)
            await worker.close_session("restored")
            await worker.close_session("cold")

            await worker.restore_checkpoint_from_cpu(
                checkpoint,
                "retry",
                position=position - 1,
                expected_identity=store.identity,
            )
            prompt = list(checkpoint.tokens)
            actual = await generate(worker, "retry", prompt, 16)
            expected = await generate(worker, "retry-cold", prompt, 16)
            assert actual.generated_token_ids == expected.generated_token_ids
            assert actual.reused_prompt_tokens == position - 1
            assert actual.prefilled_prompt_tokens == 1
            await worker.close_session("retry")
            await worker.close_session("retry-cold")

            boundaries = [
                layer["begin"] + layer["window"] - (cache["dflash_block_size"] is None)
                for cache in checkpoint.manifest["caches"]
                for layer in cache["layers"]
                if layer["begin"] > 0
            ]
            partial = max(boundaries) if boundaries else position // 2
            if boundaries:
                with pytest.raises(ValueError, match="no longer retains"):
                    checkpoint.validate_restore_position(partial - 1)
                # Bypass Python preflight to verify the native cache rejects it
                # without reserving the destination session key.
                staging = tmp_path / "native-restore"
                staging.write_bytes(checkpoint.data)
                request = worker._submit(
                    {
                        "op": "restore_checkpoint",
                        "session_id": "partial",
                        "position": partial - 1,
                        "path": str(staging),
                        "data_offset": 0,
                        "manifest": checkpoint.manifest,
                    },
                    op="restore_checkpoint",
                    ack="restored",
                )
                try:
                    result = await request.completion
                    assert isinstance(result, WorkerError)
                    assert result.code == "invalid_argument"
                finally:
                    worker._abandon(request)
            await worker.restore_checkpoint_from_cpu(
                checkpoint,
                "partial",
                position=partial,
                expected_identity=store.identity,
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
