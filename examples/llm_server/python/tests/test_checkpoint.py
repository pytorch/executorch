# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace

import pytest
from executorch.examples.llm_server.python.checkpoint import (
    checkpoint_identity,
    CPUCheckpoint,
    load_checkpoint,
    save_checkpoint,
)


def sample(tokens=(11, 12, 13), *, begin=0, window=0, dflash=None):
    position = len(tokens)
    token_data = struct.pack(f"<{position}i", *tokens)
    size = (position - begin) * 2
    offset = len(token_data)
    return CPUCheckpoint(
        "model-a",
        {
            "version": 2,
            "format": "test-v1",
            "position": position,
            "tokens": [0, offset],
            "data_bytes": offset + 2 * size,
            "hidden_row_bytes": 0,
            "hidden": [offset + 2 * size, 0],
            "caches": [
                {
                    "position": position,
                    "dtype": 5,
                    "dflash_block_size": dflash,
                    "layers": [
                        {
                            "begin": begin,
                            "heads": 1,
                            "head_dim": 1,
                            "window": window,
                            "key": [offset, size],
                            "value": [offset + size, size],
                        }
                    ],
                }
            ],
        },
        token_data + b"\x01\x02" * size,
    )


def test_roundtrip_and_explicit_partial_match(tmp_path):
    path = tmp_path / "prefix.etcp"
    before = sample()
    save_checkpoint(before, path)
    after = load_checkpoint(path, expected_identity="model-a")
    assert after == before
    assert after.common_prefix_length([11, 12, 99]) == 2
    after.validate_restore_position(2)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, expected_identity="model-b")
    with pytest.raises(ValueError, match="exceeds limit"):
        load_checkpoint(path, expected_identity="model-a", max_bytes=11)


def dflash_sample():
    target = sample(tuple(range(10)))
    manifest = copy.deepcopy(target.manifest)
    offset = len(target.data)
    manifest["caches"].append(
        {
            "position": 8,
            "dtype": 5,
            "dflash_block_size": 2,
            "layers": [
                {
                    "begin": 4,
                    "heads": 1,
                    "head_dim": 1,
                    "window": 2,
                    "key": [offset, 8],
                    "value": [offset + 8, 8],
                }
            ],
        }
    )
    manifest["hidden_row_bytes"] = 4
    manifest["hidden"] = [offset + 16, 8]
    data = target.data + b"\x02\x03" * 8 + b"features"
    manifest["data_bytes"] = len(data)
    return CPUCheckpoint(target.identity, manifest, data)


def test_two_cache_roundtrip_and_restore_across_draft_frontier(tmp_path):
    checkpoint = dflash_sample()
    path = tmp_path / "dflash.etcp"
    save_checkpoint(checkpoint, path)
    restored = load_checkpoint(path, expected_identity=checkpoint.identity)
    assert restored == checkpoint
    for position in (6, 7, 8, 9, 10):
        restored.validate_restore_position(position)
    with pytest.raises(ValueError, match="no longer retains"):
        restored.validate_restore_position(5)


@pytest.mark.parametrize(
    "kind", ["short", "extra", "offset", "zero_row", "draft_ahead", "single_cache_row"]
)
def test_draft_hidden_state_must_match_both_cache_positions(kind):
    checkpoint = dflash_sample()
    if kind == "short":
        checkpoint = replace(checkpoint, data=checkpoint.data[:-1])
        checkpoint.manifest["data_bytes"] -= 1
    elif kind == "extra":
        checkpoint = replace(checkpoint, data=checkpoint.data + b"\x00")
        checkpoint.manifest["data_bytes"] += 1
        checkpoint.manifest["hidden"][1] += 1
    elif kind == "offset":
        checkpoint.manifest["hidden"][0] -= 1
    elif kind == "zero_row":
        checkpoint.manifest["hidden_row_bytes"] = 0
    elif kind == "draft_ahead":
        checkpoint.manifest["caches"][1]["position"] = 11
    else:
        checkpoint = sample()
        checkpoint.manifest["hidden_row_bytes"] = 4
    with pytest.raises(ValueError):
        checkpoint.validate(checkpoint.identity)


def test_missing_dflash_metadata_is_rejected_before_restore(tmp_path):
    checkpoint = sample(tuple(range(10)), begin=7, window=3)
    del checkpoint.manifest["caches"][0]["dflash_block_size"]
    with pytest.raises(ValueError, match="cache record"):
        checkpoint.validate_restore_position(10)
    with pytest.raises(ValueError, match="cache record"):
        save_checkpoint(checkpoint, tmp_path / "invalid.etcp")


@pytest.mark.parametrize("kind", ["non_ascii_digest", "nested_header"])
def test_malformed_header_reports_value_error(tmp_path, kind):
    path = tmp_path / "invalid.etcp"
    save_checkpoint(sample(), path)
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[8:16])[0]
    header = json.loads(raw[16 : 16 + header_size])
    header["sha256"] = "é" * 64
    data = (
        json.dumps(header).encode()
        if kind == "non_ascii_digest"
        else b"[" * 2000 + b"0" + b"]" * 2000
    )
    path.write_bytes(raw[:8] + struct.pack("<Q", len(data)) + data + sample().data)
    with pytest.raises(ValueError, match="invalid checkpoint header"):
        load_checkpoint(path, expected_identity="model-a")


@pytest.mark.skipif(os.name != "posix", reason="POSIX file types")
@pytest.mark.parametrize("kind", ["fifo", "symlink", "directory"])
def test_load_rejects_nonregular_files_without_waiting(tmp_path, kind):
    path = tmp_path / "untrusted.etcp"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        saved = tmp_path / "valid.etcp"
        save_checkpoint(sample(), saved)
        path.symlink_to(saved)
    else:
        path.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from executorch.examples.llm_server.python.checkpoint "
            "import load_checkpoint; load_checkpoint(sys.argv[1], expected_identity='model-a')",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert (
        "regular file" in result.stderr
        if kind != "symlink"
        else "OSError" in result.stderr
    )


@pytest.mark.parametrize("directory", [False, True])
def test_load_closes_descriptor_before_file_object_owns_it(
    tmp_path, monkeypatch, directory
):
    path = tmp_path if directory else tmp_path / "state.etcp"
    if not directory:
        save_checkpoint(sample(), path)
    opened = []
    original = os.open

    def record_open(*args, **kwargs):
        fd = original(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail_fdopen(*args, **kwargs):
        raise ValueError("file object construction failed")

    monkeypatch.setattr(os, "open", record_open)
    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    with pytest.raises(ValueError, match="regular file|construction failed"):
        load_checkpoint(path, expected_identity="model-a")
    assert len(opened) == 1
    try:
        with pytest.raises(OSError):
            os.fstat(opened[0])
    finally:
        try:
            os.close(opened[0])
        except OSError:
            pass


@pytest.mark.parametrize(
    "kind", ["payload", "tokens", "truncate", "header_limit", "version"]
)
def test_corruption_rejected(tmp_path, kind):
    path = tmp_path / "prefix.etcp"
    save_checkpoint(sample(), path)
    data = bytearray(path.read_bytes())
    if kind == "payload":
        data[-1] ^= 1
    elif kind == "tokens":
        size = struct.unpack("<Q", data[8:16])[0]
        data[16 + size] ^= 1
    elif kind == "truncate":
        data = data[:-1]
    elif kind == "header_limit":
        data[8:16] = struct.pack("<Q", 2**40)
    else:
        data[7] = ord("9")
    path.write_bytes(data)
    with pytest.raises(ValueError):
        load_checkpoint(path, expected_identity="model-a")


def test_failed_save_preserves_previous_file(tmp_path):
    path = tmp_path / "prefix.etcp"
    before = sample()
    save_checkpoint(before, path)
    invalid = copy.deepcopy(before)
    invalid.manifest["caches"][0]["layers"][0]["key"][1] += 1
    with pytest.raises(ValueError):
        save_checkpoint(invalid, path)
    assert load_checkpoint(path, expected_identity="model-a") == before


def test_layer_retention_controls_partial_restore():
    checkpoint = sample(tuple(range(10)), begin=7, window=3)
    checkpoint.validate_restore_position(10)
    checkpoint.validate_restore_position(9)
    with pytest.raises(ValueError, match="no longer retains"):
        checkpoint.validate_restore_position(8)


def test_identity_tracks_contents_and_configuration_not_paths(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"model")
    b.write_bytes(b"model")
    assert checkpoint_identity([a], {"rope": 1}) == checkpoint_identity(
        [b], {"rope": 1}
    )
    assert checkpoint_identity([a], {"rope": 1}) != checkpoint_identity(
        [a], {"rope": 2}
    )
    b.write_bytes(b"changed weights")
    assert checkpoint_identity([a], {}) != checkpoint_identity([b], {})


@pytest.mark.parametrize("restore", [False, True])
def test_repeated_cancellation_waits_for_native_transfer(restore):
    import asyncio
    from pathlib import Path

    from executorch.examples.llm_server.python.tests.test_multiplexed_worker_client import (
        _fake_client,
    )

    async def run():
        async with _fake_client() as (client, proc):
            checkpoint = sample()
            if restore:
                task = asyncio.create_task(
                    client.restore_checkpoint_from_cpu(
                        checkpoint, "new", position=3, expected_identity="model-a"
                    )
                )
            else:
                task = asyncio.create_task(
                    client.export_checkpoint_to_cpu(
                        "seed", position=3, identity="model-a"
                    )
                )
            request = await proc.stdin.frames.get()
            path = Path(request["path"])
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert path.exists() and not task.done()
            if restore:
                proc.send(request["request_id"], restored=True)
            else:
                path.write_bytes(checkpoint.data)
                proc.send(
                    request["request_id"], exported=True, manifest=checkpoint.manifest
                )
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not path.exists() and not client._requests

    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize("block", [0, 2])
def test_dflash_requires_the_full_window_before_the_next_block(block):
    checkpoint = sample(tuple(range(10)), begin=7, window=3, dflash=block)
    with pytest.raises(ValueError, match="no longer retains"):
        checkpoint.validate_restore_position(9)
    checkpoint.validate_restore_position(10)


def test_long_history_uses_binary_payload_and_bounded_metadata(tmp_path):
    checkpoint = sample((123456,) * 300000, begin=299997, window=3)
    assert checkpoint.tokens == (123456,) * 300000
    assert len(json.dumps(checkpoint.manifest)) < 1024
    path = tmp_path / "long.etcp"
    save_checkpoint(checkpoint, path)
    assert load_checkpoint(path, expected_identity="model-a") == checkpoint
    checkpoint.validate_restore_position(300000)


@pytest.mark.parametrize("kind", ["version", "extent", "negative_token"])
def test_binary_token_metadata_rejects_invalid_state(kind):
    checkpoint = sample()
    if kind == "version":
        checkpoint.manifest["version"] = 1
    elif kind == "extent":
        checkpoint.manifest["tokens"][1] -= 4
    else:
        checkpoint = CPUCheckpoint(
            checkpoint.identity,
            checkpoint.manifest,
            struct.pack("<i", -1) + checkpoint.data[4:],
        )
    with pytest.raises(ValueError):
        checkpoint.validate("model-a")
