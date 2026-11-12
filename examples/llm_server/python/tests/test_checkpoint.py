# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import struct

import pytest
from executorch.examples.llm_server.python.checkpoint import (
    checkpoint_identity,
    CPUCheckpoint,
    load_checkpoint,
    save_checkpoint,
)


def sample():
    return CPUCheckpoint(
        "model-a",
        {
            "format": "test-v1",
            "position": 3,
            "tokens": [11, 12, 13],
            "data_bytes": 12,
            "hidden_row_bytes": 0,
            "hidden": [12, 0],
            "caches": [
                {
                    "position": 3,
                    "dtype": 5,
                    "dflash_block_size": None,
                    "layers": [
                        {
                            "begin": 0,
                            "heads": 1,
                            "head_dim": 1,
                            "window": 0,
                            "key": [0, 6],
                            "value": [6, 6],
                        }
                    ],
                }
            ],
        },
        bytes(range(12)),
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
        header = json.loads(data[16 : 16 + size])
        header["metadata"]["manifest"]["tokens"][0] = 19
        encoded = json.dumps(header, separators=(",", ":")).encode()
        data = data[:8] + struct.pack("<Q", len(encoded)) + encoded + data[16 + size :]
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
    checkpoint = sample()
    checkpoint.manifest.update(position=10, tokens=list(range(10)))
    checkpoint.manifest["caches"][0].update(position=10)
    checkpoint.manifest["caches"][0]["layers"][0].update(begin=7, window=3)
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
