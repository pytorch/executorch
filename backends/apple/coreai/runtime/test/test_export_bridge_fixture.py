# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executorch.backends.apple.coreai.runtime.test import (
    export_bridge_fixture as fixture,
)


@pytest.mark.parametrize("dynamic", [False, True])
def test_export_replaces_stale_fixture_files(tmp_path, dynamic):
    """Re-exports must not retain stale assets or freeze dynamic extents."""
    output = tmp_path / "fixture"
    bundle = output / "model.aimodel"
    bundle.mkdir(parents=True)
    (bundle / "obsolete.bin").write_bytes(b"obsolete")
    (output / "stale.json").write_text("{}")
    manifest = {"hash": "ab", "files": {"model.aimodel/graph.bin": 3}}
    manifest_bytes = json.dumps(manifest).encode()
    delegate = SimpleNamespace(
        processed_bytes=manifest_bytes,
        named_data_store_output=SimpleNamespace(
            pte_data={
                "coreai/ab/model.aimodel/graph.bin": SimpleNamespace(buffer_index=0)
            },
            buffers=[b"abc"],
        ),
    )
    with (
        patch.object(fixture, "to_edge_transform_and_lower"),
        patch.object(fixture, "get_lowered_backend_modules", return_value=[delegate]),
        patch.object(fixture, "get_default_passes", return_value=[]),
        patch.object(fixture, "get_default_compile_config"),
        patch.object(fixture, "CoreAIPartitioner"),
        patch.object(
            fixture.torch.export, "export", wraps=fixture.torch.export.export
        ) as export,
    ):
        fixture.export_fixture(output, dynamic=dynamic)

    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert (bundle / "graph.bin").read_bytes() == b"abc"
    assert sorted(
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    ) == [
        "manifest.json",
        "model.aimodel/graph.bin",
    ]
    shapes = export.call_args.kwargs["dynamic_shapes"]
    if dynamic:
        assert shapes[0][0] is shapes[1][0]
        assert shapes[0][0].min == 1
        assert shapes[0][0].max == 8
    else:
        assert shapes is None
