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
@pytest.mark.parametrize("include_pte", [False, True])
def test_export_replaces_stale_fixture_files(tmp_path, dynamic, include_pte):
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
        patch.object(fixture, "to_edge_transform_and_lower") as lower,
        patch.object(fixture, "get_lowered_backend_modules", return_value=[delegate]),
        patch.object(fixture, "get_default_passes", return_value=[]),
        patch.object(fixture, "get_default_compile_config"),
        patch.object(fixture, "CoreAIPartitioner"),
        patch.object(
            fixture.torch.export, "export", wraps=fixture.torch.export.export
        ) as export,
    ):
        lower.return_value.to_executorch.return_value.buffer = b"serialized-pte"
        fixture.export_fixture(output, dynamic=dynamic, include_pte=include_pte)
        if include_pte:
            lower.return_value.to_executorch.assert_called_once_with()
            assert (output / "model.pte").read_bytes() == b"serialized-pte"
        else:
            lower.return_value.to_executorch.assert_not_called()

    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert (bundle / "graph.bin").read_bytes() == b"abc"
    assert sorted(
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    ) == [
        "manifest.json",
        "model.aimodel/graph.bin",
    ] + (
        ["model.pte"] if include_pte else []
    )
    shapes = export.call_args.kwargs["dynamic_shapes"]
    if dynamic:
        assert shapes[0][0] is shapes[1][0]
        assert shapes[0][0].min == 1
        assert shapes[0][0].max == 8
    else:
        assert shapes is None
