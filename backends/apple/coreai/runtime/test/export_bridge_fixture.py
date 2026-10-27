# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import shutil
from pathlib import Path

import torch

from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import get_lowered_backend_modules


class Arithmetic(torch.nn.Module):
    """Distinct outputs make reordered bindings observable."""

    def forward(self, left, right):
        """Return the elementwise sum and difference, in that order."""
        return left + right, left - right


def export_fixture(output_dir: Path, *, dynamic: bool = False) -> None:
    """Export into a dedicated fixture directory, replacing its contents."""
    size = torch.export.Dim("size", min=1, max=8)
    exported = torch.export.export(
        Arithmetic().eval(),
        (torch.ones(2), torch.ones(2)),
        dynamic_shapes=({0: size}, {0: size}) if dynamic else None,
    )
    lowered = to_edge_transform_and_lower(
        exported,
        transform_passes=get_default_passes(),
        partitioner=[CoreAIPartitioner()],
        compile_config=get_default_compile_config(),
    )
    delegates = get_lowered_backend_modules(lowered.exported_program().graph_module)
    assert len(delegates) == 1, "Expected the fixture to lower to one Core AI delegate"
    manifest_bytes = bytes(delegates[0].processed_bytes)
    manifest = json.loads(manifest_bytes)
    store = delegates[0].named_data_store_output
    assert store is not None
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for relative_path in manifest["files"]:
        entry = store.pte_data[f"coreai/{manifest['hash']}/{relative_path}"]
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(store.buffers[entry.buffer_index])
    (output_dir / "manifest.json").write_bytes(manifest_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path, help="Dedicated fixture directory, replaced on export"
    )
    output_dir = parser.parse_args().output_dir
    export_fixture(output_dir)
    export_fixture(output_dir / "dynamic", dynamic=True)
