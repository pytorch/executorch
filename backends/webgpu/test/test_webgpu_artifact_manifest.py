# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

from executorch.backends.webgpu.scripts.webgpu_artifact_manifest import (
    create_manifest,
    validate_manifest,
)


class WebGPUArtifactManifestTest(unittest.TestCase):
    def test_round_trip_preserves_ordered_ptds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in {
                "runner.js": b"js",
                "runner.wasm": b"wasm",
                "model.pte": b"pte",
                "first.ptd": b"first",
                "second.ptd": b"second",
            }.items():
                (root / name).write_bytes(data)
            manifest = create_manifest(
                root,
                {
                    "javascript": Path("runner.js"),
                    "wasm": Path("runner.wasm"),
                    "pte": Path("model.pte"),
                },
                [Path("first.ptd"), Path("second.ptd")],
            )
            validate_manifest(root, manifest)
            self.assertEqual(manifest["ptd_order"], ["first.ptd", "second.ptd"])
            self.assertNotIn(str(root), str(manifest))

    def test_tampered_bytes_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "runner.wasm"
            artifact.write_bytes(b"before")
            manifest = create_manifest(root, {"wasm": artifact})
            artifact.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_manifest(root, manifest)

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "outside.pte"
            outside.write_bytes(b"outside")
            (root / "escape.pte").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                create_manifest(root, {"pte": Path("escape.pte")})

    def test_in_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.pte"
            target.write_bytes(b"model")
            (root / "alias.pte").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                create_manifest(root, {"pte": Path("alias.pte")})

    def test_ptd_reordering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.ptd").write_bytes(b"first")
            (root / "second.ptd").write_bytes(b"second")
            manifest = create_manifest(
                root, {}, [Path("first.ptd"), Path("second.ptd")]
            )
            manifest["ptd_order"] = ["second.ptd", "first.ptd"]
            with self.assertRaisesRegex(ValueError, "order"):
                validate_manifest(root, manifest)
