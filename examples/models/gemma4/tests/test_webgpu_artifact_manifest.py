# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import json
import tempfile
import unittest

from pathlib import Path
from typing import Any
from unittest import mock

from executorch.examples.models.gemma4 import (
    webgpu_artifact_manifest as gemma4_manifest,
)

from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    ARCHITECTURE_FINGERPRINT,
    create_plain_manifest,
    validate_plain_manifest,
)


def _set_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _test_source_manifest() -> dict[str, Any]:
    logical_path = "examples/models/gemma4/webgpu_artifact_manifest.py"
    identity = {"bytes": 7, "sha256": "3" * 64}
    return {
        "checkouts": {
            "fbsource": {"clean": True, "head": "1" * 40},
            "oss": {"clean": True, "head": "2" * 40},
        },
        "file_set_sha256": _set_digest([logical_path]),
        "files": [
            {
                "copies": {
                    "fbcode": {
                        **identity,
                        "path": f"fbcode/executorch/{logical_path}",
                    },
                    "oss": {**identity, "path": logical_path},
                    "xplat": {
                        **identity,
                        "path": f"xplat/executorch/{logical_path}",
                    },
                },
                "path": logical_path,
            }
        ],
        "schema_version": 1,
    }


def _test_wgsl_manifest() -> dict[str, Any]:
    roles = (
        ("runtime/WebGPUShaderRegistry.cpp", "global_registry"),
        ("runtime/ops/add/binary_add.wgsl", "wgsl"),
        ("runtime/ops/add/binary_add_wgsl.h", "generated_header"),
        ("scripts/gen_wgsl_headers.py", "generator"),
    )
    files = [
        {"bytes": 7, "path": path, "role": role, "sha256": "4" * 64}
        for path, role in roles
    ]
    return {
        "fbsource_commit": "1" * 40,
        "file_set_sha256": _set_digest(
            [{"path": path, "role": role} for path, role in roles]
        ),
        "files": files,
        "orphans": [],
        "schema_version": 1,
    }


def _sealed_source_receipt() -> dict[str, Any]:
    return {
        "fbsource_commit": "1" * 40,
        "oss_commit": "2" * 40,
        "schema_version": 3,
        "source_current": True,
        "source_manifest": _test_source_manifest(),
        "verification": {
            "source_checkout": "verified",
            "wgsl_codegen": "verified",
        },
        "wgsl_manifest": _test_wgsl_manifest(),
    }


class WebGPUArtifactManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.pte = self.root / "model.pte"
        self.ptds = [self.root / f"constants_{index}.ptd" for index in range(3)]
        self.source_receipt = self.root / "source_receipt.json"
        self.pte.write_bytes(b"shared")
        self.ptds[0].write_bytes(b"shared")
        self.ptds[1].write_bytes(b"ptd-one")
        self.ptds[2].write_bytes(b"ptd-two")
        self.source_receipt.write_text(
            json.dumps(_sealed_source_receipt()),
            encoding="utf-8",
        )
        self.manifest = create_plain_manifest(
            self.root,
            {
                "pte": Path(self.pte.name),
                "source": Path(self.source_receipt.name),
            },
            [Path(path.name) for path in self.ptds],
        )

    def test_round_trip_and_order(self) -> None:
        validate_plain_manifest(self.root, self.manifest)
        self.assertEqual(
            self.manifest["ptd_order"], [path.name for path in self.ptds]
        )
        self.assertEqual(
            self.manifest["model"]["architecture"], ARCHITECTURE_FINGERPRINT
        )

    def test_rejects_wrong_bytes_and_hash(self) -> None:
        self.ptds[1].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "byte count|SHA-256"):
            validate_plain_manifest(self.root, self.manifest)

    def test_rejects_missing_and_extra_artifacts(self) -> None:
        self.ptds[2].unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            validate_plain_manifest(self.root, self.manifest)
        self.ptds[2].write_bytes(b"ptd-two")
        (self.root / "extra.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            validate_plain_manifest(self.root, self.manifest)

    def test_rejects_internal_symlink(self) -> None:
        self.pte.unlink()
        self.pte.symlink_to(self.ptds[0].name)
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate_plain_manifest(self.root, self.manifest)

    def test_rejects_architecture_mutation(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["model"]["architecture"]["hidden_size"] += 1
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_plain_manifest(self.root, mutated)

    def test_rejects_unsealed_source_receipt(self) -> None:
        receipt = json.loads(self.source_receipt.read_text(encoding="utf-8"))
        receipt["source_current"] = False
        self.source_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source-current"):
            create_plain_manifest(
                self.root,
                {
                    "pte": Path(self.pte.name),
                    "source": Path(self.source_receipt.name),
                },
                [Path(path.name) for path in self.ptds],
            )

    def test_rejects_source_mirror_mismatch(self) -> None:
        source_manifest = _test_source_manifest()
        source_manifest["files"][0]["copies"]["oss"]["sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "mirror/OSS identity mismatch"):
            gemma4_manifest.validate_source_manifest(source_manifest)

    def test_rejects_source_file_set_digest_mismatch(self) -> None:
        source_manifest = _test_source_manifest()
        source_manifest["file_set_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "file-set identity mismatch"):
            gemma4_manifest.validate_source_manifest(source_manifest)

    def test_rejects_invalid_checkout_head(self) -> None:
        source_manifest = _test_source_manifest()
        source_manifest["checkouts"]["fbsource"]["head"] = "short"
        with self.assertRaisesRegex(ValueError, "checkout identity is invalid"):
            gemma4_manifest.validate_source_manifest(source_manifest)

    def test_rejects_declared_wgsl_orphan(self) -> None:
        wgsl_manifest = _test_wgsl_manifest()
        wgsl_manifest["orphans"] = ["runtime/ops/add/orphan_wgsl.h"]
        with self.assertRaisesRegex(ValueError, "orphan"):
            gemma4_manifest.validate_wgsl_manifest(wgsl_manifest)

    def test_rejects_incomplete_source_verification(self) -> None:
        receipt = _sealed_source_receipt()
        receipt["verification"] = {"source_checkout": "verified"}
        self.source_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification is incomplete"):
            create_plain_manifest(
                self.root,
                {
                    "pte": Path(self.pte.name),
                    "source": Path(self.source_receipt.name),
                },
                [Path(path.name) for path in self.ptds],
            )

    def test_rejects_receipt_wgsl_head_mismatch(self) -> None:
        receipt = _sealed_source_receipt()
        receipt["wgsl_manifest"]["fbsource_commit"] = "9" * 40
        self.source_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatched WGSL checkout"):
            create_plain_manifest(
                self.root,
                {
                    "pte": Path(self.pte.name),
                    "source": Path(self.source_receipt.name),
                },
                [Path(path.name) for path in self.ptds],
            )


class SourceClosureManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.fbsource_root = self.root / "fbsource"
        self.oss_root = self.root / "oss"
        self.fbsource_root.mkdir()
        self.oss_root.mkdir()
        self.logical_path = "examples/models/gemma4/webgpu_artifact_manifest.py"
        for path in (
            self.fbsource_root / "fbcode/executorch" / self.logical_path,
            self.fbsource_root / "xplat/executorch" / self.logical_path,
            self.oss_root / self.logical_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"source")

    def _create_source_manifest(self) -> dict[str, Any]:
        def snapshot(_root: Path, kind: str) -> dict[str, object]:
            return {
                "clean": True,
                "head": ("1" if kind == "fbsource" else "2") * 40,
            }

        with mock.patch.object(
            gemma4_manifest, "_checkout_snapshot", side_effect=snapshot
        ), mock.patch.object(
            gemma4_manifest,
            "_derive_owned_paths",
            return_value=[self.logical_path],
        ):
            return gemma4_manifest.create_source_manifest(
                self.fbsource_root,
                self.oss_root,
            )

    def test_source_manifest_producer_round_trip(self) -> None:
        manifest = self._create_source_manifest()
        gemma4_manifest.validate_source_manifest(manifest)
        self.assertEqual([entry["path"] for entry in manifest["files"]], [self.logical_path])

    def test_owned_union_uses_the_reviewed_plain_summaries(self) -> None:
        summaries = gemma4_manifest._GEMMA_PRODUCTION_DIFF_SUMMARIES
        self.assertEqual(
            summaries,
            (
                "[ExecuTorch][WebGPU] Add shared model runtime prerequisites",
                "[ExecuTorch][Vulkan] Support scoped Gemma symbolic partitioning",
                "[ExecuTorch][WebGPU] Add Gemma 4 plain runtime and guarded routes",
                "[ExecuTorch][WebGPU] Add Gemma 4 plain export and artifact contract",
                "[ExecuTorch][WebGPU] Add plain Gemma 4 source-closure tests",
            ),
        )
        expected_reverse = tuple(reversed(summaries))

        def source_control(argv: list[str], _label: str) -> str:
            if "log" in argv:
                revision = argv[argv.index("-r") + 1]
                offset = 0 if revision == "." else int(revision.removeprefix(".~"))
                return f"{offset + 1:040x}\n{expected_reverse[offset]}\n"
            node = argv[argv.index("--change") + 1]
            offset = int(node, 16) - 1
            path = f"runtime/plain_owned_{offset}.cpp"
            return f"xplat/executorch/{path}\nfbcode/executorch/{path}\n"

        with mock.patch.object(
            gemma4_manifest, "_run_source_control", side_effect=source_control
        ):
            paths = gemma4_manifest._derive_owned_paths(
                self.fbsource_root, summaries
            )
        self.assertEqual(
            paths,
            [f"runtime/plain_owned_{index}.cpp" for index in range(5)],
        )

    def test_create_source_manifest_cli_round_trip(self) -> None:
        output = self.root / "source.json"
        manifest = _test_source_manifest()
        with mock.patch.object(
            gemma4_manifest, "create_source_manifest", return_value=manifest
        ):
            self.assertEqual(
                gemma4_manifest.main(
                    [
                        "create-source-manifest",
                        "--fbsource-root",
                        str(self.fbsource_root),
                        "--oss-root",
                        str(self.oss_root),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
        document = json.loads(output.read_text(encoding="utf-8"))
        gemma4_manifest.validate_source_manifest(document)

    def test_create_wgsl_manifest_cli_round_trip(self) -> None:
        output = self.root / "wgsl.json"
        manifest = _test_wgsl_manifest()
        with mock.patch.object(
            gemma4_manifest, "create_wgsl_manifest", return_value=manifest
        ):
            self.assertEqual(
                gemma4_manifest.main(
                    [
                        "create-wgsl-manifest",
                        "--backend-root",
                        str(self.fbsource_root),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
        document = json.loads(output.read_text(encoding="utf-8"))
        gemma4_manifest.validate_wgsl_manifest(document)

    def test_create_source_receipt_cli_round_trip(self) -> None:
        output = self.root / "receipt.json"
        receipt = _sealed_source_receipt()
        with mock.patch.object(
            gemma4_manifest, "create_source_closure_receipt", return_value=receipt
        ):
            self.assertEqual(
                gemma4_manifest.main(
                    [
                        "create-source-receipt",
                        "--fbsource-root",
                        str(self.fbsource_root),
                        "--oss-root",
                        str(self.oss_root),
                        "--backend-root",
                        str(self.fbsource_root),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)

    def test_wgsl_producer_rejects_stale_generated_output(self) -> None:
        backend_root = self.root / "fbsource/xplat/executorch/backends/webgpu"
        shader = backend_root / "runtime/ops/add/binary_add.wgsl"
        header = backend_root / "runtime/ops/add/binary_add_wgsl.h"
        registry = backend_root / "runtime/WebGPUShaderRegistry.cpp"
        generator_path = backend_root / "scripts/gen_wgsl_headers.py"
        for path, contents in (
            (shader, b"shader"),
            (header, b"stale"),
            (registry, b"registry"),
            (generator_path, b"generator"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)

        class Generator:
            def discover(self) -> list[Path]:
                return [shader]

            def collect_outputs(self) -> tuple[dict[Path, bytes], list[Path]]:
                return {header: b"fresh", registry: b"registry"}, []

            def registry_path(self) -> Path:
                return registry

        with mock.patch.object(
            gemma4_manifest,
            "_checkout_snapshot",
            return_value={"clean": True, "head": "1" * 40},
        ), mock.patch.object(
            gemma4_manifest, "_load_wgsl_generator", return_value=Generator()
        ):
            with self.assertRaisesRegex(ValueError, "generated output is stale"):
                gemma4_manifest.create_wgsl_manifest(backend_root)
