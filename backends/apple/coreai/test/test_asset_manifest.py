# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import json
import os
import socket
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from executorch.backends.apple.coreai.compiler.asset_manifest import (
    collect_asset_metadata,
)
from executorch.backends.apple.coreai.compiler.preprocess import (
    _deliver,
    AssetPackaging,
)
from executorch.backends.apple.coreai.runtime.test.export_fixtures import (
    _validate_asset_metadata,
)
from executorch.exir._serialize._named_data_store import NamedDataStore


_MODEL_HASH = "0123456789abcdef"
_PAYLOADS = {
    "model.aimodel/\U00010000": b"supplementary",
    "model.aimodel/\ue000": b"private",
    "model.aimodel/\u03a9": b"unicode",
    "model.aimodel/z.bin": b"\x00\xff\x80\x01\x00",
    "model.aimodel/a/empty": b"",
}


def _reference_digest(payloads):
    names = sorted(name.encode("utf-8") for name in payloads)
    frame = b"ExecuTorch.CoreAI.raw-assets.sha256.v1\x00" + struct.pack(
        ">Q", len(names)
    )
    for name in names:
        payload = payloads[name.decode("utf-8")]
        frame += (
            struct.pack(">Q", len(name))
            + name
            + struct.pack(">Q", len(payload))
            + payload
        )
    return hashlib.sha256(frame).hexdigest()


def _write_payloads(root, payloads):
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _source_manifest():
    return {
        "packaging": "inline",
        "hash": _MODEL_HASH,
        "path": f"{_MODEL_HASH}/model.aimodel",
        "function": "main",
        "input_names": ["input_0"],
        "output_names": ["output_0"],
    }


class AssetManifestTest(unittest.TestCase):
    def test_reference_framing_unicode_order_binary_and_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_payloads(root, _PAYLOADS)
            metadata = collect_asset_metadata(root, _source_manifest())
        self.assertEqual(set(metadata), {"version", "files", "bundle_digests"})
        self.assertEqual(metadata["version"], 2)
        self.assertEqual(metadata["files"], {k: len(v) for k, v in _PAYLOADS.items()})
        self.assertEqual(
            list(metadata["files"]), sorted(_PAYLOADS, key=lambda s: s.encode("utf-8"))
        )
        expected = _reference_digest(_PAYLOADS)
        self.assertEqual(
            expected, "dd400f4c64ae3aa90e66a0f82732c2f56882feff204c2189ba1b3a12723047d2"
        )
        self.assertEqual(metadata["bundle_digests"], {"model.aimodel": expected})

    def test_file_sizes_and_same_length_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "model.aimodel" / "data"
            results = []
            for payload in (b"", b"abc", b"xyz", b"longer"):
                _write_payloads(root, {"model.aimodel/data": payload})
                metadata = collect_asset_metadata(root, _source_manifest())
                self.assertEqual(
                    metadata["files"], {"model.aimodel/data": len(payload)}
                )
                self.assertEqual(
                    metadata["bundle_digests"]["model.aimodel"],
                    _reference_digest({"model.aimodel/data": payload}),
                )
                results.append(metadata["bundle_digests"]["model.aimodel"])
            self.assertEqual(len(set(results)), 4)
            self.assertEqual(path.read_bytes(), b"longer")

    def test_inline_and_sidecar_metadata_match_same_finalized_bytes(self):
        for aot in (False, True):
            with self.subTest(aot=aot), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "staging"
                manifest = _source_manifest()
                payloads = _PAYLOADS
                if aot:
                    manifest.pop("path")
                    manifest["archs"] = {
                        arch: f"{_MODEL_HASH}/model.{arch}.aimodelc"
                        for arch in ("h17p", "h15g")
                    }
                    manifest["platform"] = "macOS"
                    payloads = {
                        f"model.{arch}.aimodelc/data": arch.encode()
                        for arch in ("h17p", "h15g")
                    }
                _write_payloads(root, payloads)
                inline = _deliver(
                    root,
                    _MODEL_HASH,
                    (
                        AssetPackaging.AOT_COMPILED_INLINE
                        if aot
                        else AssetPackaging.INLINE
                    ),
                    {k: v for k, v in manifest.items() if k != "packaging"},
                    None,
                )
                sidecar = _deliver(
                    root,
                    _MODEL_HASH,
                    (
                        AssetPackaging.AOT_COMPILED_SIDECAR
                        if aot
                        else AssetPackaging.SIDECAR
                    ),
                    {k: v for k, v in manifest.items() if k != "packaging"},
                    str(Path(tmp) / "sidecars"),
                )
                inline_manifest = json.loads(inline.processed_bytes)
                sidecar_manifest = json.loads(sidecar.processed_bytes)
                _validate_asset_metadata(
                    inline_manifest, Path(tmp), inline.data_store_output
                )
                _validate_asset_metadata(sidecar_manifest, Path(tmp) / "sidecars", None)
                inline_manifest.pop("packaging")
                sidecar_manifest.pop("packaging")
                self.assertEqual(inline_manifest, sidecar_manifest)
                self.assertIsNone(sidecar.data_store_output)

    def test_inline_reads_each_payload_once_and_registers_same_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_payloads(root, _PAYLOADS)
            original_read = Path.read_bytes
            original_add = NamedDataStore.add_named_data
            buffers = {}

            def read(path):
                relative = path.relative_to(root).as_posix()
                self.assertNotIn(relative, buffers)
                buffers[relative] = original_read(path)
                return buffers[relative]

            def add(store, key, payload, **kwargs):
                relative = key.removeprefix(f"coreai/{_MODEL_HASH}/")
                self.assertIs(payload, buffers[relative])
                self.assertEqual(kwargs, {"alignment": 16})
                return original_add(store, key, payload, **kwargs)

            with mock.patch.object(Path, "read_bytes", read), mock.patch.object(
                NamedDataStore, "add_named_data", add
            ):
                result = _deliver(
                    root, _MODEL_HASH, AssetPackaging.INLINE, _source_manifest(), None
                )
            self.assertEqual(buffers, _PAYLOADS)
            self.assertEqual(
                set(result.data_store_output.pte_data),
                {f"coreai/{_MODEL_HASH}/{name}" for name in _PAYLOADS},
            )

    def test_sidecar_hashes_bounded_chunks_in_pending_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "staging"
            _write_payloads(root, _PAYLOADS)
            sidecars = Path(tmp) / "sidecars"
            original_open = Path.open
            reads = []

            @contextlib.contextmanager
            def open_file(path, *args, **kwargs):
                self.assertIn(f".{_MODEL_HASH}.partial", path.parts)
                self.assertFalse((sidecars / _MODEL_HASH).exists())
                with original_open(path, *args, **kwargs) as source:
                    proxy = mock.Mock(wraps=source)
                    reads.append(proxy.read)
                    yield proxy

            with mock.patch.object(Path, "open", open_file), mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("sidecar payload read_bytes"),
            ), mock.patch(
                "executorch.backends.apple.coreai.compiler.asset_manifest._HASH_CHUNK_SIZE",
                3,
            ):
                result = _deliver(
                    root,
                    _MODEL_HASH,
                    AssetPackaging.SIDECAR,
                    {k: v for k, v in _source_manifest().items() if k != "packaging"},
                    str(sidecars),
                )
            self.assertEqual(len(reads), len(_PAYLOADS))
            for read in reads:
                self.assertTrue(read.call_args_list)
                self.assertTrue(
                    all(call == mock.call(3) for call in read.call_args_list)
                )
            self.assertTrue((sidecars / _MODEL_HASH).is_dir())
            self.assertFalse((sidecars / f".{_MODEL_HASH}.partial").exists())
            _validate_asset_metadata(json.loads(result.processed_bytes), sidecars, None)

    def test_aot_changes_only_modified_bundle_identity(self):
        manifest = {
            "packaging": "aot_compiled_inline",
            "hash": _MODEL_HASH,
            "archs": {
                arch: f"{_MODEL_HASH}/model.{arch}.aimodelc"
                for arch in ("h15g", "h17p")
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = {
                f"model.{arch}.aimodelc/data": b"old" for arch in ("h15g", "h17p")
            }
            _write_payloads(root, payloads)
            before = collect_asset_metadata(root, manifest)
            payloads["model.h15g.aimodelc/data"] = b"new"
            _write_payloads(root, payloads)
            after = collect_asset_metadata(root, manifest)
        self.assertEqual(before["files"], after["files"])
        self.assertNotEqual(
            before["bundle_digests"]["model.h15g.aimodelc"],
            after["bundle_digests"]["model.h15g.aimodelc"],
        )
        self.assertEqual(
            before["bundle_digests"]["model.h17p.aimodelc"],
            after["bundle_digests"]["model.h17p.aimodelc"],
        )
        self.assertNotEqual(
            before["bundle_digests"]["model.h15g.aimodelc"],
            before["bundle_digests"]["model.h17p.aimodelc"],
        )

    def test_rejects_unsafe_declared_paths(self):
        for path in (
            "/model.aimodel",
            "../model.aimodel",
            f"{_MODEL_HASH}//model.aimodel",
            f"{_MODEL_HASH}/../model.aimodel",
            f"{_MODEL_HASH}/model.aimodel/",
            f"{_MODEL_HASH}\\model.aimodel",
            f"{_MODEL_HASH}/model.aimodel\0",
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                _write_payloads(Path(tmp), _PAYLOADS)
                manifest = {**_source_manifest(), "path": path}
                with self.assertRaises(ValueError):
                    collect_asset_metadata(Path(tmp), manifest)

    def test_rejects_missing_undeclared_and_zero_file_bundles(self):
        for kind in ("missing", "undeclared", "loose", "empty", "nested_empty"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if kind == "missing":
                    pass
                elif kind in ("empty", "nested_empty"):
                    (root / "model.aimodel").mkdir()
                    if kind == "nested_empty":
                        (root / "model.aimodel" / "empty").mkdir()
                else:
                    _write_payloads(root, _PAYLOADS)
                    if kind == "undeclared":
                        (root / "model.other.aimodelc").mkdir()
                    else:
                        (root / "unexpected").write_bytes(b"loose")
                with self.assertRaises(ValueError):
                    collect_asset_metadata(root, _source_manifest())

    def test_rejects_symlinks_specials_and_unsafe_names_in_both_deliveries(self):
        for kind in (
            "root_link",
            "bundle_link",
            "directory_link",
            "file_link",
            "dangling_link",
            "fifo",
            "socket",
            "backslash",
        ):
            for sidecar in (False, True):
                with self.subTest(
                    kind=kind, sidecar=sidecar
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "staging"
                    bundle = root / "model.aimodel"
                    _write_payloads(root, {"model.aimodel/data": b"data"})
                    if kind == "root_link":
                        root = Path(tmp) / "link"
                        root.symlink_to(Path(tmp) / "staging", target_is_directory=True)
                    elif kind == "bundle_link":
                        bundle.rename(Path(tmp) / "bundle")
                        bundle.symlink_to(
                            Path(tmp) / "bundle", target_is_directory=True
                        )
                    elif kind == "directory_link":
                        (bundle / "link").symlink_to(
                            Path(tmp), target_is_directory=True
                        )
                    elif kind in ("file_link", "dangling_link"):
                        (bundle / "link").symlink_to(
                            bundle / ("data" if kind == "file_link" else "missing")
                        )
                    elif kind == "fifo":
                        os.mkfifo(bundle / "pipe")
                    elif kind == "socket":
                        with socket.socket(socket.AF_UNIX) as sock:
                            sock.bind(str(bundle / "socket"))
                    else:
                        (bundle / "unsafe\\name").write_bytes(b"bad")
                    output = Path(tmp) / "sidecars"
                    with self.assertRaises(ValueError):
                        _deliver(
                            root,
                            _MODEL_HASH,
                            (
                                AssetPackaging.SIDECAR
                                if sidecar
                                else AssetPackaging.INLINE
                            ),
                            {
                                k: v
                                for k, v in _source_manifest().items()
                                if k != "packaging"
                            },
                            str(output) if sidecar else None,
                        )
                    if sidecar and output.exists():
                        self.assertEqual(list(output.iterdir()), [])

    def test_fixture_validation_rejects_incorrect_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_payloads(root, {"model.aimodel/data": b"x"})
            result = _deliver(
                root, _MODEL_HASH, AssetPackaging.INLINE, _source_manifest(), None
            )
            manifest = json.loads(result.processed_bytes)
            _validate_asset_metadata(manifest, None, result.data_store_output)
            for field, value in (
                ("version", 1),
                ("files", {"model.aimodel/data": 2}),
                ("files", {"model.aimodel/data": True}),
                ("bundle_digests", {"model.aimodel": "0" * 64}),
            ):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(RuntimeError, "metadata"):
                        _validate_asset_metadata(
                            {**manifest, field: value}, None, result.data_store_output
                        )

    def test_rejects_invalid_aot_declarations(self):
        for archs in (
            {},
            [],
            {"../h15g": f"{_MODEL_HASH}/model.h15g.aimodelc"},
            {"h15g": f"{_MODEL_HASH}/model.h17p.aimodelc"},
        ):
            with self.subTest(archs=archs), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_payloads(root, {"model.h15g.aimodelc/data": b"x"})
                manifest = {
                    "hash": _MODEL_HASH,
                    "packaging": "aot_compiled_inline",
                    "archs": archs,
                }
                with self.assertRaises(ValueError):
                    collect_asset_metadata(root, manifest)

    def test_rejects_file_size_changed_during_read(self):
        for inline in (False, True):
            with self.subTest(inline=inline), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_payloads(root, {"model.aimodel/data": b"before"})
                original_open = Path.open

                def changed_open(path, *args, original_open=original_open, **kwargs):
                    with original_open(path, "wb") as target:
                        target.write(b"after")
                    return original_open(path, *args, **kwargs)

                with mock.patch.object(Path, "open", changed_open):
                    with self.assertRaisesRegex(ValueError, "size changed"):
                        collect_asset_metadata(
                            root,
                            _source_manifest(),
                            register_payload=mock.Mock() if inline else None,
                        )


if __name__ == "__main__":
    unittest.main()
