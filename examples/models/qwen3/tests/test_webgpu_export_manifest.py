# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import tempfile
import unittest
from pathlib import Path

from executorch.examples.models.qwen3.webgpu_artifact_manifest import (
    check_contract_agreement,
    check_delegation,
    check_export_config,
    check_methods,
    check_no_extra_files,
    check_role_suffixes,
    create_qwen3_manifest,
    expected_export_config,
    load_contract,
    ManifestError,
    MAX_CONTEXT_LEN,
    MAX_INPUT_LEN,
    REQUIRED_METHODS,
    SERIALIZATION_BACKEND,
    TARGET_RUNTIME,
    validate_qwen3_manifest,
    WEBGPU_BACKEND_ID,
)

_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

_ACQUISITION = {
    "checkpoint": {
        "repo": "Qwen/Qwen3-0.6B",
        "revision": _REVISION,
        "filename": "model.safetensors",
        "sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "bytes": 1503300328,
    },
    "tokenizer": {
        "repo": "Qwen/Qwen3-0.6B",
        "revision": _REVISION,
        "filename": "tokenizer.json",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "bytes": 11422654,
    },
}

_ROLES = {
    "pte": Path("qwen3_0_6b.pte"),
    "javascript": Path("runner.js"),
    "wasm": Path("runner.wasm"),
}


def _acquisition(**overrides) -> dict:
    result = copy.deepcopy(_ACQUISITION)
    for dotted, value in overrides.items():
        kind, _, field = dotted.partition("__")
        result[kind][field] = value
    return result


def _populate(root: Path) -> None:
    for name, data in {
        "qwen3_0_6b.pte": b"pte-bytes",
        "runner.js": b"js-bytes",
        "runner.wasm": b"wasm-bytes",
    }.items():
        (root / name).write_bytes(data)


def _build(root: Path, acquisition=None) -> dict:
    return create_qwen3_manifest(root, dict(_ROLES), (), acquisition or _ACQUISITION)


class ExportContractTest(unittest.TestCase):
    def _config(self, **overrides) -> dict:
        config = expected_export_config()
        config.update(overrides)
        return config

    def test_accepts_the_declared_webgpu_runtime_target(self) -> None:
        check_export_config(self._config())
        self.assertEqual(expected_export_config()["target_runtime"], TARGET_RUNTIME)

    def test_rejects_a_wrong_runtime_target(self) -> None:
        for wrong in ("vulkan", "xnnpack", "", None):
            with self.subTest(target_runtime=wrong):
                with self.assertRaisesRegex(ManifestError, "target_runtime"):
                    check_export_config(self._config(target_runtime=wrong))

    def test_rejects_a_wrong_serialization_backend(self) -> None:
        with self.assertRaisesRegex(ManifestError, "serialization_backend"):
            check_export_config(self._config(serialization_backend="xnnpack"))

    def test_vulkan_serialization_alone_is_not_a_webgpu_claim(self) -> None:
        config = self._config()
        del config["target_runtime"]
        self.assertEqual(config["serialization_backend"], SERIALIZATION_BACKEND)
        with self.assertRaisesRegex(ManifestError, "target_runtime"):
            check_export_config(config)

    def test_rejects_a_wrong_capacity(self) -> None:
        for field, value in (
            ("max_input_len", MAX_INPUT_LEN + 1),
            ("max_context_len", MAX_CONTEXT_LEN - 1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ManifestError, field):
                    check_export_config(self._config(**{field: value}))


class CheckedContractTest(unittest.TestCase):
    def test_the_checked_contract_is_self_consistent(self) -> None:
        contract = load_contract()
        self.assertEqual(contract["model"], "qwen3_0_6b")
        self.assertEqual(contract["export"]["target_runtime"], TARGET_RUNTIME)
        self.assertEqual(
            contract["export"]["serialization_backend"], SERIALIZATION_BACKEND
        )
        self.assertEqual(contract["export"]["max_input_len"], MAX_INPUT_LEN)
        self.assertEqual(contract["export"]["max_context_len"], MAX_CONTEXT_LEN)
        self.assertEqual(contract["acquisition"]["checkpoint"]["revision"], _REVISION)

    def test_a_built_manifest_must_agree_with_the_contract(self) -> None:
        contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root, contract["acquisition"])
            validate_qwen3_manifest(root, manifest, contract["acquisition"], contract)

    def test_contract_agreement_rejects_a_swapped_revision(self) -> None:
        contract = load_contract()
        manifest = {
            "model": "qwen3_0_6b",
            "acquisition": _acquisition(checkpoint__revision="0" * 40),
            "export": expected_export_config(),
        }
        with self.assertRaisesRegex(ManifestError, "revision"):
            check_contract_agreement(manifest, contract)

    def test_contract_agreement_rejects_a_downgraded_runtime_target(self) -> None:
        contract = load_contract()
        export = expected_export_config()
        export["target_runtime"] = "vulkan"
        manifest = {
            "model": "qwen3_0_6b",
            "acquisition": copy.deepcopy(contract["acquisition"]),
            "export": export,
        }
        with self.assertRaisesRegex(ManifestError, "target_runtime"):
            check_contract_agreement(manifest, contract)


class DelegationTest(unittest.TestCase):
    def test_accepts_a_fully_delegated_webgpu_graph(self) -> None:
        check_delegation([WEBGPU_BACKEND_ID], [])

    def test_rejects_a_portable_operator(self) -> None:
        for ops in (["aten::add.out"], ["aten::mm.out", "aten::view_copy.out"]):
            with self.subTest(ops=ops):
                with self.assertRaisesRegex(ManifestError, "portable"):
                    check_delegation([WEBGPU_BACKEND_ID], ops)

    def test_rejects_a_foreign_or_mixed_delegate_census(self) -> None:
        for ids in (["XnnpackBackend"], [WEBGPU_BACKEND_ID, "XnnpackBackend"], []):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ManifestError, "backend"):
                    check_delegation(ids, [])

    def test_method_census_must_match_exactly(self) -> None:
        check_methods(list(REQUIRED_METHODS))
        for observed in ([*REQUIRED_METHODS, "extra"], [], ["not_forward"]):
            with self.subTest(observed=observed):
                with self.assertRaisesRegex(ManifestError, "method"):
                    check_methods(observed)


class AcquisitionPinTest(unittest.TestCase):
    def _expect(self, pattern, **overrides) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            with self.assertRaisesRegex(ManifestError, pattern):
                _build(root, _acquisition(**overrides))

    def test_rejects_a_wrong_checkpoint_digest_shape(self) -> None:
        self._expect("sha256", checkpoint__sha256="not-a-digest")
        self._expect("sha256", checkpoint__sha256="F" * 64)

    def test_rejects_a_wrong_checkpoint_size(self) -> None:
        for bad in (0, -1, "1503300328", None):
            with self.subTest(bytes=bad):
                self._expect("bytes", checkpoint__bytes=bad)

    def test_rejects_a_non_commit_revision(self) -> None:
        for bad in ("main", "v1.0", "c1899de", ""):
            with self.subTest(revision=bad):
                self._expect("revision", checkpoint__revision=bad)

    def test_rejects_a_missing_acquisition_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            for kind in ("checkpoint", "tokenizer"):
                acquisition = copy.deepcopy(_ACQUISITION)
                del acquisition[kind]
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(ManifestError, kind):
                        _build(root, acquisition)

    def test_rejects_a_tampered_pin_at_validation(self) -> None:
        for field, value in (
            ("sha256", "c" * 64),
            ("bytes", 1),
            ("revision", "d" * 40),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _populate(root)
                    manifest = _build(root)
                    manifest["acquisition"]["checkpoint"][field] = value
                    with self.assertRaisesRegex(ManifestError, field):
                        validate_qwen3_manifest(root, manifest, _ACQUISITION)


class ArtifactRoleTest(unittest.TestCase):
    def test_rejects_role_confusion_between_pte_and_wasm(self) -> None:
        artifacts = [
            {"role": "pte", "path": "runner.wasm"},
            {"role": "wasm", "path": "runner.wasm"},
        ]
        with self.assertRaisesRegex(ManifestError, "role confusion"):
            check_role_suffixes(artifacts)

    def test_rejects_javascript_role_pointing_at_the_pte(self) -> None:
        with self.assertRaisesRegex(ManifestError, "role confusion"):
            check_role_suffixes([{"role": "javascript", "path": "qwen3_0_6b.pte"}])

    def test_accepts_correctly_typed_roles(self) -> None:
        check_role_suffixes(
            [
                {"role": "pte", "path": "qwen3_0_6b.pte"},
                {"role": "javascript", "path": "runner.js"},
                {"role": "wasm", "path": "runner.wasm"},
                {"role": "ptd", "path": "weights.ptd"},
            ]
        )

    def test_create_rejects_a_role_confused_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            roles = dict(_ROLES)
            roles["pte"] = Path("runner.wasm")
            with self.assertRaisesRegex(ManifestError, "role confusion"):
                create_qwen3_manifest(root, roles, (), _ACQUISITION)


class ManifestRoundTripTest(unittest.TestCase):
    def test_round_trip_validates_and_records_no_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root)
            validate_qwen3_manifest(root, manifest)
            self.assertEqual(manifest["model"], "qwen3_0_6b")
            self.assertEqual(manifest["ptd_order"], [])
            self.assertNotIn(str(root), json.dumps(manifest))

    def test_rejects_a_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root)
            (root / "runner.wasm").unlink()
            with self.assertRaises(ManifestError):
                validate_qwen3_manifest(root, manifest)

    def test_rejects_a_missing_required_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root)
            manifest["artifacts"] = [
                a for a in manifest["artifacts"] if a["role"] != "wasm"
            ]
            (root / "runner.wasm").unlink()
            with self.assertRaisesRegex(ManifestError, "wasm"):
                validate_qwen3_manifest(root, manifest)

    def test_rejects_an_extra_untracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root)
            (root / "stowaway.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(ManifestError, "stowaway.bin"):
                validate_qwen3_manifest(root, manifest)

    def test_rejects_a_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root)
            (root / "qwen3_0_6b.pte").write_bytes(b"tampered!")
            with self.assertRaises(ManifestError):
                validate_qwen3_manifest(root, manifest)

    def test_rejects_a_wrong_recorded_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            manifest = _build(root)
            for artifact in manifest["artifacts"]:
                if artifact["role"] == "pte":
                    artifact["bytes"] += 1
            with self.assertRaises(ManifestError):
                validate_qwen3_manifest(root, manifest)

    def test_rejects_a_symlinked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _populate(root)
            (root / "aliased.pte").symlink_to(root / "qwen3_0_6b.pte")
            roles = dict(_ROLES)
            roles["pte"] = Path("aliased.pte")
            with self.assertRaises(ManifestError):
                create_qwen3_manifest(root, roles, (), _ACQUISITION)


class ExtraFileScanTest(unittest.TestCase):
    def test_reports_the_offending_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kept.bin").write_bytes(b"kept")
            nested = root / "nested"
            nested.mkdir()
            (nested / "sneaky.bin").write_bytes(b"sneaky")
            check_no_extra_files(root, {"kept.bin", "nested/sneaky.bin"})
            with self.assertRaisesRegex(ManifestError, "nested/sneaky.bin"):
                check_no_extra_files(root, {"kept.bin"})


if __name__ == "__main__":
    unittest.main()
