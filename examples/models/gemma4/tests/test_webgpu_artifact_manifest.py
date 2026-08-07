# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from executorch.backends.webgpu.scripts import (
    webgpu_artifact_manifest as backend_manifest,
)
from executorch.examples.models.gemma4 import (
    target_prefill_contract,
    webgpu_artifact_manifest as gemma4_manifest,
)
from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    ARCHITECTURE_FINGERPRINT,
    CHECKPOINT_ACQUISITION,
    create_plain_manifest,
    EXPORT_CONTRACT,
    validate_plain_manifest,
)


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
        self.assertEqual(self.manifest["ptd_order"], [path.name for path in self.ptds])
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


def _set_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _test_source_manifest() -> dict[str, object]:
    logical_path = "examples/models/gemma4/webgpu_artifact_manifest.py"
    identity = {"bytes": 7, "sha256": "3" * 64}
    files = [
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
    ]
    return {
        "checkouts": {
            "fbsource": {"clean": True, "head": "1" * 40},
            "oss": {"clean": True, "head": "2" * 40},
        },
        "file_set_sha256": _set_digest([logical_path]),
        "files": files,
        "schema_version": 1,
    }


def _test_wgsl_manifest() -> dict[str, object]:
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


def _sealed_source_receipt() -> dict[str, object]:
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


_SEALED_SOURCE_RECEIPT = _sealed_source_receipt()
_BINARY_SUFFIXES = frozenset({".bin", ".gguf", ".pte", ".ptd", ".safetensors"})
_ARTIFACT_KEYS = ["bytes", "path", "role", "sha256"]


def _write_source_receipt(path: Path) -> None:
    path.write_text(json.dumps(_sealed_source_receipt()), encoding="utf-8")


def _identity(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _target_prefill_receipt(
    runtime_source_path: Path,
) -> dict[str, object]:
    runtime_source = json.loads(runtime_source_path.read_text(encoding="utf-8"))
    producer_path = target_prefill_contract.reviewed_producer_source_path()
    contexts: dict[str, object] = {}
    for context in target_prefill_contract.TARGET_PREFILL_CONTEXTS:
        start, length = target_prefill_contract.final_chunk_range(context)
        tensor = {
            "byte_order": "little",
            "dtype": "float32",
            "layout": "row_major_contiguous",
            "sha256": "a" * 64,
        }
        arm_config = {
            "dtype": "float32",
            "enable_dynamic_shape": True,
            "group_size": 128,
            "max_seq_len": 8960,
            "text_quantize": "8da4w+emb4",
            "use_kv_cache": True,
            "variant": "e2b",
        }
        contexts[str(context)] = {
            "arm_configs": {
                "custom_sdpa_fused": {**arm_config, "use_custom_sdpa": True},
                "manual_unfused": {**arm_config, "use_custom_sdpa": False},
            },
            "cache_reset_counts": {
                "custom_sdpa_fused": 15,
                "manual_unfused": 15,
            },
            "chunk_size": 512,
            "context": context,
            "final_chunk_length": length,
            "final_chunk_start": start,
            "layer0_manual_unfused_vs_custom_sdpa_fused": {
                "agreement": {
                    "atol": 1e-4,
                    "max_abs": 1e-5,
                    "passed": True,
                    "rel_rms": 1e-6,
                    "rtol": 1e-3,
                },
                "custom_sdpa_fused": {
                    **tensor,
                    "shape": [1, length, 8, 256],
                },
                "manual_unfused": {
                    **tensor,
                    "sha256": "b" * 64,
                    "shape": [1, length, 8, 256],
                },
            },
            "logits_post_softcap": {
                **tensor,
                "sha256": "c" * 64,
                "shape": [1, 1, 262144],
            },
            "logits_pre_softcap": {
                **tensor,
                "sha256": "d" * 64,
                "shape": [1, 1, 262144],
            },
            "prefill_token_post_softcap": 17,
            "prefill_token_raw": 17,
            "prompt_plan_sha256": target_prefill_contract.prompt_plan_sha256(context),
        }
    return {
        "authority": "target_only_eager",
        "checkpoint_acquisition": gemma4_manifest.CHECKPOINT_ACQUISITION,
        "contexts": contexts,
        "envelope_kind": "target_prefill_v2",
        "producer": {
            "fbsource_commit": runtime_source["fbsource_commit"],
            "runtime_source_receipt": _identity(runtime_source_path),
            "source_path": producer_path.name,
            "source_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
        },
        "run": {
            "command": ["generate_target_prefill_oracle", "--contexts", "all"],
            "finished_at_utc": "2026-08-07T12:00:01Z",
            "host": "test-host",
            "started_at_utc": "2026-08-07T12:00:00Z",
        },
        "schema_version": 2,
    }


def _generated_mtp_evidence() -> dict[str, object]:
    mutation_order: list[dict[str, object]] = [
        {
            "logicalTarget": "seed_feature",
            "role": "nextFeatureSeed",
            "shape": [1, 1, 1, 1536],
            "logicalLayout": "BSHD",
            "logicalDimOrder": [0, 1, 2, 3],
            "vulkanSourceStorage": "BUFFER",
            "vulkanDestinationStorage": "TEXTURE_3D",
        }
    ]
    for layer in range(15):
        head_dim = 512 if layer in {4, 9, 14} else 256
        for cache_kind in ("k_cache", "v_cache"):
            mutation_order.append(
                {
                    "logicalTarget": (
                        f"self_decoder.layers.{layer}.self_attn.kv_cache."
                        f"{cache_kind}"
                    ),
                    "role": "targetKvCache",
                    "layer": layer,
                    "cacheKind": cache_kind,
                    "shape": [1, 8960, 1, head_dim],
                    "logicalLayout": "BSHD",
                    "logicalDimOrder": [0, 1, 2, 3],
                    "vulkanSourceStorage": "BUFFER",
                    "vulkanDestinationStorage": "BUFFER",
                }
            )
    token_record: dict[str, object] = {
        "max": 262143,
        "min": 0,
        "numel": 262144,
        "permutationExact": True,
        "rawShape": [262144],
        "sha256": "5" * 64,
        "shape": [2048, 128],
        "uniqueCount": 262144,
    }
    token_ordering = {
        **token_record,
        "loaded": copy.deepcopy(token_record),
        "raw": copy.deepcopy(token_record),
        "rawLoadedByteExact": True,
        "rawSha256": token_record["sha256"],
    }
    donor_sequence = [2, 16, 511, 512, 513, 514, 1024, 8960, 2]
    cases = [
        {
            "caseIndex": index,
            "donorLength": donor_length,
            "greedyTokenExact": True,
            "inputSha256": ["6" * 64],
            "outputs": [
                {
                    "actualSha256": "7" * 64,
                    "bitExact": True,
                    "close": True,
                    "maxAbsError": 0.0,
                    "name": name,
                    "referenceSha256": "7" * 64,
                    "shape": [1, 1],
                }
                for name in ("logits", "last_hidden_state")
            ],
            "topk": {
                "allFinite": True,
                "boundaryGap": 1.0,
                "indicesSha256": "8" * 64,
                "stableReferenceExact": True,
                "top32PairwiseDistinct": True,
                "top33IndicesSha256": "9" * 64,
                "top33ValuesSha256": "a" * 64,
                "valuesSha256": "b" * 64,
            },
        }
        for index, donor_length in enumerate(donor_sequence)
    ]
    return {
        "assistant_checkpoint": gemma4_manifest.ASSISTANT_CHECKPOINT_ACQUISITION,
        "k2_abi": {
            "bufferMutationCount": 31,
            "donorViewOrder": [
                {
                    "role": "fullK",
                    "layer": 14,
                    "cacheKind": "k_cache",
                    "layout": "BHKD",
                },
                {
                    "role": "fullV",
                    "layer": 14,
                    "cacheKind": "v_cache",
                    "layout": "BHKD",
                },
                {
                    "role": "slidingK",
                    "layer": 13,
                    "cacheKind": "k_cache",
                    "layout": "BHKD",
                },
                {
                    "role": "slidingV",
                    "layer": 13,
                    "cacheKind": "v_cache",
                    "layout": "BHKD",
                },
            ],
            "inputOrder": ["input_ids", "input_pos", "is_round", "donor_length"],
            "mutationOrder": mutation_order,
            "operatorCounts": {
                "aten.argmax.default": 3,
                "aten.scatter.src": 2,
                "aten.topk.default": 2,
                "llama.custom_sdpa.default": 43,
                "llama.update_cache.default": 31,
            },
            "outputOrder": [
                "candidates",
                "target_greedy",
                "output_matches",
                "output_bonus",
                "state_probe",
            ],
            "seedMutationCount": 1,
            "stateAlias": {
                "logicalSource": "nextFeature[1,1,1536]",
                "physicalDestination": "seed_feature[1,1,1,1536]",
                "mutation": "llama.update_cache.default",
            },
        },
        "lowering": {
            "delegate_count": 1,
            "edge": gemma4_manifest.MTP_EDGE_CENSUS,
            "portable_operator_count": 0,
        },
        "qat_selection": {
            "cases": cases,
            "donorSequence": donor_sequence,
            "eagerEquivalence": {"allClose": True, "atol": 1e-4, "rtol": 1e-3},
            "selectionContract": {
                "centroidTopK": 32,
                "numCentroids": 2048,
                "selectedTokenCount": 4096,
                "tokensPerCentroid": 128,
            },
            "tokenOrdering": token_ordering,
        },
        "target_checkpoint": gemma4_manifest.CHECKPOINT_ACQUISITION,
    }


def _artifact_entry(manifest: dict[str, object], path: str) -> dict[str, object]:
    artifacts = manifest["artifacts"]
    matches = [entry for entry in artifacts if entry["path"] == path]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one artifact for {path}")
    return matches[0]


def _package_root() -> Path:
    return Path(gemma4_manifest.__file__).resolve().parent


def _internal_patterns() -> list[re.Pattern[str]]:
    return [
        re.compile(pattern)
        for pattern in (
            r"manifold",  # oss-closure-fixture
            r"internalfb",  # oss-closure-fixture
            r"/data/users/",  # oss-closure-fixture
            r"fburl",  # oss-closure-fixture
            r"/home/",  # oss-closure-fixture
            r"\bfb/",
        )
    ]


class SourceClosureManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.fbsource_root = self.root / "fbsource"
        self.oss_root = self.root / "oss"
        self.fbsource_root.mkdir()
        self.oss_root.mkdir()
        self.owned_paths = [
            "examples/models/gemma4/webgpu_artifact_manifest.py",
            "backends/webgpu/runtime/WebGPUShaderRegistry.cpp",
        ]
        for index, logical_path in enumerate(self.owned_paths):
            contents = f"source-{index}".encode("utf-8")
            for path in (
                self.fbsource_root / "fbcode/executorch" / logical_path,
                self.fbsource_root / "xplat/executorch" / logical_path,
                self.oss_root / logical_path,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)

    def _create_source_manifest(self) -> dict[str, object]:
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
            return_value=sorted(self.owned_paths),
        ):
            return gemma4_manifest.create_source_manifest(
                self.fbsource_root,
                self.oss_root,
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
        gemma4_manifest.validate_source_manifest(
            json.loads(output.read_text(encoding="utf-8"))
        )

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
        gemma4_manifest.validate_wgsl_manifest(
            json.loads(output.read_text(encoding="utf-8"))
        )

    def test_create_source_receipt_cli_round_trip(self) -> None:
        output = self.root / "receipt.json"
        receipt = _sealed_source_receipt()
        with mock.patch.object(
            gemma4_manifest,
            "create_source_closure_receipt",
            return_value=receipt,
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

    def _copy_backend_root(self) -> Path:
        source = Path(backend_manifest.__file__).resolve().parents[1]
        backend_root = (
            self.root / "xplat" / "executorch" / "backends" / "webgpu"
        )
        shutil.copytree(source, backend_root)
        return backend_root

    def test_source_manifest_does_not_accept_a_caller_selected_subset(self) -> None:
        parameters = inspect.signature(
            gemma4_manifest.create_source_manifest
        ).parameters
        self.assertNotIn("owned_paths", parameters)

    def test_owned_union_is_derived_from_every_reviewed_diff(self) -> None:
        production_summaries = (
            "[ExecuTorch][WebGPU] Add shared model runtime prerequisites",
            "[ExecuTorch][Vulkan] Support scoped Gemma symbolic partitioning",
            "[ExecuTorch][WebGPU] Add Gemma 4 plain runtime and guarded routes",
            "[ExecuTorch][WebGPU] Add Gemma 4 plain export and artifact contract",
            "[ExecuTorch][WebGPU] Add plain Gemma 4 source-closure tests",
            "[ExecuTorch][WebGPU] Add Gemma 4 MTP operator and route support",
            "[ExecuTorch][WebGPU] Add Gemma 4 MTP export path",
            "[ExecuTorch][WebGPU] Add Gemma 4 speculative decode runtime",
            "[ExecuTorch][WebGPU] Add Gemma 4 MTP and speculative-decode source-closure tests",
        )
        self.assertEqual(
            gemma4_manifest._GEMMA_PRODUCTION_DIFF_SUMMARIES,
            production_summaries,
        )
        summaries = tuple(reversed(production_summaries))

        def source_control(argv: list[str], _label: str) -> str:
            if "log" in argv:
                revision = argv[argv.index("-r") + 1]
                offset = 0 if revision == "." else int(revision.removeprefix(".~"))
                return f"{offset + 1:040x}\n{summaries[offset]}\n"
            node = argv[argv.index("--change") + 1]
            offset = int(node, 16) - 1
            path = f"runtime/owned_{offset}.cpp"
            return f"xplat/executorch/{path}\nfbcode/executorch/{path}\n"

        with mock.patch.object(
            gemma4_manifest, "_run_source_control", side_effect=source_control
        ):
            paths = gemma4_manifest._derive_owned_paths(self.fbsource_root)
        self.assertEqual(
            paths,
            [f"runtime/owned_{index}.cpp" for index in range(9)],
        )

    def test_source_manifest_derives_heads_and_binds_every_copy(self) -> None:
        manifest = self._create_source_manifest()
        gemma4_manifest.validate_source_manifest(manifest)
        self.assertEqual(
            manifest["checkouts"],
            {
                "fbsource": {"clean": True, "head": "1" * 40},
                "oss": {"clean": True, "head": "2" * 40},
            },
        )
        files = manifest["files"]
        self.assertEqual([entry["path"] for entry in files], sorted(self.owned_paths))
        for entry in files:
            identities = {
                (copy["bytes"], copy["sha256"]) for copy in entry["copies"].values()
            }
            self.assertEqual(len(identities), 1)

    def test_source_manifest_rejects_mirror_or_oss_drift(self) -> None:
        (self.oss_root / self.owned_paths[0]).write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "mirror/OSS identity mismatch"):
            self._create_source_manifest()

    def test_source_manifest_rejects_ancestor_symlink_traversal(self) -> None:
        parent = self.fbsource_root / "xplat/executorch/examples"
        saved = self.root / "saved-examples"
        parent.rename(saved)
        parent.symlink_to(saved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink traversal"):
            self._create_source_manifest()

    def test_source_manifest_fails_closed_without_clean_oss_identity(self) -> None:
        def snapshot(_root: Path, kind: str) -> dict[str, object]:
            if kind == "oss":
                raise ValueError("oss checkout is not clean")
            return {"clean": True, "head": "1" * 40}

        with mock.patch.object(
            gemma4_manifest, "_checkout_snapshot", side_effect=snapshot
        ):
            with self.assertRaisesRegex(ValueError, "oss checkout is not clean"):
                gemma4_manifest.create_source_manifest(
                    self.fbsource_root, self.oss_root
                )

    def test_wgsl_manifest_uses_generator_complete_dynamic_closure(self) -> None:
        backend_root = self._copy_backend_root()
        with mock.patch.object(
            gemma4_manifest,
            "_checkout_snapshot",
            return_value={"clean": True, "head": "1" * 40},
        ):
            manifest = gemma4_manifest.create_wgsl_manifest(backend_root)
        gemma4_manifest.validate_wgsl_manifest(manifest)
        self.assertEqual(manifest["fbsource_commit"], "1" * 40)
        roles = [entry["role"] for entry in manifest["files"]]
        self.assertEqual(roles.count("generator"), 1)
        self.assertEqual(roles.count("global_registry"), 1)
        self.assertGreater(roles.count("wgsl"), 0)
        self.assertGreater(roles.count("generated_header"), roles.count("wgsl"))
        self.assertEqual(manifest["orphans"], [])

    def test_wgsl_manifest_rejects_generator_output_byte_drift(self) -> None:
        backend_root = self._copy_backend_root()
        generator = gemma4_manifest._load_wgsl_generator(backend_root)
        outputs, orphans = generator.collect_outputs()
        stale_outputs = dict(outputs)
        path = next(iter(stale_outputs))
        generated = stale_outputs[path]
        stale_outputs[path] = bytes([generated[0] ^ 1]) + generated[1:]
        with mock.patch.object(
            gemma4_manifest,
            "_checkout_snapshot",
            return_value={"clean": True, "head": "1" * 40},
        ), mock.patch.object(
            gemma4_manifest, "_load_wgsl_generator", return_value=generator
        ), mock.patch.object(
            generator, "collect_outputs", return_value=(stale_outputs, orphans)
        ):
            with self.assertRaisesRegex(ValueError, "generated output is stale"):
                gemma4_manifest.create_wgsl_manifest(backend_root)

    def test_source_receipt_is_derived_from_live_roots_not_manifest_json(self) -> None:
        parameters = inspect.signature(
            gemma4_manifest.create_source_closure_receipt
        ).parameters
        self.assertNotIn("source_manifest_path", parameters)
        self.assertNotIn("wgsl_manifest_path", parameters)
        with mock.patch.object(
            gemma4_manifest,
            "create_source_manifest",
            return_value=_test_source_manifest(),
        ) as create_source, mock.patch.object(
            gemma4_manifest,
            "create_wgsl_manifest",
            return_value=_test_wgsl_manifest(),
        ) as create_wgsl:
            receipt = gemma4_manifest.create_source_closure_receipt(
                self.fbsource_root, self.oss_root, self.fbsource_root
            )
        create_source.assert_called_once_with(self.fbsource_root, self.oss_root)
        create_wgsl.assert_called_once_with(self.fbsource_root)
        self.assertEqual(receipt["fbsource_commit"], "1" * 40)

    def test_source_receipt_rejects_wgsl_from_another_checkout(self) -> None:
        wgsl = _test_wgsl_manifest()
        wgsl["fbsource_commit"] = "9" * 40
        with mock.patch.object(
            gemma4_manifest,
            "create_source_manifest",
            return_value=_test_source_manifest(),
        ), mock.patch.object(
            gemma4_manifest, "create_wgsl_manifest", return_value=wgsl
        ):
            with self.assertRaisesRegex(ValueError, "different fbsource heads"):
                gemma4_manifest.create_source_closure_receipt(
                    self.fbsource_root, self.oss_root, self.fbsource_root
                )


class MTPArtifactManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        # Deferred: the plain cases above must load without the D8 manifest extension.
        from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
            create_mtp_manifest,
            validate_mtp_manifest,
        )

        self.create_mtp_manifest = create_mtp_manifest
        self.validate_mtp_manifest = validate_mtp_manifest
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.pte = self.root / "k2_round.pte"
        self.ptds = [self.root / f"constants_{index}.ptd" for index in range(3)]
        # Written only by the source-verified cases: staging must stay exact.
        self.source_receipt = self.root / "source_receipt.json"
        self.pte.write_bytes(b"k2-round-pte")
        for index, path in enumerate(self.ptds):
            path.write_bytes(f"mtp-ptd-{index}".encode("utf-8"))
        self.manifest = self.create_mtp_manifest(
            self.root,
            {"pte": Path(self.pte.name)},
            [Path(path.name) for path in self.ptds],
        )
        self.manifest["provenance"] = copy.deepcopy(
            gemma4_manifest.MTP_ACCEPTED_PROVENANCE
        )

    def _pending_manifest(self) -> dict[str, object]:
        manifest = self.create_mtp_manifest(
            self.root,
            {"pte": Path(self.pte.name)},
            [Path(path.name) for path in self.ptds],
        )
        manifest["evidence"] = _generated_mtp_evidence()
        return manifest

    def _verified_manifest(self) -> dict[str, object]:
        _write_source_receipt(self.source_receipt)
        manifest = self.create_mtp_manifest(
            self.root,
            {
                "pte": Path(self.pte.name),
                "source": Path(self.source_receipt.name),
            },
            [Path(path.name) for path in self.ptds],
        )
        manifest["evidence"] = _generated_mtp_evidence()
        return manifest

    def test_single_k2_pte_and_three_ptds_round_trip(self) -> None:
        self.validate_mtp_manifest(self.root, self.manifest)
        artifacts = self.manifest["artifacts"]
        self.assertEqual(
            sorted(entry["role"] for entry in artifacts),
            ["ptd", "ptd", "ptd", "pte"],
        )
        for entry in artifacts:
            self.assertEqual(sorted(entry), _ARTIFACT_KEYS)
            self.assertRegex(str(entry["sha256"]), "^[0-9a-f]{64}$")
        self.assertEqual(
            _artifact_entry(self.manifest, "k2_round.pte")["sha256"],
            hashlib.sha256(b"k2-round-pte").hexdigest(),
        )
        self.assertEqual(_artifact_entry(self.manifest, "k2_round.pte")["bytes"], 12)

    def test_create_mtp_cli_requires_evidence_and_emits_a_valid_manifest(
        self,
    ) -> None:
        arguments = [
            "create-mtp",
            "--root",
            str(self.root),
            "--role",
            f"pte={self.pte.name}",
            *(argument for path in self.ptds for argument in ("--ptd", path.name)),
        ]
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            gemma4_manifest.main([*arguments, "--output", str(self.root / "unused")])

        with tempfile.TemporaryDirectory() as metadata_directory:
            metadata_root = Path(metadata_directory)
            evidence_path = metadata_root / "evidence.json"
            output_path = metadata_root / "mtp.json"
            evidence = _generated_mtp_evidence()
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertEqual(
                gemma4_manifest.main(
                    [
                        *arguments,
                        "--evidence",
                        str(evidence_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            manifest = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["evidence"], evidence)
            self.validate_mtp_manifest(self.root, manifest)

    def test_mtp_creation_rejects_split_model_roles(self) -> None:
        for role in ("assistant", "speculative"):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, "one K=2 PTE role"):
                    self.create_mtp_manifest(
                        self.root,
                        {"pte": Path(self.pte.name), role: Path(self.pte.name)},
                        [Path(path.name) for path in self.ptds],
                    )

    def test_mtp_creation_rejects_duplicate_normalized_artifact_paths(self) -> None:
        ptd_paths = [Path(path.name) for path in self.ptds]
        for source_alias in (self.pte, *self.ptds):
            with self.subTest(source_alias=source_alias.name):
                with self.assertRaisesRegex(
                    ValueError, "duplicate normalized artifact path"
                ):
                    self.create_mtp_manifest(
                        self.root,
                        {
                            "pte": Path(self.pte.name),
                            "source": Path(source_alias.name),
                        },
                        ptd_paths,
                    )

    def test_mtp_validation_rejects_duplicate_normalized_artifact_paths(
        self,
    ) -> None:
        for source_alias in (self.pte, *self.ptds):
            with self.subTest(source_alias=source_alias.name):
                duplicated = self._verified_manifest()
                alias_entry = _artifact_entry(duplicated, source_alias.name)
                source_entry = _artifact_entry(duplicated, self.source_receipt.name)
                source_entry.update(
                    {
                        "bytes": alias_entry["bytes"],
                        "path": alias_entry["path"],
                        "sha256": alias_entry["sha256"],
                    }
                )
                with self.assertRaisesRegex(
                    ValueError, "duplicate normalized artifact path"
                ):
                    self.validate_mtp_manifest(self.root, duplicated)

    def test_mtp_ptd_order_is_the_artifact_order(self) -> None:
        artifacts = self.manifest["artifacts"]
        self.assertEqual(
            self.manifest["ptd_order"],
            [entry["path"] for entry in artifacts if entry["role"] == "ptd"],
        )
        self.assertEqual(self.manifest["ptd_order"], [path.name for path in self.ptds])

        reordered = copy.deepcopy(self.manifest)
        order = reordered["ptd_order"]
        order[0], order[1] = order[1], order[0]
        with self.assertRaisesRegex(ValueError, "PTD order does not match"):
            self.validate_mtp_manifest(self.root, reordered)

    def test_mtp_rejects_symlinked_artifact(self) -> None:
        self.pte.unlink()
        self.pte.symlink_to(self.ptds[0].name)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.validate_mtp_manifest(self.root, self.manifest)

    def test_mtp_rejects_absolute_artifact_path(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        _artifact_entry(mutated, "k2_round.pte")["path"] = str(self.pte)
        with self.assertRaisesRegex(ValueError, "absolute artifact path"):
            self.validate_mtp_manifest(self.root, mutated)

    def test_mtp_rejects_non_canonical_artifact_path(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        _artifact_entry(mutated, "k2_round.pte")["path"] = "./k2_round.pte"
        with self.assertRaisesRegex(ValueError, "non-canonical artifact path"):
            self.validate_mtp_manifest(self.root, mutated)

    def test_mtp_rejects_wrong_size(self) -> None:
        self.pte.write_bytes(b"k2-round-pte-grew")
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            self.validate_mtp_manifest(self.root, self.manifest)

    def test_mtp_rejects_wrong_hash(self) -> None:
        self.pte.write_bytes(b"k2-round-ptX")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate_mtp_manifest(self.root, self.manifest)

    def test_mtp_rejects_missing_artifact(self) -> None:
        self.ptds[2].unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate_mtp_manifest(self.root, self.manifest)

    def test_mtp_rejects_extra_artifact(self) -> None:
        extra = self.root / "constants_3.ptd"
        extra.write_bytes(b"mtp-ptd-3")
        mutated = copy.deepcopy(self.manifest)
        mutated["artifacts"].append(
            {
                "bytes": extra.stat().st_size,
                "path": extra.name,
                "role": "ptd",
                "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
            }
        )
        with self.assertRaisesRegex(ValueError, "PTD order does not match"):
            self.validate_mtp_manifest(self.root, mutated)

    def test_mtp_rejects_stale_staging(self) -> None:
        (self.root / "stale_constants.ptd").write_bytes(b"stale")
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            self.validate_mtp_manifest(self.root, self.manifest)

    def test_pending_source_manifest_is_a_valid_state(self) -> None:
        pending = self._pending_manifest()
        self.assertEqual(
            pending["provenance"], gemma4_manifest.MTP_PENDING_SOURCE_PROVENANCE
        )
        self.validate_mtp_manifest(self.root, pending)

    def test_pending_source_manifest_is_not_source_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "source closure is still pending"):
            gemma4_manifest._validate_source_complete_mtp_manifest(
                self.root, self._pending_manifest()
            )

    def test_sealed_source_receipt_stamps_source_verified(self) -> None:
        verified = self._verified_manifest()
        self.assertEqual(
            verified["provenance"], gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE
        )
        self.validate_mtp_manifest(self.root, verified)
        gemma4_manifest._validate_source_complete_mtp_manifest(self.root, verified)

    def test_tampered_source_receipt_bytes_are_rejected(self) -> None:
        verified = self._verified_manifest()
        self.source_receipt.write_text(
            json.dumps({**_SEALED_SOURCE_RECEIPT, "oss_commit": "3" * 40}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate_mtp_manifest(self.root, verified)

    def test_unsealed_source_receipt_is_rejected_after_rehashing(self) -> None:
        verified = self._verified_manifest()
        self.source_receipt.write_text(
            json.dumps({**_SEALED_SOURCE_RECEIPT, "source_current": False}),
            encoding="utf-8",
        )
        entry = _artifact_entry(verified, self.source_receipt.name)
        entry["bytes"] = self.source_receipt.stat().st_size
        entry["sha256"] = hashlib.sha256(self.source_receipt.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "not source-current"):
            self.validate_mtp_manifest(self.root, verified)

    def test_source_receipt_field_checks_are_enforced(self) -> None:
        mutations = (
            (("schema_version",), 2, "not source-current"),
            (("fbsource_commit",), "g" * 40, "invalid fbsource commit"),
            (("oss_commit",), "z" * 40, "invalid OSS commit"),
            (
                ("source_manifest", "file_set_sha256"),
                "X" * 64,
                "source manifest file-set identity",
            ),
            (
                ("wgsl_manifest", "file_set_sha256"),
                "q" * 64,
                "WGSL manifest file-set identity",
            ),
            (
                ("verification", "source_checkout"),
                "pending",
                "verification is incomplete",
            ),
        )
        for keys, value, message in mutations:
            with self.subTest(keys=keys):
                receipt = copy.deepcopy(_SEALED_SOURCE_RECEIPT)
                target = receipt
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self.source_receipt.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.create_mtp_manifest(
                        self.root,
                        {
                            "pte": Path(self.pte.name),
                            "source": Path(self.source_receipt.name),
                        },
                        [Path(path.name) for path in self.ptds],
                    )

    def test_mtp_rejects_fake_semantic_evidence(self) -> None:
        mutations = {
            "negative mutation count": lambda evidence: evidence["k2_abi"].update(
                {"bufferMutationCount": -1}
            ),
            "empty input order": lambda evidence: evidence["k2_abi"].update(
                {"inputOrder": []}
            ),
            "false state alias": lambda evidence: evidence["k2_abi"].update(
                {"stateAlias": False}
            ),
            "negative operator count": lambda evidence: evidence["k2_abi"][
                "operatorCounts"
            ].update({"aten.argmax.default": -1}),
            "empty QAT cases": lambda evidence: evidence["qat_selection"].update(
                {"cases": []}
            ),
            "invalid token digest": lambda evidence: evidence["qat_selection"][
                "tokenOrdering"
            ].update({"sha256": "not-a-digest", "rawSha256": "not-a-digest"}),
            "incoherent token statistics": lambda evidence: evidence["qat_selection"][
                "tokenOrdering"
            ].update({"min": 1, "max": 2, "numel": 3, "uniqueCount": 4}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                manifest = self._verified_manifest()
                evidence = manifest["evidence"]
                assert isinstance(evidence, dict)
                mutate(evidence)
                with self.assertRaises(ValueError):
                    self.validate_mtp_manifest(self.root, manifest)

    def test_forced_verified_label_without_a_receipt_is_rejected(self) -> None:
        forced = self._pending_manifest()
        forced["provenance"] = copy.deepcopy(
            gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE
        )
        with self.assertRaisesRegex(ValueError, "unexpected artifact roles"):
            self.validate_mtp_manifest(self.root, forced)

    def test_forced_pending_label_with_a_receipt_is_rejected(self) -> None:
        forced = self._verified_manifest()
        forced["provenance"] = copy.deepcopy(
            gemma4_manifest.MTP_PENDING_SOURCE_PROVENANCE
        )
        with self.assertRaisesRegex(ValueError, "unexpected artifact roles"):
            self.validate_mtp_manifest(self.root, forced)

    def test_source_verified_is_distinct_from_the_pending_closures(self) -> None:
        verified = gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE["source_closure"]
        pending = gemma4_manifest.MTP_PENDING_SOURCE_PROVENANCE["source_closure"]
        self.assertEqual(verified, "source_verified")
        self.assertEqual(pending, "pending_final_source_receipt")
        self.assertNotEqual(verified, pending)
        self.assertNotEqual(
            gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE,
            gemma4_manifest.MTP_PENDING_SOURCE_PROVENANCE,
        )
        self.assertIn(pending, gemma4_manifest.MTP_PENDING_SOURCE_CLOSURES)
        self.assertIn(
            gemma4_manifest.MTP_ACCEPTED_PROVENANCE["source_closure"],
            gemma4_manifest.MTP_PENDING_SOURCE_CLOSURES,
        )
        self.assertNotIn(verified, gemma4_manifest.MTP_PENDING_SOURCE_CLOSURES)


class MTPExportPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.artifact_root = self.root / "sealed"
        self.output_path = self.artifact_root / "model.pte"
        self.receipt_path = self.root / "receipts" / "manifest.json"
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        self.source_receipt = self.source_root / "source_receipt.json"
        _write_source_receipt(self.source_receipt)

    def _write_staged_artifacts(self, staging: Path) -> tuple[Path, list[Path]]:
        staged_pte = staging / self.output_path.name
        staged_ptds = [staging / f"constants_{index}.ptd" for index in range(3)]
        staged_pte.write_bytes(b"mtp-pte")
        for index, path in enumerate(staged_ptds):
            path.write_bytes(f"mtp-ptd-{index}".encode("utf-8"))
        return staged_pte, staged_ptds

    def _finalize(
        self,
        staging: Path,
        staged_pte: Path,
        staged_ptds: list[Path],
        source_receipt: Path | None = None,
    ) -> Path:
        self.assertTrue(hasattr(gemma4_manifest, "finalize_mtp_export"))
        return gemma4_manifest.finalize_mtp_export(
            staging,
            self.output_path,
            self.receipt_path,
            staged_pte,
            staged_ptds,
            source_receipt or self.source_receipt,
            _generated_mtp_evidence(),
        )

    def _published_paths(self) -> tuple[Path, ...]:
        return (
            *(self.artifact_root / f"constants_{index}.ptd" for index in range(3)),
            self.artifact_root / self.source_receipt.name,
            self.output_path,
            self.receipt_path,
        )

    def _assert_no_publications(self) -> None:
        for path in self._published_paths():
            with self.subTest(unpublished=path.name):
                self.assertFalse(path.exists() or path.is_symlink())

    def _assert_validation_failure_rolls_back(self, failure_call: int) -> None:
        real_validate = gemma4_manifest.validate_mtp_manifest
        call_count = 0

        def validate_then_fail(root: Path, manifest: dict[str, object]) -> None:
            nonlocal call_count
            call_count += 1
            real_validate(root, manifest)
            if call_count == failure_call:
                self.assertEqual(self.receipt_path.is_file(), failure_call == 3)
                raise ValueError(f"injected validation failure at call {failure_call}")

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with mock.patch.object(
                gemma4_manifest,
                "validate_mtp_manifest",
                side_effect=validate_then_fail,
            ) as validator:
                with self.assertRaisesRegex(
                    ValueError,
                    f"injected validation failure at call {failure_call}",
                ):
                    self._finalize(staging, staged_pte, staged_ptds)

        self.assertEqual(validator.call_count, failure_call)
        self._assert_no_publications()

    def test_source_receipt_survives_staging_cleanup_and_final_manifest_validates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            result = self._finalize(staging, staged_pte, staged_ptds)

        self.assertEqual(result, self.receipt_path)
        self.assertFalse(staging.exists())
        published_source = self.artifact_root / self.source_receipt.name
        self.assertEqual(
            published_source.read_bytes(), self.source_receipt.read_bytes()
        )
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        gemma4_manifest.validate_mtp_manifest(self.artifact_root, receipt)
        self.assertEqual(
            sorted(path.name for path in self.artifact_root.iterdir()),
            [
                "constants_0.ptd",
                "constants_1.ptd",
                "constants_2.ptd",
                "model.pte",
                "source_receipt.json",
            ],
        )

    def test_final_validation_failure_rolls_back_every_published_artifact(
        self,
    ) -> None:
        self.artifact_root.mkdir()
        extra = self.artifact_root / "unsealed.bin"
        extra.write_bytes(b"not part of the sealed export")
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                self._finalize(staging, staged_pte, staged_ptds)

        self._assert_no_publications()
        self.assertEqual(list(self.artifact_root.iterdir()), [extra])

    def test_second_validation_failure_rolls_back_every_published_artifact(
        self,
    ) -> None:
        self._assert_validation_failure_rolls_back(2)

    def test_third_validation_failure_rolls_back_published_receipt_and_artifacts(
        self,
    ) -> None:
        self._assert_validation_failure_rolls_back(3)

    def test_keyboard_interrupt_rolls_back_every_published_artifact(self) -> None:
        real_validate = gemma4_manifest.validate_mtp_manifest
        call_count = 0

        def validate_then_interrupt(root: Path, manifest: dict[str, object]) -> None:
            nonlocal call_count
            call_count += 1
            real_validate(root, manifest)
            if call_count == 2:
                raise KeyboardInterrupt("injected publication interrupt")

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with mock.patch.object(
                gemma4_manifest,
                "validate_mtp_manifest",
                side_effect=validate_then_interrupt,
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "injected publication interrupt"
                ):
                    self._finalize(staging, staged_pte, staged_ptds)
        self._assert_no_publications()

    def _assert_post_link_sigint_rolls_back(
        self, interrupted_destination: Path
    ) -> None:
        real_link = os.link

        def link_then_sigint(source: Path, target: Path, **kwargs: object) -> None:
            real_link(source, target, **kwargs)
            if target == interrupted_destination:
                os.kill(os.getpid(), signal.SIGINT)

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            previous_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
            try:
                with mock.patch.object(
                    os,
                    "link",
                    side_effect=link_then_sigint,
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        self._finalize(staging, staged_pte, staged_ptds)
            finally:
                signal.signal(signal.SIGINT, previous_handler)

        self._assert_no_publications()
        self.assertTrue(self.source_receipt.is_file())

    def test_post_link_sigint_is_deferred_and_rolls_back_each_destination(self) -> None:
        for interrupted_destination in self._published_paths():
            with self.subTest(destination=interrupted_destination.name):
                self._assert_post_link_sigint_rolls_back(interrupted_destination)

    def test_post_link_sigint_fixture_restores_inherited_handler(self) -> None:
        inherited_signals: list[int] = []

        def inherited_handler(signum: int, _frame: object) -> None:
            inherited_signals.append(signum)

        previous_handler = signal.signal(signal.SIGINT, inherited_handler)
        try:
            self._assert_post_link_sigint_rolls_back(self.output_path)
            self.assertIs(signal.getsignal(signal.SIGINT), inherited_handler)
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        self.assertEqual(inherited_signals, [])

    def test_post_link_sigint_redelivers_to_inherited_handler(self) -> None:
        inherited_signals: list[int] = []
        real_link = os.link

        def inherited_handler(signum: int, _frame: object) -> None:
            inherited_signals.append(signum)

        def link_then_sigint(source: Path, target: Path, **kwargs: object) -> None:
            real_link(source, target, **kwargs)
            if target == self.output_path:
                os.kill(os.getpid(), signal.SIGINT)

        previous_handler = signal.signal(signal.SIGINT, inherited_handler)
        try:
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                staging = Path(directory)
                staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                with mock.patch.object(os, "link", side_effect=link_then_sigint):
                    self.assertEqual(
                        self.receipt_path,
                        self._finalize(staging, staged_pte, staged_ptds),
                    )
            self.assertIs(signal.getsignal(signal.SIGINT), inherited_handler)
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        self.assertEqual(inherited_signals, [signal.SIGINT])
        for path in self._published_paths():
            with self.subTest(published=path.name):
                self.assertTrue(path.is_file())

    def test_uncertain_post_link_exception_leaves_no_receipt_and_retry_fails_closed(
        self,
    ) -> None:
        real_link = os.link
        exception_injected = False

        def link_then_raise(source: Path, target: Path, **kwargs: object) -> None:
            nonlocal exception_injected
            real_link(source, target, **kwargs)
            if target == self.output_path and not exception_injected:
                exception_injected = True
                raise RuntimeError("injected uncertain post-link exception")

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with mock.patch.object(os, "link", side_effect=link_then_raise):
                with self.assertRaisesRegex(
                    RuntimeError, "injected uncertain post-link exception"
                ):
                    self._finalize(staging, staged_pte, staged_ptds)

        self.assertTrue(exception_injected)
        self.assertEqual(b"mtp-pte", self.output_path.read_bytes())
        self.assertFalse(self.receipt_path.exists())
        for candidate in self._published_paths():
            if candidate != self.output_path:
                self.assertFalse(candidate.exists() or candidate.is_symlink())

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                self._finalize(staging, staged_pte, staged_ptds)

        self.assertEqual(b"mtp-pte", self.output_path.read_bytes())
        self.assertFalse(self.receipt_path.exists())
        for candidate in self._published_paths():
            if candidate != self.output_path:
                self.assertFalse(candidate.exists() or candidate.is_symlink())

    def test_preexisting_destination_is_preserved_for_each_publication(self) -> None:
        for destination in self._published_paths():
            with self.subTest(destination=destination.name):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"foreign-preexisting")
                with tempfile.TemporaryDirectory(dir=self.root) as directory:
                    staging = Path(directory)
                    staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                    with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                        self._finalize(staging, staged_pte, staged_ptds)

                self.assertEqual(b"foreign-preexisting", destination.read_bytes())
                for candidate in self._published_paths():
                    if candidate != destination:
                        self.assertFalse(candidate.exists() or candidate.is_symlink())
                destination.unlink()

    def test_preexisting_same_inode_destination_is_not_owned_on_link_failure(
        self,
    ) -> None:
        destination = self.artifact_root / "constants_0.ptd"
        destination.parent.mkdir(parents=True)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            os.link(staged_ptds[0], destination)
            expected_identity = destination.stat().st_dev, destination.stat().st_ino

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                self._finalize(staging, staged_pte, staged_ptds)

        observed = destination.stat()
        self.assertEqual(expected_identity, (observed.st_dev, observed.st_ino))
        self.assertEqual(b"mtp-ptd-0", destination.read_bytes())
        for candidate in self._published_paths():
            if candidate != destination:
                self.assertFalse(candidate.exists() or candidate.is_symlink())

    def test_prelink_interrupt_never_owns_same_inode_destination(self) -> None:
        destination = self.artifact_root / "constants_0.ptd"
        destination.parent.mkdir(parents=True)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            os.link(staged_ptds[0], destination)
            expected_identity = destination.stat().st_dev, destination.stat().st_ino

            with mock.patch.object(
                os,
                "link",
                side_effect=KeyboardInterrupt("injected pre-link interrupt"),
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "injected pre-link interrupt"
                ):
                    self._finalize(staging, staged_pte, staged_ptds)

        observed = destination.stat()
        self.assertEqual(expected_identity, (observed.st_dev, observed.st_ino))
        self.assertEqual(b"mtp-ptd-0", destination.read_bytes())
        for candidate in self._published_paths():
            if candidate != destination:
                self.assertFalse(candidate.exists() or candidate.is_symlink())

    def test_destination_replacement_during_rollback_is_preserved(self) -> None:
        destination = self.artifact_root / "constants_0.ptd"
        foreign_contents = b"foreign-rollback-racer"
        real_rename = os.rename
        real_unlink = Path.unlink
        replacement_injected = False
        real_validate = gemma4_manifest.validate_mtp_manifest
        validation_count = 0

        def inject_replacement(path: Path) -> None:
            nonlocal replacement_injected
            if path != destination or replacement_injected:
                return
            replacement_injected = True
            real_unlink(path)
            path.write_bytes(foreign_contents)

        def rename_after_replacement(
            source: Path, target: Path, *args: object, **kwargs: object
        ) -> None:
            inject_replacement(Path(source))
            real_rename(source, target, *args, **kwargs)

        def unlink_after_replacement(
            path: Path, *args: object, **kwargs: object
        ) -> None:
            inject_replacement(Path(path))
            real_unlink(path, *args, **kwargs)

        def validate_then_fail(root: Path, manifest: dict[str, object]) -> None:
            nonlocal validation_count
            validation_count += 1
            real_validate(root, manifest)
            if validation_count == 2:
                raise ValueError("injected validation failure")

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with (
                mock.patch.object(os, "rename", side_effect=rename_after_replacement),
                mock.patch.object(
                    Path, "unlink", autospec=True, side_effect=unlink_after_replacement
                ),
                mock.patch.object(
                    gemma4_manifest,
                    "validate_mtp_manifest",
                    side_effect=validate_then_fail,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "injected validation failure"):
                    self._finalize(staging, staged_pte, staged_ptds)

        self.assertTrue(replacement_injected)
        self.assertEqual(foreign_contents, destination.read_bytes())
        for candidate in self._published_paths():
            if candidate != destination:
                self.assertFalse(candidate.exists() or candidate.is_symlink())

    def test_rollback_continues_after_foreign_restore_failure(self) -> None:
        earlier_destination = self.artifact_root / "constants_0.ptd"
        later_destination = self.output_path
        quarantined_foreign = b"foreign-moved-to-recovery"
        replacement_foreign = b"foreign-later-occupant"
        real_link = os.link
        real_rename = os.rename
        real_unlink = Path.unlink
        real_validate = gemma4_manifest.validate_mtp_manifest
        validation_count = 0
        replacement_injected = False
        restore_blocked = False
        injected_failure = ValueError("injected final validation failure")

        def rename_after_replacement(
            source: Path, target: Path, *args: object, **kwargs: object
        ) -> None:
            nonlocal replacement_injected
            if Path(source) == later_destination and not replacement_injected:
                replacement_injected = True
                real_unlink(later_destination)
                later_destination.write_bytes(quarantined_foreign)
            real_rename(source, target, *args, **kwargs)

        def block_foreign_restore(source: Path, target: Path, **kwargs: object) -> None:
            nonlocal restore_blocked
            if Path(target) == later_destination and Path(
                source
            ).parent.name.startswith(".mtp-publication-quarantine."):
                restore_blocked = True
                later_destination.write_bytes(replacement_foreign)
            real_link(source, target, **kwargs)

        def validate_then_fail(root: Path, manifest: dict[str, object]) -> None:
            nonlocal validation_count
            validation_count += 1
            real_validate(root, manifest)
            if validation_count == 2:
                raise injected_failure

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with (
                mock.patch.object(os, "rename", side_effect=rename_after_replacement),
                mock.patch.object(os, "link", side_effect=block_foreign_restore),
                mock.patch.object(
                    gemma4_manifest,
                    "validate_mtp_manifest",
                    side_effect=validate_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "injected final validation failure"
                ) as raised:
                    self._finalize(staging, staged_pte, staged_ptds)

        self.assertIs(injected_failure, raised.exception)
        self.assertTrue(replacement_injected)
        self.assertTrue(restore_blocked)
        self.assertFalse(earlier_destination.exists())
        self.assertEqual(replacement_foreign, later_destination.read_bytes())
        for candidate in self._published_paths():
            if candidate != later_destination:
                self.assertFalse(candidate.exists() or candidate.is_symlink())

        recovery_directories = [
            entry
            for entry in self.artifact_root.iterdir()
            if entry.is_dir() and entry.name.startswith(".mtp-publication-quarantine.")
        ]
        self.assertEqual(1, len(recovery_directories))
        recovery_entries = list(recovery_directories[0].iterdir())
        self.assertEqual(1, len(recovery_entries))
        self.assertEqual(quarantined_foreign, recovery_entries[0].read_bytes())
        notes = getattr(raised.exception, "__notes__", [])
        self.assertEqual(1, len(notes))
        self.assertIn(f"rollback cleanup failed for {later_destination}", notes[0])
        self.assertIn("foreign publication entry retained for recovery", notes[0])

    def test_staged_replacement_during_cleanup_is_preserved(self) -> None:
        foreign_contents = b"foreign-staged-racer"
        real_rename = os.rename
        real_unlink = Path.unlink
        replacement_injected = False

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            staged = staged_ptds[0]

            def inject_replacement(path: Path) -> None:
                nonlocal replacement_injected
                if path != staged or replacement_injected:
                    return
                replacement_injected = True
                real_unlink(path)
                path.write_bytes(foreign_contents)

            def rename_after_replacement(
                source: Path, target: Path, *args: object, **kwargs: object
            ) -> None:
                inject_replacement(Path(source))
                real_rename(source, target, *args, **kwargs)

            def unlink_after_replacement(
                path: Path, *args: object, **kwargs: object
            ) -> None:
                inject_replacement(Path(path))
                real_unlink(path, *args, **kwargs)

            with (
                mock.patch.object(os, "rename", side_effect=rename_after_replacement),
                mock.patch.object(
                    Path, "unlink", autospec=True, side_effect=unlink_after_replacement
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "staged artifact ownership changed"
                ):
                    self._finalize(staging, staged_pte, staged_ptds)

            self.assertTrue(replacement_injected)
            self.assertEqual(foreign_contents, staged.read_bytes())

        self._assert_no_publications()

    def test_foreign_racer_is_preserved_before_and_after_link(self) -> None:
        real_link = os.link
        for timing in ("before", "after"):
            for destination in self._published_paths():
                with self.subTest(timing=timing, destination=destination.name):

                    def race_then_interrupt(
                        source: Path,
                        target: Path,
                        expected_destination: Path = destination,
                        race_timing: str = timing,
                        **kwargs: object,
                    ) -> None:
                        if target != expected_destination:
                            real_link(source, target, **kwargs)
                            return
                        if race_timing == "after":
                            real_link(source, target, **kwargs)
                            target.unlink()
                        target.write_bytes(b"foreign-racer")
                        raise KeyboardInterrupt(f"injected {race_timing}-link race")

                    with tempfile.TemporaryDirectory(dir=self.root) as directory:
                        staging = Path(directory)
                        staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                        with mock.patch.object(
                            os,
                            "link",
                            side_effect=race_then_interrupt,
                        ):
                            with self.assertRaisesRegex(
                                KeyboardInterrupt, f"injected {timing}-link race"
                            ):
                                self._finalize(staging, staged_pte, staged_ptds)

                    self.assertEqual(b"foreign-racer", destination.read_bytes())
                    for candidate in self._published_paths():
                        if candidate != destination:
                            self.assertFalse(
                                candidate.exists() or candidate.is_symlink()
                            )
                    destination.unlink()

    def test_foreign_racer_after_link_return_is_detected(self) -> None:
        real_link = os.link
        for destination in self._published_paths():
            with self.subTest(destination=destination.name):

                def replace_link_with_foreign(
                    source: Path,
                    target: Path,
                    expected_destination: Path = destination,
                    **kwargs: object,
                ) -> None:
                    real_link(source, target, **kwargs)
                    if target == expected_destination:
                        target.unlink()
                        target.write_bytes(b"foreign-racer")

                with tempfile.TemporaryDirectory(dir=self.root) as directory:
                    staging = Path(directory)
                    staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                    with mock.patch.object(
                        os,
                        "link",
                        side_effect=replace_link_with_foreign,
                    ):
                        with self.assertRaisesRegex(ValueError, "ownership changed"):
                            self._finalize(staging, staged_pte, staged_ptds)

                self.assertEqual(b"foreign-racer", destination.read_bytes())
                for candidate in self._published_paths():
                    if candidate != destination:
                        self.assertFalse(candidate.exists() or candidate.is_symlink())
                destination.unlink()

    def test_publication_order_places_receipt_last(self) -> None:
        observed: list[Path] = []
        real_link = os.link

        def record_link(source: Path, target: Path, **kwargs: object) -> None:
            observed.append(target)
            real_link(source, target, **kwargs)

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with mock.patch.object(os, "link", side_effect=record_link):
                self._finalize(staging, staged_pte, staged_ptds)

        self.assertEqual(list(self._published_paths()), observed)

    def test_cross_device_publication_fails_before_linking(self) -> None:
        real_stat = os.stat

        def report_different_destination_device(
            path: Path, *args: object, **kwargs: object
        ) -> os.stat_result:
            result = real_stat(path, *args, **kwargs)
            if Path(path) != self.artifact_root:
                return result
            fields = list(result)
            fields[2] = result.st_dev + 1
            return os.stat_result(fields)

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            staging = Path(directory)
            staged_pte, staged_ptds = self._write_staged_artifacts(staging)
            with mock.patch.object(
                os,
                "stat",
                side_effect=report_different_destination_device,
            ):
                with self.assertRaisesRegex(ValueError, "same filesystem"):
                    self._finalize(staging, staged_pte, staged_ptds)

        self._assert_no_publications()

    def test_source_receipt_basename_must_not_alias_pte_or_ptd(self) -> None:
        for basename in (
            "model.pte",
            "constants_0.ptd",
            "constants_1.ptd",
            "constants_2.ptd",
        ):
            with self.subTest(basename=basename):
                source = self.source_root / basename
                _write_source_receipt(source)
                with tempfile.TemporaryDirectory(dir=self.root) as directory:
                    staging = Path(directory)
                    staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                    with self.assertRaisesRegex(
                        ValueError, "duplicate normalized artifact path"
                    ):
                        self._finalize(
                            staging,
                            staged_pte,
                            staged_ptds,
                            source_receipt=source,
                        )
                self._assert_no_publications()

    def test_source_receipt_must_be_a_regular_non_symlink_file(self) -> None:
        for name, target in (
            ("source-link.json", self.source_receipt.name),
            ("dangling-source-link.json", "missing-source.json"),
        ):
            with self.subTest(target=target):
                symlink = self.source_root / name
                symlink.symlink_to(target)
                with tempfile.TemporaryDirectory(dir=self.root) as directory:
                    staging = Path(directory)
                    staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                    with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                        self._finalize(
                            staging,
                            staged_pte,
                            staged_ptds,
                            source_receipt=symlink,
                        )
                self._assert_no_publications()

    def test_dangling_staged_pte_and_each_ptd_are_rejected(self) -> None:
        for artifact_index in range(4):
            with self.subTest(artifact_index=artifact_index):
                with tempfile.TemporaryDirectory(dir=self.root) as directory:
                    staging = Path(directory)
                    staged_pte, staged_ptds = self._write_staged_artifacts(staging)
                    artifact = [staged_pte, *staged_ptds][artifact_index]
                    artifact.unlink()
                    artifact.symlink_to("missing-staged-artifact")
                    with self.assertRaisesRegex(
                        ValueError, "staged artifact is not regular"
                    ):
                        self._finalize(staging, staged_pte, staged_ptds)
                self._assert_no_publications()


class CombinedRuntimeEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.plain_root = self.root / "plain-input"
        self.mtp_root = self.root / "mtp-input"
        self.plain_root.mkdir()
        self.mtp_root.mkdir()

        plain_pte = self.plain_root / "plain.pte"
        plain_source = self.plain_root / "source.json"
        plain_ptds = [self.plain_root / f"plain-{index}.ptd" for index in range(3)]
        plain_pte.write_bytes(b"plain-pte")
        _write_source_receipt(plain_source)
        for index, path in enumerate(plain_ptds):
            path.write_bytes(f"plain-{index}".encode())
        self.plain_manifest = gemma4_manifest.create_plain_manifest(
            self.plain_root,
            {"pte": Path(plain_pte.name), "source": Path(plain_source.name)},
            [Path(path.name) for path in plain_ptds],
        )

        self.mtp_pte = self.mtp_root / "mtp.pte"
        self.mtp_source = self.mtp_root / "source.json"
        self.mtp_ptds = [self.mtp_root / f"mtp-{index}.ptd" for index in range(3)]
        self.mtp_pte.write_bytes(b"mtp-pte")
        _write_source_receipt(self.mtp_source)
        for index, path in enumerate(self.mtp_ptds):
            path.write_bytes(f"mtp-{index}".encode())
        self.mtp_manifest = gemma4_manifest.create_mtp_manifest(
            self.mtp_root,
            {
                "pte": Path(self.mtp_pte.name),
                "source": Path(self.mtp_source.name),
            },
            [Path(path.name) for path in self.mtp_ptds],
        )
        self.mtp_manifest.update(
            {
                "evidence": _generated_mtp_evidence(),
                "provenance": copy.deepcopy(
                    gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE
                ),
            }
        )

        self.plain_receipt = self.root / "plain.json"
        self.mtp_receipt = self.root / "mtp.json"
        self.runtime_receipt = self.root / "runtime-source.json"
        self.plain_receipt.write_text(json.dumps(self.plain_manifest))
        self.mtp_receipt.write_text(json.dumps(self.mtp_manifest))
        self.runtime_paths: dict[str, dict[str, Path]] = {
            "profile": {
                "javascript": self.root / "mtp-profile/gemma4_mtp_profile.js",
                "wasm": self.root / "mtp-profile/gemma4_mtp_profile.wasm",
            },
            "wall": {
                "javascript": self.root / "mtp-wall/gemma4_mtp.js",
                "wasm": self.root / "mtp-wall/gemma4_mtp.wasm",
            },
        }
        for flavor, paths in self.runtime_paths.items():
            for kind, path in paths.items():
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(f"{flavor}-{kind}".encode())
        self.plain_runtime_paths: dict[str, dict[str, Path]] = {
            "profile": {
                "javascript": self.root / "plain-profile/webgpu_llama.js",
                "wasm": self.root / "plain-profile/webgpu_llama.wasm",
            },
            "wall": {
                "javascript": self.root / "plain-wall/webgpu_llama.js",
                "wasm": self.root / "plain-wall/webgpu_llama.wasm",
            },
        }
        for flavor, paths in self.plain_runtime_paths.items():
            for kind, path in paths.items():
                path.parent.mkdir(exist_ok=True)
                value = "plain-javascript" if kind == "javascript" else flavor
                path.write_bytes(value.encode())
        self.source_manifest = self.root / "source-manifest.json"
        self.wgsl_manifest = self.root / "wgsl-manifest.json"
        self.source_manifest.write_text(json.dumps(_test_source_manifest()))
        self.wgsl_manifest.write_text(json.dumps(_test_wgsl_manifest()))
        self.build_commands = {
            "mtp": {
                "profile": self.root / "mtp-profile-recipe.json",
                "wall": self.root / "mtp-wall-recipe.json",
            },
            "plain": {
                "profile": self.root / "plain-profile-recipe.json",
                "wall": self.root / "plain-wall-recipe.json",
            },
        }
        for model, builds in self.build_commands.items():
            for flavor, path in builds.items():
                path.write_text(
                    json.dumps(
                        gemma4_manifest.canonical_build_recipe(model, flavor),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
        self._write_runtime_receipt()
        self.target_prefill_receipt = self.root / "target-prefill.json"
        self._write_target_prefill_receipt()
        self.destination = self.root / "staged"

    def _write_runtime_receipt(self) -> None:
        receipt = self._create_runtime_receipt()
        self.runtime_receipt.write_text(json.dumps(receipt))

    def _write_target_prefill_receipt(self) -> None:
        self.target_prefill_receipt.write_text(
            json.dumps(_target_prefill_receipt(self.runtime_receipt)),
            encoding="utf-8",
        )

    def _create_runtime_receipt(
        self,
        *,
        runtime_paths: dict[str, object] | None = None,
        build_command_paths: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with mock.patch.object(
            gemma4_manifest,
            "create_source_manifest",
            return_value=_test_source_manifest(),
        ), mock.patch.object(
            gemma4_manifest,
            "create_wgsl_manifest",
            return_value=_test_wgsl_manifest(),
        ):
            return gemma4_manifest.create_runtime_source_receipt(
                fbsource_root=self.root,
                oss_root=self.root,
                backend_root=self.root,
                source_manifest_path=self.source_manifest,
                wgsl_manifest_path=self.wgsl_manifest,
                manifest_paths={
                    "mtp": self.mtp_receipt,
                    "plain": self.plain_receipt,
                },
                model_roots={"mtp": self.mtp_root, "plain": self.plain_root},
                runtime_paths=runtime_paths
                or {
                    "mtp": self.runtime_paths,
                    "plain": self.plain_runtime_paths,
                },
                build_command_paths=build_command_paths or self.build_commands,
            )

    def _demote_mtp_to_pending(self) -> dict[str, object]:
        """Pending means no source receipt at all, not merely a pending label."""
        self.mtp_source.unlink()
        manifest = gemma4_manifest.create_mtp_manifest(
            self.mtp_root,
            {"pte": Path(self.mtp_pte.name)},
            [Path(path.name) for path in self.mtp_ptds],
        )
        manifest["evidence"] = _generated_mtp_evidence()
        return manifest

    def _staged_envelope(self) -> dict[str, object]:
        return json.loads(
            (self.destination / "gemma4_webgpu_combined_runtime.json").read_text()
        )

    def _stage(self, *, refresh_target_prefill: bool = True) -> None:
        if refresh_target_prefill:
            self._write_target_prefill_receipt()
        gemma4_manifest.stage_combined_runtime(
            self.destination,
            self.plain_root,
            self.plain_receipt,
            self.mtp_root,
            self.mtp_receipt,
            self.runtime_receipt,
            self.runtime_paths["wall"]["javascript"],
            self.runtime_paths["wall"]["wasm"],
            self.runtime_paths["profile"]["javascript"],
            self.runtime_paths["profile"]["wasm"],
            plain_profile_javascript_path=self.plain_runtime_paths["profile"][
                "javascript"
            ],
            plain_profile_wasm_path=self.plain_runtime_paths["profile"]["wasm"],
            plain_wall_javascript_path=self.plain_runtime_paths["wall"]["javascript"],
            plain_wall_wasm_path=self.plain_runtime_paths["wall"]["wasm"],
            source_manifest_path=self.source_manifest,
            wgsl_manifest_path=self.wgsl_manifest,
            build_recipe_paths=self.build_commands,
            target_prefill_receipt_path=self.target_prefill_receipt,
        )

    def test_target_prefill_receipt_is_staged_in_envelope_v3(self) -> None:
        self._stage()
        envelope = self._staged_envelope()
        self.assertEqual(envelope["schema_version"], 3)
        self.assertEqual(set(envelope["receipts"]), {"mtp", "plain", "target_prefill"})
        identity = envelope["receipts"]["target_prefill"]
        self.assertEqual(identity["path"], "receipts/target_prefill.json")
        self.assertEqual(
            {key: identity[key] for key in ("bytes", "sha256")},
            _identity(self.target_prefill_receipt),
        )

    def test_final_schema_versions_are_split_by_receipt_role(self) -> None:
        self.assertEqual(_sealed_source_receipt()["schema_version"], 3)
        self.assertEqual(
            gemma4_manifest.canonical_build_recipe("plain", "profile")[
                "schema_version"
            ],
            2,
        )
        self.assertEqual(
            json.loads(self.runtime_receipt.read_text())["schema_version"], 4
        )
        self._stage()
        self.assertEqual(self._staged_envelope()["schema_version"], 3)

    def test_build_recipes_bind_model_specific_factories_and_stems(self) -> None:
        expected = {
            ("mtp", "profile"): ("createGemma4MtpProfile", "gemma4_mtp_profile"),
            ("mtp", "wall"): ("createGemma4Mtp", "gemma4_mtp"),
            ("plain", "profile"): ("createWebGPULlama", "webgpu_llama"),
            ("plain", "wall"): ("createWebGPULlama", "webgpu_llama"),
        }
        for (model, flavor), (factory, output_stem) in expected.items():
            with self.subTest(model=model, flavor=flavor):
                recipe = gemma4_manifest.canonical_build_recipe(model, flavor)
                self.assertEqual(recipe["factory"], factory)
                self.assertEqual(recipe["output_stem"], output_stem)
                self.assertEqual(recipe["profiling_enabled"], flavor == "profile")
                self.assertTrue(
                    str(recipe["outputs"]["javascript"]).endswith(f"/{output_stem}.js")
                )
                self.assertTrue(
                    str(recipe["outputs"]["wasm"]).endswith(f"/{output_stem}.wasm")
                )

    def test_combined_staging_requires_target_prefill_receipt(self) -> None:
        self.assertIn(
            "target_prefill_receipt_path",
            inspect.signature(gemma4_manifest.stage_combined_runtime).parameters,
        )

    def test_target_prefill_receipt_must_bind_runtime_source(self) -> None:
        receipt = json.loads(self.target_prefill_receipt.read_text())
        receipt["producer"]["runtime_source_receipt"]["sha256"] = "f" * 64
        self.target_prefill_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "runtime source identity"):
            self._stage(refresh_target_prefill=False)

    def test_target_prefill_receipt_must_bind_runtime_source_head(self) -> None:
        receipt = json.loads(self.target_prefill_receipt.read_text())
        receipt["producer"]["fbsource_commit"] = "f" * 40
        self.target_prefill_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "fbsource commit"):
            self._stage(refresh_target_prefill=False)

    def test_target_prefill_receipt_tamper_after_staging_is_rejected(self) -> None:
        self._stage()
        envelope = self._staged_envelope()
        path = self.destination / "receipts/target_prefill.json"
        receipt = json.loads(path.read_text())
        receipt["contexts"]["513"]["prefill_token_raw"] += 1
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "target.prefill|target-prefill"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_recanonicalized_target_receipt_swap_still_rechecks_source_link(
        self,
    ) -> None:
        self._stage()
        envelope = self._staged_envelope()
        target_path = self.destination / "receipts/target_prefill.json"
        swapped = json.loads(target_path.read_text())
        other_runtime_source = self.root / "other-runtime-source.json"
        other_runtime_source.write_bytes(
            (self.destination / "receipts/runtime_source.json").read_bytes() + b"\n"
        )
        swapped["producer"]["runtime_source_receipt"] = _identity(other_runtime_source)
        target_path.write_text(json.dumps(swapped), encoding="utf-8")
        envelope["receipts"]["target_prefill"].update(_identity(target_path))

        with self.assertRaisesRegex(ValueError, "runtime source identity"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_recanonicalized_target_receipt_still_binds_producer_resource(
        self,
    ) -> None:
        self._stage()
        envelope = self._staged_envelope()
        target_path = self.destination / "receipts/target_prefill.json"
        swapped = json.loads(target_path.read_text())
        swapped["producer"]["source_sha256"] = "f" * 64
        target_path.write_text(json.dumps(swapped), encoding="utf-8")
        envelope["receipts"]["target_prefill"].update(_identity(target_path))

        with self.assertRaisesRegex(ValueError, "producer source hash"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_combined_runtime_v2_is_rejected(self) -> None:
        self._stage()
        envelope = self._staged_envelope()
        envelope["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema version"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_plain_wall_and_profile_runtimes_are_staged_and_bound(self) -> None:
        self._stage()
        plain = self._staged_envelope()["runtime"]["plain"]
        self.assertEqual(set(plain), {"profile", "wall"})
        for flavor, paths in self.plain_runtime_paths.items():
            for kind, suffix in (("javascript", "js"), ("wasm", "wasm")):
                with self.subTest(flavor=flavor, kind=kind):
                    identity = plain[flavor][kind]
                    self.assertEqual(sorted(identity), ["bytes", "path", "sha256"])
                    self.assertEqual(
                        identity["path"], f"runtime/plain/{flavor}.{suffix}"
                    )
                    self.assertEqual(
                        {key: identity[key] for key in ("bytes", "sha256")},
                        _identity(paths[kind]),
                    )
                    self.assertEqual(
                        (self.destination / str(identity["path"])).read_bytes(),
                        paths[kind].read_bytes(),
                    )

    def test_plain_identical_javascript_still_binds_distinct_wasm_pairs(
        self,
    ) -> None:
        receipt = self._create_runtime_receipt()
        wall = receipt["runtime"]["plain"]["wall"]
        profile = receipt["runtime"]["plain"]["profile"]
        self.assertEqual(wall["javascript"], profile["javascript"])
        self.assertNotEqual(wall["wasm"], profile["wasm"])

    def test_plain_runtime_removes_the_missing_adapter_status(self) -> None:
        self._stage()
        views = self._staged_envelope()["views"]
        self.assertNotIn("blocked_missing_generic_browser_adapter", json.dumps(views))
        self.assertEqual(views["plain"]["runtime"], "plain")

    def test_tampered_plain_runtime_is_rejected(self) -> None:
        self._stage()
        envelope = self._staged_envelope()
        wasm = envelope["runtime"]["plain"]["wall"]["wasm"]
        (self.destination / str(wasm["path"])).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "plain wall wasm is not bound"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_runtime_receipt_binds_plain_wall_bytes(self) -> None:
        self.plain_runtime_paths["wall"]["wasm"].write_bytes(b"rebuilt-after-receipt")
        with self.assertRaisesRegex(ValueError, "plain wall wasm is not bound"):
            self._stage()

    def test_runtime_receipt_v1_is_rejected(self) -> None:
        receipt = json.loads(self.runtime_receipt.read_text())
        receipt["schema_version"] = 1
        self.runtime_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "schema version 4"):
            self._stage()

    def test_runtime_receipt_v2_is_rejected(self) -> None:
        receipt = json.loads(self.runtime_receipt.read_text())
        receipt["schema_version"] = 2
        self.runtime_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "schema version 4"):
            self._stage()

    def test_runtime_receipt_v3_is_rejected(self) -> None:
        receipt = json.loads(self.runtime_receipt.read_text())
        receipt["schema_version"] = 3
        self.runtime_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "schema version 4"):
            self._stage()

    def test_runtime_receipt_does_not_accept_caller_supplied_heads(self) -> None:
        parameters = inspect.signature(
            gemma4_manifest.create_runtime_source_receipt
        ).parameters
        self.assertNotIn("fbsource_commit", parameters)
        self.assertNotIn("oss_commit", parameters)

    def test_runtime_receipt_rejects_model_source_receipt_from_other_head(
        self,
    ) -> None:
        receipt = _sealed_source_receipt()
        receipt["fbsource_commit"] = "9" * 40
        receipt["oss_commit"] = "8" * 40
        source_manifest = receipt["source_manifest"]
        wgsl_manifest = receipt["wgsl_manifest"]
        assert isinstance(source_manifest, dict) and isinstance(wgsl_manifest, dict)
        checkouts = source_manifest["checkouts"]
        assert isinstance(checkouts, dict)
        checkouts["fbsource"]["head"] = "9" * 40
        checkouts["oss"]["head"] = "8" * 40
        wgsl_manifest["fbsource_commit"] = "9" * 40
        (self.plain_root / "source.json").write_text(json.dumps(receipt))
        self.mtp_source.write_text(json.dumps(receipt))

        self.plain_manifest = gemma4_manifest.create_plain_manifest(
            self.plain_root,
            {"pte": Path("plain.pte"), "source": Path("source.json")},
            [Path(f"plain-{index}.ptd") for index in range(3)],
        )
        self.mtp_manifest = gemma4_manifest.create_mtp_manifest(
            self.mtp_root,
            {"pte": Path("mtp.pte"), "source": Path("source.json")},
            [Path(f"mtp-{index}.ptd") for index in range(3)],
        )
        self.mtp_manifest["evidence"] = _generated_mtp_evidence()
        self.plain_receipt.write_text(json.dumps(self.plain_manifest))
        self.mtp_receipt.write_text(json.dumps(self.mtp_manifest))

        with self.assertRaisesRegex(ValueError, "source receipt.*head"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_opaque_source_manifest(self) -> None:
        self.source_manifest.write_text('{"source": "current"}')
        with self.assertRaisesRegex(ValueError, "source manifest"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_opaque_wgsl_manifest(self) -> None:
        self.wgsl_manifest.write_text('{"wgsl": "current"}')
        with self.assertRaisesRegex(ValueError, "WGSL manifest"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_text_build_recipe(self) -> None:
        self.build_commands["plain"]["wall"].write_text("build plain wall\n")
        with self.assertRaisesRegex(ValueError, "plain wall recipe"):
            self._write_runtime_receipt()

    def test_build_recipe_v1_is_rejected(self) -> None:
        path = self.build_commands["plain"]["wall"]
        recipe = json.loads(path.read_text())
        recipe["schema_version"] = 1
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "plain wall recipe"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_source_copy_identity_mutation(self) -> None:
        source = json.loads(self.source_manifest.read_text())
        for copy_identity in source["files"][0]["copies"].values():
            copy_identity["sha256"] = "0" * 64
        self.source_manifest.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "live clean checkouts"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_structurally_valid_wgsl_hash_mutation(
        self,
    ) -> None:
        manifest = json.loads(self.wgsl_manifest.read_text())
        manifest["files"][0]["sha256"] = "0" * 64
        self.wgsl_manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "live generator closure"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_missing_wgsl_registry(self) -> None:
        manifest = json.loads(self.wgsl_manifest.read_text())
        manifest["files"] = [
            entry for entry in manifest["files"] if entry["role"] != "global_registry"
        ]
        manifest["file_set_sha256"] = _set_digest(
            [
                {"path": entry["path"], "role": entry["role"]}
                for entry in manifest["files"]
            ]
        )
        self.wgsl_manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "WGSL manifest.*registry"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_declared_wgsl_orphan(self) -> None:
        manifest = json.loads(self.wgsl_manifest.read_text())
        manifest["orphans"] = ["runtime/ops/add/orphan_wgsl.h"]
        self.wgsl_manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "WGSL manifest.*orphan"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_recipe_contract_mutations(self) -> None:
        mutations = {
            "target": ("target", "wrong_target"),
            "profile": ("profiling_enabled", True),
            "factory": ("factory", "createWrongFactory"),
        }
        for label, (key, value) in mutations.items():
            with self.subTest(label=label):
                path = self.build_commands["plain"]["wall"]
                original = path.read_text()
                recipe = json.loads(original)
                recipe[key] = value
                path.write_text(json.dumps(recipe))
                with self.assertRaisesRegex(ValueError, "plain wall recipe"):
                    self._write_runtime_receipt()
                path.write_text(original)

        path = self.build_commands["mtp"]["wall"]
        recipe = json.loads(path.read_text())
        recipe["outputs"]["wasm"] = "wrong.wasm"
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "MTP wall recipe"):
            self._write_runtime_receipt()

    def test_runtime_receipt_rejects_factory_or_output_stem_mutation(self) -> None:
        for key, value in (
            ("factory", "createWrongFactory"),
            ("output_stem", "wrong_output"),
        ):
            with self.subTest(key=key):
                receipt = json.loads(self.runtime_receipt.read_text())
                receipt["runtime"]["mtp"]["profile"][key] = value
                self.runtime_receipt.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, f"{key} mismatch"):
                    self._stage()
                self._write_runtime_receipt()

    def test_runtime_receipt_rejects_noncanonical_product_basename(self) -> None:
        runtime_paths = {
            "mtp": {
                flavor: dict(paths) for flavor, paths in self.runtime_paths.items()
            },
            "plain": {
                flavor: dict(paths)
                for flavor, paths in self.plain_runtime_paths.items()
            },
        }
        wrong = self.root / "mtp-wall/wrong.js"
        wrong.write_bytes(b"wrong-basename")
        runtime_paths["mtp"]["wall"]["javascript"] = wrong
        with self.assertRaisesRegex(ValueError, "JavaScript basename mismatch"):
            self._create_runtime_receipt(runtime_paths=runtime_paths)

    def test_runtime_receipt_records_only_validated_not_executed_claims(self) -> None:
        receipt = json.loads(self.runtime_receipt.read_text())
        self.assertEqual(
            receipt.get("verification"),
            {
                "build_execution": "not_attested",
                "recipe": "validated",
                "source_checkout": "verified",
                "wgsl_codegen": "verified",
            },
        )

    def test_runtime_receipt_rejects_unattested_execution_claim_upgrade(self) -> None:
        receipt = json.loads(self.runtime_receipt.read_text())
        receipt["verification"]["build_execution"] = "verified"
        self.runtime_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "verification claims"):
            self._stage()

    def test_runtime_receipt_generator_binds_source_builds_and_artifacts(
        self,
    ) -> None:
        receipt = self._create_runtime_receipt()
        self.assertEqual(receipt["schema_version"], 4)
        self.assertEqual(receipt["fbsource_commit"], "1" * 40)
        self.assertEqual(receipt["oss_commit"], "2" * 40)
        self.assertEqual(
            receipt["source_inputs"]["source_manifest"],
            {
                "path": "closure/source_manifest.json",
                **_identity(self.source_manifest),
            },
        )
        self.assertEqual(
            receipt["source_inputs"]["wgsl_manifest"],
            {
                "path": "closure/wgsl_manifest.json",
                **_identity(self.wgsl_manifest),
            },
        )
        self.assertEqual(receipt["runtime"]["plain"]["target"], "gemma4_plain_wasm")
        self.assertEqual(receipt["runtime"]["mtp"]["target"], "gemma4_spec_browser")
        self.assertEqual(
            receipt["runtime"]["plain"]["profile"]["factory"],
            "createWebGPULlama",
        )
        self.assertEqual(
            receipt["runtime"]["mtp"]["profile"]["factory"],
            "createGemma4MtpProfile",
        )
        self.assertEqual(
            receipt["runtime"]["mtp"]["profile"]["output_stem"],
            "gemma4_mtp_profile",
        )
        self.assertEqual(
            receipt["runtime"]["plain"]["wall"]["recipe"],
            {
                "path": "closure/recipes/plain-wall.json",
                **_identity(self.build_commands["plain"]["wall"]),
            },
        )

    def test_runtime_receipt_rejects_aliased_mtp_roles(self) -> None:
        for role in ("javascript", "wasm", "recipe"):
            with self.subTest(role=role):
                runtime_paths = {
                    "mtp": {
                        flavor: dict(paths)
                        for flavor, paths in self.runtime_paths.items()
                    },
                    "plain": {
                        flavor: dict(paths)
                        for flavor, paths in self.plain_runtime_paths.items()
                    },
                }
                build_recipes = {
                    model: dict(paths) for model, paths in self.build_commands.items()
                }
                if role == "recipe":
                    build_recipes["mtp"]["profile"] = build_recipes["mtp"]["wall"]
                else:
                    runtime_paths["mtp"]["profile"][role] = runtime_paths["mtp"][
                        "wall"
                    ][role]
                with self.assertRaisesRegex(
                    ValueError, f"wall/profile {role} identities must differ"
                ):
                    self._create_runtime_receipt(
                        runtime_paths=runtime_paths,
                        build_command_paths=build_recipes,
                    )

    def test_runtime_receipt_rejects_aliased_plain_wasm_recipe_or_pair(
        self,
    ) -> None:
        for role in ("pair", "wasm", "recipe"):
            with self.subTest(role=role):
                runtime_paths = {
                    "mtp": {
                        flavor: dict(paths)
                        for flavor, paths in self.runtime_paths.items()
                    },
                    "plain": {
                        flavor: dict(paths)
                        for flavor, paths in self.plain_runtime_paths.items()
                    },
                }
                build_recipes = {
                    model: dict(paths) for model, paths in self.build_commands.items()
                }
                if role == "recipe":
                    build_recipes["plain"]["profile"] = build_recipes["plain"]["wall"]
                else:
                    runtime_paths["plain"]["profile"]["wasm"] = runtime_paths["plain"][
                        "wall"
                    ]["wasm"]
                    if role == "pair":
                        runtime_paths["plain"]["profile"]["javascript"] = runtime_paths[
                            "plain"
                        ]["wall"]["javascript"]
                    else:
                        distinct_javascript = (
                            self.root / "plain-profile-distinct/webgpu_llama.js"
                        )
                        distinct_javascript.parent.mkdir(exist_ok=True)
                        distinct_javascript.write_bytes(b"distinct-javascript")
                        runtime_paths["plain"]["profile"][
                            "javascript"
                        ] = distinct_javascript
                with self.assertRaisesRegex(
                    ValueError, f"wall/profile {role} identities must differ"
                ):
                    self._create_runtime_receipt(
                        runtime_paths=runtime_paths,
                        build_command_paths=build_recipes,
                    )

    def test_staging_rehashes_source_and_build_recipe_bytes(self) -> None:
        for label, path in (
            ("source manifest", self.source_manifest),
            ("WGSL manifest", self.wgsl_manifest),
            ("plain profile recipe", self.build_commands["plain"]["profile"]),
            ("plain wall recipe", self.build_commands["plain"]["wall"]),
            ("MTP wall recipe", self.build_commands["mtp"]["wall"]),
            ("MTP profile recipe", self.build_commands["mtp"]["profile"]),
        ):
            with self.subTest(label=label):
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(ValueError, "is not bound"):
                    self._stage()
                path.write_bytes(original)

    def test_staged_closure_carries_every_bound_source_and_recipe_byte(self) -> None:
        self._stage()
        expected = {
            "closure/source_manifest.json": self.source_manifest,
            "closure/wgsl_manifest.json": self.wgsl_manifest,
            "closure/recipes/plain-profile.json": self.build_commands["plain"][
                "profile"
            ],
            "closure/recipes/plain-wall.json": self.build_commands["plain"]["wall"],
            "closure/recipes/mtp-wall.json": self.build_commands["mtp"]["wall"],
            "closure/recipes/mtp-profile.json": self.build_commands["mtp"]["profile"],
        }
        for relative, source in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    (self.destination / relative).read_bytes(), source.read_bytes()
                )

    def test_pending_plain_provenance_cannot_stage(self) -> None:
        self.plain_manifest["provenance"] = copy.deepcopy(
            gemma4_manifest.MTP_ACCEPTED_PROVENANCE
        )
        self.plain_receipt.write_text(json.dumps(self.plain_manifest))
        with self.assertRaisesRegex(ValueError, "plain.*provenance"):
            self._write_runtime_receipt()

    def test_stages_and_validates_exact_runtime_bytes(self) -> None:
        self._stage()
        manifest_path = self.destination / "gemma4_webgpu_combined_runtime.json"
        envelope = json.loads(manifest_path.read_text())
        gemma4_manifest.validate_combined_runtime_envelope(self.destination, envelope)
        self.assertEqual(
            envelope["views"]["mtp"]["status"],
            "pending_gpu_execution_validation",
        )

    def test_create_runtime_source_cli_carries_plain_profile_inputs(self) -> None:
        output = self.root / "runtime-source-cli.json"
        argv = [
            "create-runtime-source",
            "--output",
            str(output),
            "--fbsource-root",
            str(self.root),
            "--oss-root",
            str(self.root),
            "--backend-root",
            str(self.root),
            "--source-manifest",
            str(self.source_manifest),
            "--wgsl-manifest",
            str(self.wgsl_manifest),
            "--plain-manifest",
            str(self.plain_receipt),
            "--mtp-manifest",
            str(self.mtp_receipt),
            "--plain-root",
            str(self.plain_root),
            "--mtp-root",
            str(self.mtp_root),
        ]
        for model, paths in (
            ("plain", self.plain_runtime_paths),
            ("mtp", self.runtime_paths),
        ):
            for flavor, artifacts in paths.items():
                argv.extend(
                    [
                        f"--{model}-{flavor}-javascript",
                        str(artifacts["javascript"]),
                        f"--{model}-{flavor}-wasm",
                        str(artifacts["wasm"]),
                        f"--{model}-{flavor}-recipe",
                        str(self.build_commands[model][flavor]),
                    ]
                )
        with mock.patch.object(
            gemma4_manifest,
            "create_source_manifest",
            return_value=_test_source_manifest(),
        ), mock.patch.object(
            gemma4_manifest,
            "create_wgsl_manifest",
            return_value=_test_wgsl_manifest(),
        ):
            self.assertEqual(gemma4_manifest.main(argv), 0)
        receipt = json.loads(output.read_text())
        self.assertEqual(
            set(receipt["runtime"]["plain"]), {"profile", "target", "wall"}
        )

    def test_stage_runtime_cli_carries_every_plain_profile_input(self) -> None:
        self._write_target_prefill_receipt()
        destination = self.root / "staged-cli"
        argv = [
            "stage-runtime",
            "--destination-root",
            str(destination),
            "--plain-root",
            str(self.plain_root),
            "--plain-receipt",
            str(self.plain_receipt),
            "--mtp-root",
            str(self.mtp_root),
            "--mtp-receipt",
            str(self.mtp_receipt),
            "--runtime-source-receipt",
            str(self.runtime_receipt),
            "--target-prefill-receipt",
            str(self.target_prefill_receipt),
            "--plain-wall-javascript",
            str(self.plain_runtime_paths["wall"]["javascript"]),
            "--plain-wall-wasm",
            str(self.plain_runtime_paths["wall"]["wasm"]),
            "--plain-wall-recipe",
            str(self.build_commands["plain"]["wall"]),
            "--plain-profile-javascript",
            str(self.plain_runtime_paths["profile"]["javascript"]),
            "--plain-profile-wasm",
            str(self.plain_runtime_paths["profile"]["wasm"]),
            "--plain-profile-recipe",
            str(self.build_commands["plain"]["profile"]),
            "--mtp-wall-javascript",
            str(self.runtime_paths["wall"]["javascript"]),
            "--mtp-wall-wasm",
            str(self.runtime_paths["wall"]["wasm"]),
            "--mtp-wall-recipe",
            str(self.build_commands["mtp"]["wall"]),
            "--mtp-profile-javascript",
            str(self.runtime_paths["profile"]["javascript"]),
            "--mtp-profile-wasm",
            str(self.runtime_paths["profile"]["wasm"]),
            "--mtp-profile-recipe",
            str(self.build_commands["mtp"]["profile"]),
            "--source-manifest",
            str(self.source_manifest),
            "--wgsl-manifest",
            str(self.wgsl_manifest),
        ]
        self.assertEqual(gemma4_manifest.main(argv), 0)
        for relative, source in (
            (
                "runtime/plain/profile.js",
                self.plain_runtime_paths["profile"]["javascript"],
            ),
            (
                "runtime/plain/profile.wasm",
                self.plain_runtime_paths["profile"]["wasm"],
            ),
            (
                "closure/recipes/plain-profile.json",
                self.build_commands["plain"]["profile"],
            ),
        ):
            with self.subTest(relative=relative):
                self.assertEqual(
                    (destination / relative).read_bytes(), source.read_bytes()
                )

    def test_rejects_accepted_oracle_as_current_source(self) -> None:
        self.mtp_manifest = self._demote_mtp_to_pending()
        self.mtp_manifest["provenance"] = copy.deepcopy(
            gemma4_manifest.MTP_ACCEPTED_PROVENANCE
        )
        del self.mtp_manifest["evidence"]
        self.mtp_receipt.write_text(json.dumps(self.mtp_manifest))
        with self.assertRaisesRegex(ValueError, "source closure is still pending"):
            self._write_runtime_receipt()

    def test_rejects_pending_source_mtp_manifest(self) -> None:
        self.mtp_receipt.write_text(json.dumps(self._demote_mtp_to_pending()))
        with self.assertRaisesRegex(ValueError, "source closure is still pending"):
            self._write_runtime_receipt()

    def test_sealed_source_receipt_stages_as_source_verified(self) -> None:
        self.assertEqual(
            self.mtp_manifest["provenance"],
            gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE,
        )
        self._stage()
        gemma4_manifest.validate_combined_runtime_envelope(
            self.destination, self._staged_envelope()
        )

    def test_status_only_pending_label_cannot_stage(self) -> None:
        self._stage()
        envelope = self._staged_envelope()
        envelope["source_verification"]["mtp"]["provenance"][
            "source_closure"
        ] = "pending_D10_reproduction_receipt"
        with self.assertRaisesRegex(ValueError, "source closure is still pending"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_envelope_carries_a_source_verification_block(self) -> None:
        # Source closure only; PTE/WASM behaviour is a separate runtime gate.
        self._stage()
        verification = self._staged_envelope()["source_verification"]
        self.assertEqual(sorted(verification), ["mtp", "plain"])
        for label in ("mtp", "plain"):
            with self.subTest(view=label):
                self.assertEqual(
                    sorted(verification[label]), ["provenance", "source_receipt"]
                )
                self.assertEqual(
                    sorted(verification[label]["source_receipt"]),
                    ["bytes", "path", "sha256"],
                )
                self.assertEqual(
                    verification[label]["source_receipt"]["path"], "source.json"
                )
        self.assertEqual(
            verification["mtp"]["provenance"],
            gemma4_manifest.MTP_SOURCE_VERIFIED_PROVENANCE,
        )
        self.assertIsNone(verification["plain"]["provenance"])

    def test_source_verification_is_independent_of_runtime_evidence(self) -> None:
        self._stage()
        first = self._staged_envelope()
        for flavor, paths in self.runtime_paths.items():
            for kind, path in paths.items():
                path.write_bytes(f"{flavor}-{kind}-rebuilt".encode())
        self._write_runtime_receipt()
        self.destination = self.root / "staged-rebuilt"
        self._stage()
        second = self._staged_envelope()
        self.assertNotEqual(first["runtime"], second["runtime"])
        self.assertEqual(first["source_verification"], second["source_verification"])

    def test_tampered_source_receipt_is_not_masked_by_valid_runtime(self) -> None:
        self._stage()
        envelope = self._staged_envelope()
        (self.destination / "mtp" / "source.json").write_text(
            json.dumps({**_SEALED_SOURCE_RECEIPT, "oss_commit": "3" * 40})
        )
        wasm = envelope["runtime"]["mtp"]["wall"]["wasm"]
        self.assertEqual(
            _identity(self.destination / str(wasm["path"])),
            {"bytes": wasm["bytes"], "sha256": wasm["sha256"]},
        )
        with self.assertRaisesRegex(
            ValueError, "mtp source receipt byte or SHA-256 identity mismatch"
        ):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_rejects_runtime_not_bound_to_source_receipt(self) -> None:
        self.runtime_paths["wall"]["wasm"].write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "bound to its build receipt"):
            self._stage()

    def test_rejects_extra_staged_file(self) -> None:
        self._stage()
        (self.destination / "extra.bin").write_bytes(b"extra")
        envelope = json.loads(
            (self.destination / "gemma4_webgpu_combined_runtime.json").read_text()
        )
        with self.assertRaisesRegex(ValueError, "extra files"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )

    def test_rejects_swapped_runtime_roles(self) -> None:
        self._stage()
        envelope = json.loads(
            (self.destination / "gemma4_webgpu_combined_runtime.json").read_text()
        )
        profile = envelope["runtime"]["mtp"]["profile"]
        wall = envelope["runtime"]["mtp"]["wall"]
        profile["javascript"], wall["javascript"] = (
            wall["javascript"],
            profile["javascript"],
        )
        with self.assertRaisesRegex(ValueError, "non-canonical role bindings"):
            gemma4_manifest.validate_combined_runtime_envelope(
                self.destination, envelope
            )


class PlainManifestContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.pte = self.root / "model.pte"
        self.source_receipt = self.root / "source_receipt.json"
        self.ptds = [self.root / f"constants_{index}.ptd" for index in range(3)]
        self.pte.write_bytes(b"plain-pte")
        _write_source_receipt(self.source_receipt)
        for index, path in enumerate(self.ptds):
            path.write_bytes(f"plain-ptd-{index}".encode("utf-8"))
        self.manifest = self._create([path.name for path in self.ptds])

    def _create(self, ptd_names: list[str]) -> dict[str, object]:
        return create_plain_manifest(
            self.root,
            {
                "pte": Path(self.pte.name),
                "source": Path(self.source_receipt.name),
            },
            [Path(name) for name in ptd_names],
        )

    def test_plain_requires_exactly_three_ordered_ptds(self) -> None:
        for names in (
            [path.name for path in self.ptds[:2]],
            [path.name for path in self.ptds] + [self.pte.name],
        ):
            with self.subTest(count=len(names)):
                with self.assertRaisesRegex(ValueError, "exactly three ordered PTDs"):
                    self._create(names)

        truncated = copy.deepcopy(self.manifest)
        dropped = self.ptds[2].name
        truncated["artifacts"] = [
            entry for entry in truncated["artifacts"] if entry["path"] != dropped
        ]
        truncated["ptd_order"] = [
            path for path in truncated["ptd_order"] if path != dropped
        ]
        self.ptds[2].unlink()
        with self.assertRaisesRegex(ValueError, "exactly three ordered PTDs"):
            validate_plain_manifest(self.root, truncated)

    def test_plain_ptd_reordering_is_rejected(self) -> None:
        reordered = copy.deepcopy(self.manifest)
        order = reordered["ptd_order"]
        order[0], order[2] = order[2], order[0]
        with self.assertRaisesRegex(ValueError, "PTD order does not match"):
            validate_plain_manifest(self.root, reordered)

    def test_plain_export_contract_identity(self) -> None:
        self.assertEqual(self.manifest["export"], EXPORT_CONTRACT)
        mutated = copy.deepcopy(self.manifest)
        mutated["export"]["max_seq_len"] = 8961
        with self.assertRaisesRegex(ValueError, "export contract mismatch"):
            validate_plain_manifest(self.root, mutated)

    def test_plain_acquisition_identity(self) -> None:
        self.assertEqual(self.manifest["acquisition"], CHECKPOINT_ACQUISITION)
        self.assertEqual(
            self.manifest["model"]["architecture"], ARCHITECTURE_FINGERPRINT
        )
        mutated = copy.deepcopy(self.manifest)
        mutated["acquisition"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "acquisition identity mismatch"):
            validate_plain_manifest(self.root, mutated)


class CommittedArtifactHygieneTest(unittest.TestCase):
    def test_committed_manifests_reference_path_and_hash_only(self) -> None:
        documents = sorted((_package_root() / "manifests").glob("*.json"))
        self.assertIn("gemma4_e2b_webgpu.json", [path.name for path in documents])
        for path in documents:
            with self.subTest(manifest=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                artifacts = document["artifacts"]
                self.assertEqual(document["schema_version"], 1)
                self.assertNotEqual(artifacts, [])
                for entry in artifacts:
                    self.assertEqual(sorted(entry), _ARTIFACT_KEYS)
                    self.assertEqual(len(Path(str(entry["path"])).parts), 1)
                    self.assertIsInstance(entry["bytes"], int)
                    self.assertRegex(str(entry["sha256"]), "^[0-9a-f]{64}$")
                self.assertEqual(
                    document["ptd_order"],
                    [entry["path"] for entry in artifacts if entry["role"] == "ptd"],
                )

    def test_committed_plain_manifest_matches_the_export_identity(self) -> None:
        document = json.loads(
            (_package_root() / "manifests" / "gemma4_e2b_webgpu.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(document["export"], EXPORT_CONTRACT)
        self.assertEqual(document["acquisition"], CHECKPOINT_ACQUISITION)
        self.assertEqual(document["model"]["architecture"], ARCHITECTURE_FINGERPRINT)
        self.assertEqual(len(document["ptd_order"]), 3)
        self.assertEqual(
            sorted(entry["role"] for entry in document["artifacts"]),
            ["ptd", "ptd", "ptd", "pte"],
        )

    def test_no_model_binaries_are_committed(self) -> None:
        root = _package_root()
        self.assertTrue((root / "manifests" / "gemma4_e2b_webgpu.json").is_file())
        self.assertTrue((root / "config" / "e2b_config.json").is_file())
        self.assertEqual(
            sorted(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file() and path.suffix in _BINARY_SUFFIXES
            ),
            [],
        )

    def test_no_internal_paths_leak(self) -> None:
        documents = {
            "gemma4_webgpu_artifact_manifest.py": Path(gemma4_manifest.__file__),
            "backend_webgpu_artifact_manifest.py": Path(backend_manifest.__file__),
        }
        for path in sorted((_package_root() / "manifests").glob("*.json")):
            documents[path.name] = path
        self.assertGreaterEqual(len(documents), 3)
        for name, path in documents.items():
            text = path.read_text(encoding="utf-8")
            for pattern in _internal_patterns():
                with self.subTest(document=name, pattern=pattern.pattern):
                    self.assertIsNone(pattern.search(text))
