# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import re
import tempfile
import unittest

from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import torch

from executorch.examples.models.gemma4.export_assistant_webgpu_artifacts import (
    _tensor_sha256 as assistant_tensor_sha256,
)
from executorch.examples.models.gemma4.generate_target_prefill_oracle import (
    _tensor_bytes as target_prefill_tensor_bytes,
)
from executorch.examples.models.gemma4.target_prefill_contract import (
    canonical_json_bytes,
    final_chunk_range,
    prompt_plan_sha256,
    reviewed_producer_source_path,
    TARGET_PREFILL_AUTHORITY,
    TARGET_PREFILL_CONTEXTS,
    TARGET_PREFILL_ENVELOPE_KIND,
    TARGET_PREFILL_SCHEMA_VERSION,
)
from executorch.examples.models.gemma4.tests import (
    generate_mtp_spec_oracle as mtp_oracle,
)
from executorch.examples.models.gemma4.tests.generate_mtp_spec_oracle import (
    _build_parser,
    assemble_oracle_document,
    build_oracle_binding,
    K2_ROUND_ABI,
    K2_VOCAB_SIZE,
    ORACLE_CONTEXT_KEYS,
    ORACLE_TOKEN_BUDGET,
    OracleError,
    parse_contexts,
    production_oracle_is_acceptable,
    reconcile_k2_round,
    require_prefill_token_match,
    SUPPORTED_AUTHORITIES,
    TARGET_PREFILL_AUTHORITIES,
    validate_k2_abi_edge_census,
)
from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    ASSISTANT_CHECKPOINT_ACQUISITION,
    CHECKPOINT_ACQUISITION,
    MTP_EDGE_CENSUS,
    MTP_SOURCE_VERIFIED_PROVENANCE,
)


CONTRACT_CONTEXTS = "128,511,512,513,514,1024,2048,4096,4097,8192"
CONTRACT_CONTEXT_VALUES = [128, 511, 512, 513, 514, 1024, 2048, 4096, 4097, 8192]
STOP_TOKENS = [1, 106]


def _witness(index: int) -> dict[str, Any]:
    return {
        "layer0_av_sha256": f"{index:064x}",
        "layer0_qk_sha256": f"{index + 1:064x}",
        "logits_sha256": f"{index + 2:064x}",
        "prefill_token": 1000 + index,
    }


def _tensor_envelope(shape: list[int], digest_digit: int) -> dict[str, Any]:
    return {
        "byte_order": "little",
        "dtype": "float32",
        "layout": "row_major_contiguous",
        "sha256": f"{digest_digit:064x}",
        "shape": shape,
    }


def _target_prefill_v2_context(context: int, index: int) -> dict[str, Any]:
    final_start, final_length = final_chunk_range(context)
    arm_config = {
        "dtype": "float32",
        "enable_dynamic_shape": True,
        "group_size": 128,
        "max_seq_len": 8960,
        "text_quantize": "8da4w+emb4",
        "use_kv_cache": True,
        "variant": "e2b",
    }
    return {
        "arm_configs": {
            "custom_sdpa_fused": {**arm_config, "use_custom_sdpa": True},
            "manual_unfused": {**arm_config, "use_custom_sdpa": False},
        },
        "cache_reset_counts": {
            "custom_sdpa_fused": 35,
            "manual_unfused": 35,
        },
        "chunk_size": 512,
        "context": context,
        "final_chunk_length": final_length,
        "final_chunk_start": final_start,
        "layer0_manual_unfused_vs_custom_sdpa_fused": {
            "agreement": {
                "atol": 1e-4,
                "max_abs": 0.0,
                "passed": True,
                "rel_rms": 0.0,
                "rtol": 1e-3,
            },
            "custom_sdpa_fused": _tensor_envelope([1, final_length, 8, 256], index + 3),
            "manual_unfused": _tensor_envelope([1, final_length, 8, 256], index + 4),
        },
        "logits_post_softcap": _tensor_envelope([1, 1, K2_VOCAB_SIZE], index + 2),
        "logits_pre_softcap": _tensor_envelope([1, 1, K2_VOCAB_SIZE], index + 1),
        "prefill_token_post_softcap": 1000 + index,
        "prefill_token_raw": 1000 + index,
        "prompt_plan_sha256": prompt_plan_sha256(context),
    }


class MtpSpecOracleGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "assistant").mkdir()
        (self.root / "target").mkdir()
        (self.root / "target" / "generation_config.json").write_text(
            json.dumps({"eos_token_id": STOP_TOKENS}), encoding="utf-8"
        )
        self.manifest_path = self.root / "mtp-manifest.json"
        self.receipt_path = self.root / "target_prefill_oracle.json"
        self.output = self.root / "mtp_spec_oracle.json"
        self.manifest: dict[str, Any] = {
            "abi": copy.deepcopy(K2_ROUND_ABI),
            "artifacts": [
                {"path": "model.pte", "role": "pte"},
                {"path": "model0.ptd", "role": "ptd"},
                {"path": "model1.ptd", "role": "ptd"},
                {"path": "model2.ptd", "role": "ptd"},
            ],
            "checkpoints": {"assistant": "assistant", "target": "target"},
            "max_context_length": 8960,
            "method": "k2_round",
            "ptd_order": ["model0.ptd", "model1.ptd", "model2.ptd"],
            "schema_version": 1,
        }
        self.receipt: dict[str, Any] = {
            "authority": "portable_eager",
            "contexts": {
                str(context): _witness(index)
                for index, context in enumerate(CONTRACT_CONTEXT_VALUES)
            },
            "schema_version": 1,
        }
        self._write()

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def _build(self, contexts: str = CONTRACT_CONTEXTS) -> dict[str, Any]:
        return build_oracle_binding(
            self.manifest_path,
            self.receipt_path,
            "portable_eager",
            contexts,
            self.output,
        )

    def _expect_rejection(self, contexts: str = CONTRACT_CONTEXTS) -> None:
        self._write()
        with self.assertRaises(OracleError):
            self._build(contexts)

    def _records(self, binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for context in binding["contexts"]:
            key = str(context)
            record: dict[str, Any] = {name: [] for name in ORACLE_CONTEXT_KEYS}
            record["decoded_text"] = f"context {key}"
            record["target_prefill"] = binding["target_prefill"][key]
            records[key] = record
        return records

    def test_binding_binds_manifest_receipt_and_contract_contexts(self) -> None:
        binding = self._build()
        self.assertEqual(binding["contexts"], CONTRACT_CONTEXT_VALUES)
        self.assertEqual(binding["authority"], "portable_eager")
        self.assertEqual(binding["max_context_length"], 8960)
        self.assertEqual(binding["stop_tokens"], STOP_TOKENS)
        self.assertEqual(binding["token_budget"], ORACLE_TOKEN_BUDGET)
        self.assertEqual(binding["checkpoints"]["target"], self.root / "target")
        self.assertEqual(binding["checkpoints"]["assistant"], self.root / "assistant")
        self.assertEqual(
            binding["mtp_manifest_sha256"],
            hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            binding["target_prefill_oracle_sha256"],
            hashlib.sha256(self.receipt_path.read_bytes()).hexdigest(),
        )

    def test_target_prefill_witnesses_are_copied_verbatim(self) -> None:
        binding = self._build()
        self.assertEqual(
            sorted(binding["target_prefill"]),
            sorted(str(context) for context in CONTRACT_CONTEXT_VALUES),
        )
        for index, context in enumerate(CONTRACT_CONTEXT_VALUES):
            self.assertEqual(binding["target_prefill"][str(context)], _witness(index))

    def test_rejects_reinterpreted_or_augmented_witness(self) -> None:
        self.receipt["contexts"]["512"]["extra"] = 1
        self._expect_rejection()

    def test_assembled_document_binds_every_receipt_and_abi_identity(self) -> None:
        binding = self._build()
        document = assemble_oracle_document(binding, self._records(binding))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["method"], "k2_round")
        self.assertEqual(document["abi"], K2_ROUND_ABI)
        self.assertEqual(document["authority"], "portable_eager")
        self.assertEqual(document["contexts"], CONTRACT_CONTEXT_VALUES)
        self.assertEqual(document["stop_tokens"], STOP_TOKENS)
        self.assertEqual(document["token_budget"], ORACLE_TOKEN_BUDGET)
        self.assertEqual(
            document["mtp_manifest_sha256"], binding["mtp_manifest_sha256"]
        )
        self.assertEqual(
            document["target_prefill_oracle_sha256"],
            binding["target_prefill_oracle_sha256"],
        )
        self.assertEqual(
            sorted(document["records"]),
            sorted(str(context) for context in CONTRACT_CONTEXT_VALUES),
        )
        self.assertEqual(tuple(sorted(document["records"]["512"])), ORACLE_CONTEXT_KEYS)

    def test_assemble_rejects_reinterpreted_target_prefill(self) -> None:
        binding = self._build()
        records = self._records(binding)
        records["512"]["target_prefill"] = dict(records["512"]["target_prefill"])
        records["512"]["target_prefill"]["prefill_token"] += 1
        with self.assertRaises(OracleError):
            assemble_oracle_document(binding, records)

    def test_assemble_rejects_missing_extra_and_malformed_records(self) -> None:
        binding = self._build()
        missing = self._records(binding)
        del missing["512"]
        with self.assertRaises(OracleError):
            assemble_oracle_document(binding, missing)
        extra = self._records(binding)
        extra["99"] = extra["512"]
        with self.assertRaises(OracleError):
            assemble_oracle_document(binding, extra)
        malformed = self._records(binding)
        del malformed["512"]["kv_witnesses"]
        with self.assertRaises(OracleError):
            assemble_oracle_document(binding, malformed)

    def _expect_authority_rejection(self, authority: str) -> None:
        with self.assertRaises(OracleError) as caught:
            build_oracle_binding(
                self.manifest_path,
                self.receipt_path,
                authority,
                CONTRACT_CONTEXTS,
                self.output,
            )
        self.assertIn(f"unknown --authority {authority!r}", str(caught.exception))
        self.assertIn("portable_eager", str(caught.exception))

    def test_rejects_unknown_authority(self) -> None:
        for authority in ("", "portable", "PORTABLE_EAGER", "webgpu_live"):
            with self.subTest(authority=authority):
                self._expect_authority_rejection(authority)

    def test_unknown_authority_cannot_self_certify_via_a_matching_receipt(self) -> None:
        self.receipt["authority"] = "webgpu_live"
        self._write()
        self._expect_authority_rejection("webgpu_live")

    def test_rejects_manifest_schema_version(self) -> None:
        for version in (None, 0, 2, True, "1", 1.0):
            with self.subTest(version=version):
                if version is None:
                    del self.manifest["schema_version"]
                else:
                    self.manifest["schema_version"] = version
                self._expect_rejection()
                self.manifest["schema_version"] = 1

    def test_rejects_receipt_schema_version(self) -> None:
        for version in (None, 2, True, "1"):
            with self.subTest(version=version):
                if version is None:
                    del self.receipt["schema_version"]
                else:
                    self.receipt["schema_version"] = version
                self._expect_rejection()
                self.receipt["schema_version"] = 1

    def test_rejects_receipt_authority_mismatch(self) -> None:
        self.receipt["authority"] = "webgpu_live"
        self._expect_rejection()

    def test_rejects_mismatched_abi_manifest(self) -> None:
        mutations = (
            ("buffer_mutation_count", 30),
            ("seed_mutation_count", 0),
            ("user_inputs", ["input_pos", "input_ids", "is_round", "donor_length"]),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                self.manifest["abi"] = copy.deepcopy(K2_ROUND_ABI)
                self.manifest["abi"][key] = value
                self._expect_rejection()
        self.manifest["abi"] = copy.deepcopy(K2_ROUND_ABI)
        self.manifest["abi"]["operator_counts"]["aten.topk.default"] = 1
        self._expect_rejection()
        self.manifest["abi"] = copy.deepcopy(K2_ROUND_ABI)
        self.manifest["abi"]["user_outputs"][4]["dtype"] = "int64"
        self._expect_rejection()
        self.manifest["abi"] = copy.deepcopy(K2_ROUND_ABI)
        del self.manifest["abi"]["operator_counts"]
        self._expect_rejection()

    def test_rejects_wrong_method_name(self) -> None:
        for method in (None, "k1_round", "text_decoder"):
            with self.subTest(method=method):
                self.manifest["method"] = method
                self._expect_rejection()
        self.manifest["method"] = "k2_round"

    def test_rejects_artifact_role_counts(self) -> None:
        original = copy.deepcopy(self.manifest["artifacts"])
        self.manifest["artifacts"] = original + [{"path": "second.pte", "role": "pte"}]
        self._expect_rejection()
        self.manifest["artifacts"] = original[:3]
        self.manifest["ptd_order"] = ["model0.ptd", "model1.ptd"]
        self._expect_rejection()
        self.manifest["artifacts"] = original
        self.manifest["ptd_order"] = ["model0.ptd", "model1.ptd"]
        self._expect_rejection()

    def test_rejects_max_context_length(self) -> None:
        for value in (None, 0, -1, 8961, True, "8960"):
            with self.subTest(value=value):
                self.manifest["max_context_length"] = value
                self._expect_rejection()
        self.manifest["max_context_length"] = 8960

    def _expect_rejection_message(self, fragment: str) -> None:
        with self.assertRaises(OracleError) as caught:
            self._build()
        self.assertIn(fragment, str(caught.exception))

    def test_rejects_unreadable_manifest_and_receipt(self) -> None:
        manifest, receipt = self.manifest_path, self.receipt_path
        manifest.unlink()
        self._expect_rejection_message(f"unreadable MTP manifest: {manifest}")
        manifest.write_text("{not json", encoding="utf-8")
        self._expect_rejection_message(f"malformed MTP manifest: {manifest}")
        manifest.write_text("[]", encoding="utf-8")
        self._expect_rejection_message(
            f"MTP manifest must be a JSON object: {manifest}"
        )
        self._write()
        receipt.unlink()
        self._expect_rejection_message(f"unreadable target-prefill receipt: {receipt}")
        receipt.write_text("[1, 2]", encoding="utf-8")
        self._expect_rejection_message(
            f"target-prefill receipt must be a JSON object: {receipt}"
        )

    def test_rejects_existing_output(self) -> None:
        self.output.write_text("{}", encoding="utf-8")
        with self.assertRaises(OracleError):
            self._build()
        self.output.unlink()
        self.output.symlink_to(self.root / "missing.json")
        with self.assertRaises(OracleError):
            self._build()

    def test_rejects_broken_checkpoint_binding(self) -> None:
        for checkpoints in (
            None,
            {},
            {"target": "target"},
            {"assistant": "assistant", "target": "target", "extra": "target"},
            {"assistant": "assistant", "target": "missing"},
            {"assistant": "assistant", "target": ""},
            {"assistant": "assistant", "target": 3},
        ):
            with self.subTest(checkpoints=checkpoints):
                self.manifest["checkpoints"] = checkpoints
                self._expect_rejection()

    def test_rejects_invalid_stop_tokens(self) -> None:
        config = self.root / "target" / "generation_config.json"
        for value in ([], [1, 1], [-1], [K2_VOCAB_SIZE], ["1"], [True]):
            with self.subTest(value=value):
                config.write_text(json.dumps({"eos_token_id": value}), encoding="utf-8")
                self._expect_rejection()
        config.write_text(json.dumps({}), encoding="utf-8")
        self._expect_rejection()
        config.unlink()
        self._expect_rejection()

    def test_rejects_malformed_context_lists(self) -> None:
        for contexts in (
            "",
            " ",
            "128,",
            ",128",
            "128,,512",
            "-1",
            "0",
            "1.5",
            "128,128",
            "8961",
            "0x80",
            "128 512",
        ):
            with self.subTest(contexts=contexts):
                with self.assertRaises(OracleError):
                    self._build(contexts)

    def test_parse_contexts_accepts_the_command_contract_list(self) -> None:
        self.assertEqual(
            parse_contexts(CONTRACT_CONTEXTS, 8960), CONTRACT_CONTEXT_VALUES
        )

    def test_rejects_missing_or_malformed_context_witness(self) -> None:
        del self.receipt["contexts"]["4097"]
        self._expect_rejection()
        self.receipt["contexts"]["4097"] = _witness(8)
        self.receipt["contexts"]["4097"]["logits_sha256"] = "0" * 63
        self._expect_rejection()
        self.receipt["contexts"]["4097"] = _witness(8)
        self.receipt["contexts"]["4097"]["prefill_token"] = K2_VOCAB_SIZE
        self._expect_rejection()
        self.receipt["contexts"]["4097"] = _witness(8)
        del self.receipt["contexts"]["4097"]["layer0_qk_sha256"]
        self._expect_rejection()
        self.receipt["contexts"] = []
        self._expect_rejection()


class ProductionOracleBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "assistant").mkdir()
        (self.root / "target").mkdir()
        (self.root / "target" / "generation_config.json").write_text(
            json.dumps({"eos_token_id": STOP_TOKENS}), encoding="utf-8"
        )
        self.manifest_path = self.root / "mtp-manifest.json"
        self.receipt_path = self.root / "target_prefill_oracle.json"
        self.output = self.root / "mtp_spec_oracle.json"
        self.manifest: dict[str, Any] = {
            "abi": copy.deepcopy(K2_ROUND_ABI),
            "artifacts": [
                {"path": "model.pte", "role": "pte"},
                *[{"path": f"model{index}.ptd", "role": "ptd"} for index in range(3)],
            ],
            "checkpoints": {"assistant": "assistant", "target": "target"},
            "max_context_length": 8960,
            "method": "k2_round",
            "ptd_order": [f"model{index}.ptd" for index in range(3)],
            "schema_version": 1,
        }
        self.receipt: dict[str, Any] = {
            "authority": "portable_eager",
            "contexts": {
                str(context): _witness(index)
                for index, context in enumerate(CONTRACT_CONTEXT_VALUES)
            },
            "schema_version": 1,
        }
        self._write()
        self.combined_root = self.root / "combined"
        receipts = self.combined_root / "receipts"
        receipts.mkdir(parents=True)
        self.production_mtp_path = receipts / "mtp.json"
        self.production_mtp: dict[str, Any] = {
            "acquisition": {
                "assistant": ASSISTANT_CHECKPOINT_ACQUISITION,
                "target": CHECKPOINT_ACQUISITION,
            },
            "artifacts": copy.deepcopy(self.manifest["artifacts"]),
            "export": {"max_seq_len": 8960, "methods": ["k2_round"]},
            "provenance": MTP_SOURCE_VERIFIED_PROVENANCE,
            "ptd_order": copy.deepcopy(self.manifest["ptd_order"]),
        }
        self.production_mtp_path.write_text(
            json.dumps(self.production_mtp), encoding="utf-8"
        )
        self.production_receipt_path = receipts / "target_prefill.json"
        self.production_receipt: dict[str, Any] = {
            "authority": TARGET_PREFILL_AUTHORITY,
            "checkpoint_acquisition": CHECKPOINT_ACQUISITION,
            "contexts": {
                str(context): _target_prefill_v2_context(context, index)
                for index, context in enumerate(TARGET_PREFILL_CONTEXTS)
            },
            "envelope_kind": TARGET_PREFILL_ENVELOPE_KIND,
            "producer": {
                "fbsource_commit": "1" * 40,
                "runtime_source_receipt": {"bytes": 10, "sha256": "2" * 64},
                "source_path": reviewed_producer_source_path().name,
                "source_sha256": hashlib.sha256(
                    reviewed_producer_source_path().read_bytes()
                ).hexdigest(),
            },
            "run": {
                "command": ["owner-run"],
                "finished_at_utc": "2026-08-07T01:00:00Z",
                "host": "owner-host",
                "started_at_utc": "2026-08-07T00:00:00Z",
            },
            "schema_version": TARGET_PREFILL_SCHEMA_VERSION,
        }
        self._write_production_receipt()
        self.combined_envelope_path = (
            self.combined_root / "gemma4_webgpu_combined_runtime.json"
        )
        self.combined_envelope: dict[str, Any] = {
            "receipts": {
                "mtp": {"path": "receipts/mtp.json", "root": "mtp"},
                "target_prefill": {"path": "receipts/target_prefill.json"},
            },
            "schema_version": 3,
            "source_verification": {
                "mtp": {"provenance": MTP_SOURCE_VERIFIED_PROVENANCE}
            },
        }
        self.combined_envelope_path.write_text(
            json.dumps(self.combined_envelope), encoding="utf-8"
        )

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def _build(self) -> dict[str, Any]:
        return build_oracle_binding(
            self.manifest_path,
            self.receipt_path,
            "portable_eager",
            CONTRACT_CONTEXTS,
            self.output,
        )

    def _records(self, binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for context in binding["contexts"]:
            key = str(context)
            raw = {
                "bonus": 13,
                "candidates": [11, 12],
                "match_count": 2,
                "state_probe": 0.0,
                "target_greedy": [11, 12, 13],
            }
            decision = reconcile_k2_round(
                raw["candidates"],
                raw["target_greedy"],
                raw["match_count"],
                raw["bonus"],
                raw["state_probe"],
                context,
                ORACLE_TOKEN_BUDGET - 1,
                binding["stop_tokens"],
            )
            round_record = {"kv_witness": "a" * 64, **raw, **decision}
            target_prefill = binding["target_prefill"][key]
            prefill_token = target_prefill.get(
                "prefill_token_raw", target_prefill.get("prefill_token")
            )
            records[key] = {
                "accepted_prefix": [decision["accepted_drafts"]],
                "bonus_accounting": [decision["next_seed"]],
                "decoded_text": f"context {key}",
                "kv_witnesses": [round_record["kv_witness"]],
                "reset_replay": raw,
                "rounds": [round_record],
                "selected_logits": [decision["selected"]],
                "stop_handling": {
                    "discarded": [decision["discarded"]],
                    "stop_token": None,
                },
                "target_prefill": target_prefill,
                "useful_tokens": [
                    prefill_token,
                    *decision["committed"],
                ],
            }
        return records

    def _write_production_receipt(self) -> None:
        self.production_receipt_path.write_bytes(
            canonical_json_bytes(self.production_receipt)
        )

    def _build_production(self, contexts: str = CONTRACT_CONTEXTS) -> dict[str, Any]:
        with mock.patch(
            "executorch.examples.models.gemma4.tests.generate_mtp_spec_oracle."
            "validate_combined_runtime_envelope"
        ) as validator:
            binding = build_oracle_binding(
                self.manifest_path,
                None,
                "portable_eager",
                contexts,
                self.output,
                combined_runtime_root=self.combined_root,
            )
        validator.assert_called_once_with(self.combined_root, self.combined_envelope)
        return binding

    def _build_production_direct(self) -> dict[str, Any]:
        with mock.patch(
            "executorch.examples.models.gemma4.tests.generate_mtp_spec_oracle."
            "validate_combined_runtime_envelope"
        ) as validator:
            binding = mtp_oracle.build_production_oracle_binding(
                self.production_mtp_path,
                self.root / "target",
                self.root / "assistant",
                "portable_eager",
                CONTRACT_CONTEXTS,
                self.output,
                combined_runtime_root=self.combined_root,
            )
        validator.assert_called_once_with(self.combined_root, self.combined_envelope)
        return binding

    def test_production_binding_derives_staged_receipt_and_is_acceptable(self) -> None:
        binding = self._build_production_direct()
        document = assemble_oracle_document(binding, self._records(binding))
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["closure_state"], "full")
        self.assertEqual(document["target_prefill_authority"], "bound")
        self.assertEqual(document["replay_independence"], "eager_vs_lowered_only")
        self.assertTrue(production_oracle_is_acceptable(document))
        self.assertEqual(document["contexts"], list(TARGET_PREFILL_CONTEXTS))
        self.assertEqual(
            binding["target_prefill"]["512"]["prefill_token_raw"],
            self.production_receipt["contexts"]["512"]["prefill_token_raw"],
        )
        self.assertEqual(
            document["mtp_manifest_sha256"],
            hashlib.sha256(self.production_mtp_path.read_bytes()).hexdigest(),
        )

    def test_legacy_receipt_stays_non_accepting(self) -> None:
        binding = self._build()
        document = assemble_oracle_document(binding, self._records(binding))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["closure_state"], "absent")
        self.assertEqual(document["target_prefill_authority"], "legacy_unbound")
        self.assertFalse(production_oracle_is_acceptable(document))

    def test_legacy_combined_binding_uses_the_staged_mtp_identity(self) -> None:
        binding = self._build_production()
        document = assemble_oracle_document(binding, self._records(binding))
        self.assertEqual(
            hashlib.sha256(self.production_mtp_path.read_bytes()).hexdigest(),
            document["mtp_manifest_sha256"],
        )
        self.assertTrue(production_oracle_is_acceptable(document))

    def test_production_binding_requires_exact_ten_contexts(self) -> None:
        with self.assertRaisesRegex(OracleError, "exact ten contexts"):
            self._build_production("128,512")
        del self.production_receipt["contexts"]["4097"]
        self._write_production_receipt()
        with self.assertRaisesRegex(OracleError, "exact ten contexts"):
            self._build_production()

    def test_full_closure_requires_source_verified_mtp_provenance(self) -> None:
        self.production_mtp_path.write_text(
            json.dumps(
                {
                    "provenance": {
                        "artifact_status": "accepted_behavior_oracle",
                        "source_closure": "pending_final_source_rebuild",
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(OracleError, "not source verified"):
            self._build_production()

    def test_target_receipt_authority_domain_is_separate_from_replay(self) -> None:
        self.assertEqual(SUPPORTED_AUTHORITIES, ("portable_eager",))
        self.assertEqual(TARGET_PREFILL_AUTHORITIES, ("target_only_eager",))
        self.production_receipt["authority"] = "portable_eager"
        self._write_production_receipt()
        with self.assertRaisesRegex(OracleError, "target-prefill authority"):
            self._build_production()

    def test_raw_post_token_mismatch_and_k2_bonus_mismatch_fail_closed(self) -> None:
        witness = self.production_receipt["contexts"]["512"]
        witness["prefill_token_post_softcap"] += 1
        self._write_production_receipt()
        with self.assertRaisesRegex(OracleError, "raw/post-softcap"):
            self._build_production()
        witness["prefill_token_post_softcap"] = witness["prefill_token_raw"]
        require_prefill_token_match(512, witness["prefill_token_raw"], witness)
        with self.assertRaisesRegex(OracleError, "prefill token"):
            require_prefill_token_match(512, witness["prefill_token_raw"] + 1, witness)

    def test_production_validator_value_error_is_translated(self) -> None:
        with mock.patch(
            "executorch.examples.models.gemma4.tests.generate_mtp_spec_oracle."
            "validate_combined_runtime_envelope",
            side_effect=ValueError("source closure rejected"),
        ):
            with self.assertRaisesRegex(OracleError, "source closure rejected"):
                build_oracle_binding(
                    self.manifest_path,
                    None,
                    "portable_eager",
                    CONTRACT_CONTEXTS,
                    self.output,
                    combined_runtime_root=self.combined_root,
                )

    def test_acceptance_rejects_unknown_or_incomplete_states(self) -> None:
        binding = self._build_production_direct()
        document = assemble_oracle_document(binding, self._records(binding))
        for key, value in (
            ("closure_state", "absent"),
            ("target_prefill_authority", "legacy_unbound"),
        ):
            with self.subTest(key=key):
                mutated = copy.deepcopy(document)
                mutated[key] = value
                self.assertFalse(production_oracle_is_acceptable(mutated))
        mutated = copy.deepcopy(document)
        mutated["closure_state"] = "artifact_validated_pending_source"
        with self.assertRaisesRegex(OracleError, "closure_state"):
            production_oracle_is_acceptable(mutated)
        mutated = copy.deepcopy(document)
        del mutated["production_binding"]
        self.assertFalse(production_oracle_is_acceptable(mutated))

    def test_acceptance_rejects_malformed_oracle_content(self) -> None:
        binding = self._build_production_direct()
        document = assemble_oracle_document(binding, self._records(binding))
        mutations = (
            lambda value: value.__setitem__("abi", {}),
            lambda value: value.__setitem__("method", "forward"),
            lambda value: value.__setitem__("authority", "target_only_eager"),
            lambda value: value.__setitem__("records", {}),
            lambda value: value["records"]["512"].__setitem__("rounds", []),
            lambda value: value["records"]["512"].__setitem__(
                "reset_replay", {}
            ),
            lambda value: value["production_binding"].__setitem__(
                "mtp_manifest_sha256", "0" * 64
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                mutated = copy.deepcopy(document)
                mutate(mutated)
                self.assertFalse(production_oracle_is_acceptable(mutated))

    def test_acceptance_rejects_target_prefill_receipt_content_drift(self) -> None:
        binding = self._build_production_direct()
        document = assemble_oracle_document(binding, self._records(binding))
        mutated = copy.deepcopy(document)
        mutated["records"]["512"]["target_prefill"]["logits_pre_softcap"][
            "sha256"
        ] = "0" * 64
        self.assertFalse(production_oracle_is_acceptable(mutated))

    def test_main_refuses_to_write_an_unacceptable_production_oracle(self) -> None:
        binding = self._build_production_direct()
        records = self._records(binding)
        records["512"]["target_prefill"]["logits_pre_softcap"]["sha256"] = (
            "0" * 64
        )
        with mock.patch.object(
            mtp_oracle,
            "build_production_oracle_binding",
            return_value=binding,
        ), mock.patch.object(
            mtp_oracle,
            "run_portable_eager_authority",
            return_value=records,
        ):
            with self.assertRaisesRegex(OracleError, "failed production validation"):
                mtp_oracle.main(
                    [
                        "--production-mtp-manifest",
                        str(self.production_mtp_path),
                        "--target-checkpoint",
                        str(self.root / "target"),
                        "--assistant-checkpoint",
                        str(self.root / "assistant"),
                        "--combined-runtime-root",
                        str(self.combined_root),
                        "--authority",
                        "portable_eager",
                        "--contexts",
                        CONTRACT_CONTEXTS,
                        "--output",
                        str(self.output),
                    ]
                )
        self.assertFalse(self.output.exists())

    def test_cli_requires_combined_root_and_has_no_loose_receipt_flag(self) -> None:
        parser = _build_parser()
        help_text = " ".join(parser.format_help().split())
        option_strings = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertIn("target-only eager prefill evidence", help_text)
        self.assertIn("does not independently validate the shared model", help_text)
        self.assertIn("--oracle-binding-manifest", option_strings)
        self.assertIn("--production-mtp-manifest", option_strings)
        self.assertIn("--target-checkpoint", option_strings)
        self.assertIn("--assistant-checkpoint", option_strings)
        self.assertIn("--combined-runtime-root", option_strings)
        self.assertNotIn("--mtp-manifest", option_strings)
        self.assertNotIn("--target-prefill-oracle", option_strings)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--oracle-binding-manifest",
                    str(self.manifest_path),
                    "--authority",
                    "portable_eager",
                    "--contexts",
                    CONTRACT_CONTEXTS,
                    "--output",
                    str(self.output),
                ]
            )

    def test_k2_abi_counts_match_production_edge_census(self) -> None:
        validate_k2_abi_edge_census(MTP_EDGE_CENSUS)
        mutated = dict(MTP_EDGE_CENSUS)
        mutated["topk"] = 1
        with self.assertRaisesRegex(OracleError, "K=2 ABI"):
            validate_k2_abi_edge_census(mutated)

    def test_d6_and_d8_tensor_encoders_match(self) -> None:
        tensors = (
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1),
            torch.arange(12, dtype=torch.int64).reshape(3, 4),
            torch.arange(12, dtype=torch.int32).reshape(3, 4).transpose(0, 1),
        )
        for tensor in tensors:
            with self.subTest(dtype=tensor.dtype, contiguous=tensor.is_contiguous()):
                self.assertEqual(
                    hashlib.sha256(target_prefill_tensor_bytes(tensor)).hexdigest(),
                    assistant_tensor_sha256(tensor),
                )


class K2ReconcilerTest(unittest.TestCase):
    def _round(
        self,
        candidates: list[int],
        greedy: list[int],
        match_count: int,
        bonus: int,
        start_position: int = 2,
        token_budget: int = 3,
        stop_tokens: list[int] | None = None,
        state_probe: float = 0.0,
        vocab_size: int = K2_VOCAB_SIZE,
    ) -> dict[str, Any]:
        return reconcile_k2_round(
            candidates,
            greedy,
            match_count,
            bonus,
            state_probe,
            start_position,
            token_budget,
            STOP_TOKENS if stop_tokens is None else stop_tokens,
            vocab_size,
        )

    def test_match_counts_advance_position_by_match_plus_one(self) -> None:
        expected = ((0, [90], 3), (1, [10, 90], 4), (2, [10, 11, 90], 5))
        greedy_by_match = {
            0: [90, 91, 92],
            1: [10, 90, 92],
            2: [10, 11, 90],
        }
        for match_count, committed, next_position in expected:
            with self.subTest(match_count=match_count):
                decision = self._round(
                    [10, 11], greedy_by_match[match_count], match_count, 90
                )
                self.assertTrue(decision["valid"])
                self.assertEqual(decision["committed"], committed)
                self.assertEqual(decision["selected"], committed)
                self.assertEqual(decision["next_position"], next_position)
                self.assertEqual(decision["next_seed"], 90)
                self.assertEqual(decision["accepted_drafts"], match_count)
                self.assertFalse(decision["stopped"])
                self.assertEqual(decision["discarded"], [])

    def test_chained_rounds_walk_start_positions_two_three_five(self) -> None:
        first = self._round([10, 11], [90, 91, 92], 0, 90, start_position=2)
        self.assertEqual(first["next_position"], 3)
        self.assertEqual(first["next_seed"], 90)
        second = self._round([20, 21], [20, 91, 92], 1, 91, start_position=3)
        self.assertEqual(second["next_position"], 5)
        self.assertEqual(second["next_seed"], 91)
        third = self._round([30, 31], [30, 31, 92], 2, 92, start_position=5)
        self.assertEqual(third["next_position"], 8)
        self.assertEqual(third["next_seed"], 92)

    def test_seeds_from_bonus_never_from_draft_or_target_tail(self) -> None:
        decision = self._round([10, 11], [10, 90, 92], 1, 90)
        self.assertEqual(decision["next_seed"], 90)
        self.assertNotEqual(decision["next_seed"], 10)
        self.assertNotEqual(decision["next_seed"], 92)
        self.assertEqual(decision["selected"][-1], decision["next_seed"])

    def test_stop_token_is_not_committed_and_remainder_discarded(self) -> None:
        decision = self._round([106, 11], [106, 11, 90], 2, 90)
        self.assertTrue(decision["valid"])
        self.assertTrue(decision["stopped"])
        self.assertEqual(decision["stop_token"], 106)
        self.assertEqual(decision["committed"], [])
        self.assertEqual(decision["discarded"], [11, 90])
        self.assertEqual(decision["next_position"], 5)

    def test_stop_token_in_the_bonus_slot_commits_the_prefix(self) -> None:
        decision = self._round([10, 11], [10, 11, 1], 2, 1)
        self.assertTrue(decision["stopped"])
        self.assertEqual(decision["stop_token"], 1)
        self.assertEqual(decision["committed"], [10, 11])
        self.assertEqual(decision["discarded"], [])

    def test_budget_truncation_discards_from_the_overflow_token(self) -> None:
        decision = self._round([10, 11], [10, 11, 90], 2, 90, token_budget=1)
        self.assertTrue(decision["valid"])
        self.assertFalse(decision["stopped"])
        self.assertEqual(decision["committed"], [10])
        self.assertEqual(decision["discarded"], [11, 90])
        decision = self._round([10, 11], [10, 11, 90], 2, 90, token_budget=2)
        self.assertEqual(decision["committed"], [10, 11])
        self.assertEqual(decision["discarded"], [90])
        decision = self._round([10, 11], [10, 11, 90], 2, 90, token_budget=3)
        self.assertEqual(decision["committed"], [10, 11, 90])
        self.assertEqual(decision["discarded"], [])

    def test_rejects_self_inconsistent_match_metadata(self) -> None:
        self.assertFalse(self._round([10, 11], [10, 91, 92], 2, 92)["valid"])
        self.assertFalse(self._round([10, 11], [10, 11, 90], 1, 11)["valid"])
        self.assertFalse(self._round([10, 11], [90, 91, 92], 1, 91)["valid"])
        self.assertFalse(self._round([10, 11], [10, 90, 92], 1, 92)["valid"])
        self.assertFalse(self._round([10, 11], [10, 11, 90], 2, 11)["valid"])

    def test_rejects_every_input_guard(self) -> None:
        good = ([10, 11], [10, 11, 90], 2, 90)
        self.assertTrue(self._round(*good)["valid"])
        self.assertFalse(self._round(*good, start_position=1)["valid"])
        self.assertFalse(self._round(*good, start_position=0)["valid"])
        self.assertFalse(self._round(*good, start_position=-1)["valid"])
        self.assertFalse(self._round(*good, token_budget=0)["valid"])
        self.assertFalse(self._round(*good, vocab_size=0)["valid"])
        self.assertFalse(self._round(*good, vocab_size=-1)["valid"])
        self.assertFalse(self._round([10, 11], [10, 11, 90], -1, 90)["valid"])
        self.assertFalse(self._round([10, 11], [10, 11, 90], 3, 90)["valid"])
        for probe in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(probe=probe):
                self.assertFalse(self._round(*good, state_probe=probe)["valid"])
        self.assertTrue(self._round(*good, start_position=2)["valid"])

    def test_rejects_out_of_range_tokens(self) -> None:
        self.assertFalse(self._round([-1, 11], [10, 11, 90], 0, 10)["valid"])
        self.assertFalse(self._round([K2_VOCAB_SIZE, 11], [10, 11, 90], 0, 10)["valid"])
        self.assertFalse(self._round([10, 11], [-1, 11, 90], 0, 90)["valid"])
        self.assertFalse(
            self._round([10, 11], [10, 11, K2_VOCAB_SIZE], 2, K2_VOCAB_SIZE)["valid"]
        )
        self.assertFalse(
            self._round([10, 11], [10, 11, 90], 2, 90, vocab_size=90)["valid"]
        )
        self.assertTrue(
            self._round([10, 11], [10, 11, 90], 2, 90, vocab_size=91)["valid"]
        )

    def test_rejected_decision_uses_documented_defaults(self) -> None:
        decision = self._round([10, 11], [10, 91, 92], 2, 92)
        self.assertEqual(
            decision,
            {
                "accepted_drafts": 0,
                "committed": [],
                "discarded": [],
                "next_position": -1,
                "next_seed": -1,
                "selected": [],
                "stop_token": -1,
                "stopped": False,
                "valid": False,
            },
        )

    def test_empty_stop_token_list_never_stops(self) -> None:
        decision = self._round([106, 1], [106, 1, 90], 2, 90, stop_tokens=[])
        self.assertTrue(decision["valid"])
        self.assertFalse(decision["stopped"])
        self.assertEqual(decision["stop_token"], -1)
        self.assertEqual(decision["committed"], [106, 1, 90])


GEMMA4_ANCHOR = "examples/models/gemma4/targets.bzl"
SPEC_RUNNER_HEADER = "examples/models/gemma4/runner/gemma4_spec_runner.h"
SOURCE_ROOT_ENV = "EXECUTORCH_SOURCE_ROOT"

INPUT_GUARD_ANCHOR = "start_position <"
SELF_CONSISTENCY_ANCHOR = "expected_matches"

VALID_ROUND: dict[str, Any] = {
    "candidates": [10, 11],
    "target_greedy": [10, 11, 90],
    "match_count": 2,
    "bonus": 90,
    "state_probe": 0.0,
    "start_position": 2,
    "token_budget": 3,
    "stop_tokens": STOP_TOKENS,
    "vocab_size": K2_VOCAB_SIZE,
}
INPUT_GUARD_PROBES: dict[str, dict[str, Any]] = {
    "start_position < 2": {"start_position": 1},
    "token_budget == 0": {"token_budget": 0},
    "vocab_size <= 0": {"vocab_size": 0},
    "match_count < 0": {"match_count": -1},
    "match_count > 2": {"match_count": 3},
    "!std::isfinite(state_probe)": {"state_probe": float("nan")},
}
SELF_CONSISTENCY_PROBES: dict[str, dict[str, Any]] = {
    "match_count != expected_matches": {"match_count": 1, "bonus": 11},
    "!valid_token(bonus)": {"bonus": -1},
    "bonus != target_greedy[match_count]": {"bonus": 91},
}

_ASSIGNMENT_PATTERN: re.Pattern[str] = re.compile(r"decision\.(\w+)\s*=\s*([^;]+);")
_GUARD_PATTERN: re.Pattern[str] = re.compile(r"if\s*\(([^;{}]*?)\)\s*\{", re.S)


def _root_candidates() -> list[tuple[str, Path | None]]:
    override = os.environ.get(SOURCE_ROOT_ENV)
    try:
        package = importlib.util.find_spec("executorch")
    except (ImportError, ValueError):
        package = None
    staged = list(package.submodule_search_locations or ()) if package else []
    here = Path(__file__).resolve()
    walked = next(
        (parent for parent in here.parents if (parent / GEMMA4_ANCHOR).is_file()), None
    )
    return [
        (f"${SOURCE_ROOT_ENV}", Path(override) if override else None),
        ("`executorch` package runfile", Path(staged[0]) if staged else None),
        (f"__file__ walk above {here}", walked),
    ]


def _source_root() -> Path:
    attempted: list[str] = []
    for strategy, candidate in _root_candidates():
        attempted.append(f"{strategy} -> {candidate}")
        if candidate is not None and (candidate / GEMMA4_ANCHOR).is_file():
            return candidate
    raise FileNotFoundError(
        f"no ExecuTorch source root containing {GEMMA4_ANCHOR}; "
        f"tried {'; '.join(attempted)}"
    )


def _read_spec_runner_header() -> str:
    path = _source_root() / SPEC_RUNNER_HEADER
    if not path.is_file():
        raise FileNotFoundError(
            f"missing D9 source under test: {path}; `reconcile_k2_round` cannot be "
            "cross-checked against `reconcile_gemma4_k2` until it lands"
        )
    return path.read_text(encoding="utf-8")


def _normalize(expression: str) -> str:
    return " ".join(expression.replace("output.", "").split())


def _cpp_assignment(header: str, field: str) -> str:
    found = [
        match.group(2)
        for match in _ASSIGNMENT_PATTERN.finditer(header)
        if match.group(1) == field and re.search(r"[A-Za-z_]", match.group(2))
    ]
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one computed `decision.{field} = ...;` in "
            f"{SPEC_RUNNER_HEADER}, found {len(found)}"
        )
    return _normalize(found[0])


def _cpp_guard_clauses(header: str, anchor: str) -> set[str]:
    found = [
        match.group(1)
        for match in _GUARD_PATTERN.finditer(header)
        if anchor in match.group(1)
    ]
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one `if` clause naming {anchor!r} in "
            f"{SPEC_RUNNER_HEADER}, found {len(found)}"
        )
    return {_normalize(clause) for clause in found[0].split("||")}


def _mirror_expression(field: str) -> str:
    tree = ast.parse(inspect.getsource(reconcile_k2_round))
    found = [
        ast.unparse(value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == field
        and any(isinstance(child, ast.Name) for child in ast.walk(value))
    ]
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one computed {field!r} entry in "
            f"reconcile_k2_round, found {len(found)}"
        )
    return found[0]


class K2MirrorMatchesSpecRunnerHeaderTest(unittest.TestCase):
    """Pins `reconcile_k2_round` to `reconcile_gemma4_k2` in the D9 header."""

    header: str

    def setUp(self) -> None:
        self.header = _read_spec_runner_header()

    def _assert_probes_are_rejected(
        self, probes: Mapping[str, Mapping[str, Any]]
    ) -> None:
        self.assertTrue(reconcile_k2_round(**VALID_ROUND)["valid"])
        for clause, probe in probes.items():
            with self.subTest(clause=clause):
                self.assertFalse(
                    reconcile_k2_round(**{**VALID_ROUND, **probe})["valid"]
                )

    def test_progression_and_seed_expressions_match_the_mirror(self) -> None:
        for field in ("next_position", "next_seed"):
            with self.subTest(field=field):
                self.assertEqual(
                    _cpp_assignment(self.header, field), _mirror_expression(field)
                )

    def test_every_input_guard_clause_is_enforced_by_the_mirror(self) -> None:
        self.assertEqual(
            _cpp_guard_clauses(self.header, INPUT_GUARD_ANCHOR),
            set(INPUT_GUARD_PROBES),
        )
        self._assert_probes_are_rejected(INPUT_GUARD_PROBES)

    def test_every_self_consistency_clause_is_enforced_by_the_mirror(self) -> None:
        self.assertEqual(
            _cpp_guard_clauses(self.header, SELF_CONSISTENCY_ANCHOR),
            set(SELF_CONSISTENCY_PROBES),
        )
        self._assert_probes_are_rejected(SELF_CONSISTENCY_PROBES)
