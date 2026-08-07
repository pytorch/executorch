# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import tempfile
import unittest

from pathlib import Path
from typing import Any
from unittest import mock

import torch


_PACKAGE = Path(__file__).parents[1]
_CONTRACT_MODULE = "executorch.examples.models.gemma4.target_prefill_contract"
_PRODUCER_MODULE = "executorch.examples.models.gemma4.generate_target_prefill_oracle"


def _modules() -> tuple[Any, Any]:
    return importlib.import_module(_CONTRACT_MODULE), importlib.import_module(
        _PRODUCER_MODULE
    )


def _tensor_envelope(shape: list[int], digest: str = "a" * 64) -> dict[str, object]:
    return {
        "byte_order": "little",
        "dtype": "float32",
        "layout": "row_major_contiguous",
        "sha256": digest,
        "shape": shape,
    }


def _arm_config(use_custom_sdpa: bool) -> dict[str, object]:
    return {
        "dtype": "float32",
        "enable_dynamic_shape": True,
        "group_size": 128,
        "max_seq_len": 8960,
        "text_quantize": "8da4w+emb4",
        "use_custom_sdpa": use_custom_sdpa,
        "use_kv_cache": True,
        "variant": "e2b",
    }


class _OffsetProjection(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1000.0


class _RecordingLmHead(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., :4]


class _RecordingTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state = 0
        self.calls: list[tuple[int, int, int, int, int]] = []
        self.model = torch.nn.Module()
        self.model.self_decoder = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.o_proj = _OffsetProjection()
        self.model.self_decoder.layers = torch.nn.ModuleList([layer])
        self.model.lm_head = _RecordingLmHead()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        _mask: object,
    ) -> torch.Tensor:
        start = int(input_pos[0].item())
        length = int(input_ids.shape[1])
        self.calls.append(
            (
                start,
                length,
                self.state,
                int(input_ids[0, 0].item()),
                int(input_ids[0, -1].item()),
            )
        )
        av = torch.full(
            (1, length, 8 * 256),
            float(start + 1 + 100 * self.state),
            dtype=torch.float32,
        )
        projected = self.model.self_decoder.layers[0].self_attn.o_proj(av)
        raw_logits = self.model.lm_head(projected)
        self.state += 1
        return raw_logits + 50.0


def _valid_receipt(
    contract: Any,
    checkpoint: dict[str, object],
    runtime_identity: dict[str, object],
    fbsource_commit: str,
) -> dict[str, object]:
    producer_path = contract.reviewed_producer_source_path()
    producer_digest = hashlib.sha256(producer_path.read_bytes()).hexdigest()
    contexts: dict[str, object] = {}
    for context in contract.TARGET_PREFILL_CONTEXTS:
        start, length = contract.final_chunk_range(context)
        contexts[str(context)] = {
            "arm_configs": {
                "custom_sdpa_fused": _arm_config(True),
                "manual_unfused": _arm_config(False),
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
                "custom_sdpa_fused": _tensor_envelope([1, length, 8, 256]),
                "manual_unfused": _tensor_envelope([1, length, 8, 256], "b" * 64),
            },
            "logits_post_softcap": _tensor_envelope([1, 1, 262144], "c" * 64),
            "logits_pre_softcap": _tensor_envelope([1, 1, 262144], "d" * 64),
            "prefill_token_post_softcap": 17,
            "prefill_token_raw": 17,
            "prompt_plan_sha256": contract.prompt_plan_sha256(context),
        }
    return {
        "authority": "target_only_eager",
        "checkpoint_acquisition": checkpoint,
        "contexts": contexts,
        "envelope_kind": "target_prefill_v2",
        "producer": {
            "fbsource_commit": fbsource_commit,
            "runtime_source_receipt": runtime_identity,
            "source_path": producer_path.name,
            "source_sha256": producer_digest,
        },
        "run": {
            "command": ["generate_target_prefill_oracle", "--contexts", "all"],
            "finished_at_utc": "2026-08-07T12:00:01Z",
            "host": "test-host",
            "started_at_utc": "2026-08-07T12:00:00Z",
        },
        "schema_version": 2,
    }


class TargetPrefillOracleConstructionTest(unittest.TestCase):
    def test_public_contract_is_declared(self) -> None:
        contract, producer = _modules()
        contract_names = (
            "TARGET_PREFILL_CONTEXTS",
            "canonical_json_bytes",
            "file_identity",
            "final_chunk_range",
            "prompt_plan_sha256",
            "reviewed_producer_source_path",
            "validate_target_prefill_receipt",
        )
        producer_names = (
            "_compare_av",
            "_reset_target_kv_caches",
            "_tensor_envelope",
            "chunk_ranges",
        )
        self.assertEqual(
            [name for name in contract_names if not hasattr(contract, name)], []
        )
        self.assertEqual(
            [name for name in producer_names if not hasattr(producer, name)], []
        )

    def test_contract_and_producer_sources_exist(self) -> None:
        self.assertTrue((_PACKAGE / "target_prefill_contract.py").is_file())
        self.assertTrue((_PACKAGE / "generate_target_prefill_oracle.py").is_file())

    def test_producer_is_independent_of_speculative_export(self) -> None:
        tree = ast.parse(
            (_PACKAGE / "generate_target_prefill_oracle.py").read_text(encoding="utf-8")
        )
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        self.assertFalse(any("export_speculative" in module for module in imports))

    def test_producer_has_no_pinned_sha256_literal(self) -> None:
        tree = ast.parse(
            (_PACKAGE / "generate_target_prefill_oracle.py").read_text(encoding="utf-8")
        )
        pinned = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) == 64
            and all(character in "0123456789abcdef" for character in node.value)
        ]
        self.assertEqual(pinned, [])


class TargetPrefillContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract, _ = _modules()
        self.checkpoint = {
            "files": {"model.safetensors": {"bytes": 3, "sha256": "e" * 64}},
            "repo_id": "example/model",
            "revision": "f" * 40,
        }
        self.runtime_identity = {"bytes": 17, "sha256": "1" * 64}
        self.fbsource_commit = "2" * 40

    def receipt(self) -> dict[str, object]:
        return _valid_receipt(
            self.contract,
            self.checkpoint,
            self.runtime_identity,
            self.fbsource_commit,
        )

    def validate(self, receipt: dict[str, object]) -> None:
        producer_path = self.contract.reviewed_producer_source_path()
        self.contract.validate_target_prefill_receipt(
            receipt,
            expected_checkpoint_acquisition=self.checkpoint,
            expected_producer_path=producer_path,
            expected_producer_sha256=hashlib.sha256(
                producer_path.read_bytes()
            ).hexdigest(),
            expected_runtime_source_identity=self.runtime_identity,
            expected_fbsource_commit=self.fbsource_commit,
        )

    def test_context_set_and_boundary_chunk_ranges_are_exact(self) -> None:
        self.assertEqual(
            self.contract.TARGET_PREFILL_CONTEXTS,
            (128, 511, 512, 513, 514, 1024, 2048, 4096, 4097, 8192),
        )
        self.assertEqual(self.contract.final_chunk_range(511), (0, 511))
        self.assertEqual(self.contract.final_chunk_range(512), (0, 512))
        self.assertEqual(self.contract.final_chunk_range(513), (512, 1))
        self.assertEqual(self.contract.final_chunk_range(4097), (4096, 1))

    def test_prompt_hash_uses_the_pinned_token_formula(self) -> None:
        context = 514
        tokens = [(index % (262144 - 1)) + 1 for index in range(context)]
        expected = hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(self.contract.prompt_plan_sha256(context), expected)

    def test_canonical_document_has_sorted_utf8_and_trailing_newline(self) -> None:
        self.assertEqual(
            self.contract.canonical_json_bytes({"z": "é", "a": 1}),
            '{\n  "a": 1,\n  "z": "é"\n}\n'.encode(),
        )

    def test_file_identity_hashes_the_actual_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(b"receipt\n")
            self.assertEqual(
                self.contract.file_identity(path),
                {
                    "bytes": 8,
                    "sha256": hashlib.sha256(b"receipt\n").hexdigest(),
                },
            )

    def test_valid_receipt_passes(self) -> None:
        self.validate(self.receipt())

    def test_missing_boundary_context_fails(self) -> None:
        receipt = self.receipt()
        contexts = receipt["contexts"]
        assert isinstance(contexts, dict)
        del contexts["513"]
        with self.assertRaisesRegex(ValueError, "exact contexts"):
            self.validate(receipt)

    def test_wrong_producer_bytes_fail(self) -> None:
        receipt = self.receipt()
        producer = receipt["producer"]
        assert isinstance(producer, dict)
        producer["source_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "producer source"):
            self.validate(receipt)

    def test_runtime_source_identity_and_head_are_bound(self) -> None:
        receipt = self.receipt()
        producer = receipt["producer"]
        assert isinstance(producer, dict)
        producer["runtime_source_receipt"] = {"bytes": 17, "sha256": "4" * 64}
        with self.assertRaisesRegex(ValueError, "runtime source"):
            self.validate(receipt)
        receipt = self.receipt()
        producer = receipt["producer"]
        assert isinstance(producer, dict)
        producer["fbsource_commit"] = "5" * 40
        with self.assertRaisesRegex(ValueError, "fbsource commit"):
            self.validate(receipt)

    def test_av_tolerance_pass_shape_and_finite_metrics_are_enforced(self) -> None:
        mutations: list[tuple[str, object]] = [
            ("atol", 1e-3),
            ("rtol", 1e-2),
            ("passed", False),
            ("max_abs", math.inf),
            ("rel_rms", math.nan),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                receipt = self.receipt()
                context = receipt["contexts"]["513"]  # type: ignore[index]
                witness = context[  # type: ignore[index]
                    "layer0_manual_unfused_vs_custom_sdpa_fused"
                ]
                witness["agreement"][key] = value  # type: ignore[index]
                with self.assertRaises(ValueError):
                    self.validate(receipt)
        receipt = self.receipt()
        context = receipt["contexts"]["513"]  # type: ignore[index]
        witness = context[  # type: ignore[index]
            "layer0_manual_unfused_vs_custom_sdpa_fused"
        ]
        witness["custom_sdpa_fused"]["shape"] = [1, 2, 8, 256]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "AV tensor"):
            self.validate(receipt)

    def test_arm_configs_may_differ_only_in_custom_sdpa(self) -> None:
        receipt = self.receipt()
        context = receipt["contexts"]["128"]  # type: ignore[index]
        context["arm_configs"]["manual_unfused"]["max_seq_len"] = 1024  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "arm configurations"):
            self.validate(receipt)

    def test_raw_and_post_softcap_tokens_must_match(self) -> None:
        receipt = self.receipt()
        context = receipt["contexts"]["128"]  # type: ignore[index]
        context["prefill_token_post_softcap"] = 18  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "softcap"):
            self.validate(receipt)

    def test_boolean_counts_and_tokens_are_not_integers(self) -> None:
        receipt = self.receipt()
        context = receipt["contexts"]["128"]  # type: ignore[index]
        context["cache_reset_counts"]["custom_sdpa_fused"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "reset counts"):
            self.validate(receipt)

        receipt = self.receipt()
        context = receipt["contexts"]["128"]  # type: ignore[index]
        context["prefill_token_raw"] = True  # type: ignore[index]
        context["prefill_token_post_softcap"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "raw token"):
            self.validate(receipt)

    def test_unqualified_logits_digest_is_rejected(self) -> None:
        receipt = self.receipt()
        context = receipt["contexts"]["128"]  # type: ignore[index]
        context["logits_sha256"] = "6" * 64  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            self.validate(receipt)


class TargetPrefillProducerHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        _, self.producer = _modules()

    def _run_recording_arm(
        self, contexts: tuple[int, ...]
    ) -> tuple[
        _RecordingTarget,
        list[int],
        dict[int, dict[str, object]],
    ]:
        model = _RecordingTarget()
        reset_states: list[int] = []

        def reset(target: torch.nn.Module) -> int:
            if target is not model:
                raise AssertionError("reset received the wrong target")
            reset_states.append(model.state)
            model.state = 0
            return 7

        with mock.patch.object(
            self.producer, "_load_target", return_value=model
        ), mock.patch.object(
            self.producer, "_reset_target_kv_caches", side_effect=reset
        ), mock.patch.object(
            self.producer, "TARGET_PREFILL_CONTEXTS", contexts
        ):
            results = self.producer._run_arm(
                Path("model.safetensors"), use_custom_sdpa=True
            )
        return model, reset_states, results

    def test_run_arm_chunks_calls_and_resets_state_between_contexts(self) -> None:
        forward_model, forward_resets, forward = self._run_recording_arm((511, 513))
        reverse_model, reverse_resets, reverse = self._run_recording_arm((513, 511))

        self.assertEqual(forward_resets, [0, 1])
        self.assertEqual(reverse_resets, [0, 2])
        self.assertEqual(
            forward_model.calls,
            [
                (0, 511, 0, 1, 511),
                (0, 512, 0, 1, 512),
                (512, 1, 1, 513, 513),
            ],
        )
        self.assertEqual(
            reverse_model.calls,
            [
                (0, 512, 0, 1, 512),
                (512, 1, 1, 513, 513),
                (0, 511, 0, 1, 511),
            ],
        )
        for context in (511, 513):
            with self.subTest(context=context):
                self.assertEqual(forward[context]["cache_reset_count"], 7)
                self.assertEqual(reverse[context]["cache_reset_count"], 7)
                self.assertEqual(
                    forward[context]["final_chunk_start"],
                    self.producer.final_chunk_range(context)[0],
                )
                self.assertEqual(
                    forward[context]["final_chunk_length"],
                    self.producer.final_chunk_range(context)[1],
                )
                for key in ("av", "logits_pre_softcap", "logits_post_softcap"):
                    self.assertTrue(
                        torch.equal(forward[context][key], reverse[context][key])
                    )

    def test_generate_receipt_passes_checkpoint_directory_to_weight_loader(
        self,
    ) -> None:
        contract, _ = _modules()
        manifest = importlib.import_module(
            "executorch.examples.models.gemma4.webgpu_artifact_manifest"
        )
        convert_weights = importlib.import_module(
            "executorch.examples.models.gemma4.text_decoder.convert_weights"
        )
        config_module = importlib.import_module(
            "executorch.examples.models.gemma4.text_decoder.gemma4_config"
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / "checkpoint"
            checkpoint_root.mkdir()
            (checkpoint_root / "model.safetensors").write_bytes(b"fixture")
            runtime_receipt = Path(directory) / "runtime-source.json"
            runtime_receipt.write_text(
                json.dumps({"fbsource_commit": "2" * 40}), encoding="utf-8"
            )

            logits = torch.zeros(
                (1, 1, contract.TARGET_PREFILL_VOCAB_SIZE),
                dtype=torch.float32,
            )

            def run_loader_boundary(
                checkpoint: Path, *, use_custom_sdpa: bool
            ) -> dict[int, dict[str, object]]:
                config = config_module.Gemma4Config.from_config("e2b")
                convert_weights.convert_hf_to_custom(str(checkpoint), config)
                return {
                    context: {
                        "av": torch.zeros(
                            (1, self.producer.final_chunk_range(context)[1], 8, 256),
                            dtype=torch.float32,
                        ),
                        "cache_reset_count": 1,
                        "config": _arm_config(use_custom_sdpa),
                        "final_chunk_length": self.producer.final_chunk_range(context)[
                            1
                        ],
                        "final_chunk_start": self.producer.final_chunk_range(context)[
                            0
                        ],
                        "logits_post_softcap": logits,
                        "logits_pre_softcap": logits,
                    }
                    for context in self.producer.TARGET_PREFILL_CONTEXTS
                }

            safe_open = mock.MagicMock()
            safe_open.return_value.__enter__.return_value.keys.return_value = ()
            with mock.patch.object(
                manifest,
                "validate_export_identity",
                return_value=manifest.CHECKPOINT_ACQUISITION,
            ), mock.patch.object(
                self.producer, "_run_arm", side_effect=run_loader_boundary
            ), mock.patch(
                "safetensors.safe_open", safe_open
            ):
                receipt = self.producer.generate_target_prefill_receipt(
                    checkpoint_root,
                    runtime_receipt,
                    command=["generate_target_prefill_oracle"],
                )

        self.assertEqual(
            receipt["checkpoint_acquisition"], manifest.CHECKPOINT_ACQUISITION
        )

    def test_run_arm_uses_preprojection_av_and_presoftcap_logits(self) -> None:
        _, _, results = self._run_recording_arm((513,))
        result = results[513]
        av = result["av"]
        raw_logits = result["logits_pre_softcap"]
        post_logits = result["logits_post_softcap"]

        self.assertIsInstance(av, torch.Tensor)
        self.assertIsInstance(raw_logits, torch.Tensor)
        self.assertIsInstance(post_logits, torch.Tensor)
        self.assertTrue(torch.equal(av, torch.full((1, 1, 8, 256), 613.0)))
        self.assertTrue(torch.equal(raw_logits, torch.full((1, 1, 4), 1613.0)))
        self.assertTrue(torch.equal(post_logits, torch.full((1, 1, 4), 1663.0)))

    def test_chunk_ranges_never_exceed_512(self) -> None:
        for context in (511, 512, 513, 4096, 4097, 8192):
            ranges = self.producer.chunk_ranges(context)
            self.assertEqual(ranges[-1], self.producer.final_chunk_range(context))
            self.assertTrue(all(length <= 512 for _, length in ranges))
            self.assertEqual(sum(length for _, length in ranges), context)

    def test_tensor_envelope_canonicalizes_noncontiguous_tensors(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).view(3, 4).t()
        envelope = self.producer._tensor_envelope(tensor)
        expected_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
        self.assertEqual(envelope["shape"], [4, 3])
        self.assertEqual(envelope["dtype"], "float32")
        self.assertEqual(envelope["sha256"], hashlib.sha256(expected_bytes).hexdigest())

    def test_av_comparison_uses_tolerance_not_hash_equality(self) -> None:
        fused = torch.ones((1, 2, 8, 256), dtype=torch.float32)
        close = fused + 1e-5
        agreement = self.producer._compare_av(fused, close)
        self.assertTrue(agreement["passed"])
        self.assertNotEqual(
            self.producer._tensor_envelope(fused)["sha256"],
            self.producer._tensor_envelope(close)["sha256"],
        )
        with self.assertRaises(AssertionError):
            self.producer._compare_av(fused, fused + 0.1)

    def test_reset_zeros_every_real_kv_cache(self) -> None:
        from executorch.examples.models.gemma4.text_decoder.gemma4_attention import (
            Gemma4KVCache,
        )

        def cache() -> Any:
            value = Gemma4KVCache.__new__(Gemma4KVCache)
            torch.nn.Module.__init__(value)
            value.register_buffer("k_cache", torch.ones((1, 2, 1, 2)))
            value.register_buffer("v_cache", torch.ones((1, 2, 1, 2)) * 2)
            return value

        model = torch.nn.Sequential(cache(), cache())
        self.assertEqual(self.producer._reset_target_kv_caches(model), 2)
        for module in model.modules():
            if isinstance(module, Gemma4KVCache):
                self.assertEqual(torch.count_nonzero(module.k_cache).item(), 0)
                self.assertEqual(torch.count_nonzero(module.v_cache).item(), 0)

    def test_reset_fails_when_model_has_no_kv_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "no Gemma4 KV caches"):
            self.producer._reset_target_kv_caches(torch.nn.Linear(2, 2))

    def test_capture_modules_are_runtime_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "text model"):
            self.producer._resolve_capture_modules(torch.nn.Module())

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.self_decoder = torch.nn.Module()
        model.model.self_decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = model.model.self_decoder.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.o_proj = torch.nn.Linear(2, 2)
        model.model.lm_head = torch.nn.Linear(2, 2)
        o_proj, lm_head = self.producer._resolve_capture_modules(model)
        self.assertIs(o_proj, layer.self_attn.o_proj)
        self.assertIs(lm_head, model.model.lm_head)


if __name__ == "__main__":
    unittest.main()
