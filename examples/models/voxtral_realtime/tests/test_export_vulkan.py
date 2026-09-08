# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import call, MagicMock, patch

import torch
from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkStorageType
from executorch.examples.models.voxtral_realtime.export_voxtral_rt import (
    _requires_explicit_output_weight_clone,
    audit_vulkan_delegation,
    export_streaming,
    lower_to_executorch,
    TextDecoderExport,
    TokenEmbeddingExport,
    validate_vulkan_options,
    VULKAN_EXTERNAL_CONSTANTS_MAX_DATA_BYTES,
)
from executorch.examples.models.voxtral_realtime.model import (
    compute_time_embedding,
    StreamingAudioEncoderExport,
    VoxtralRealtimeConfig,
    VoxtralRealtimeModel,
)
from executorch.exir._serialize.data_serializer import DataPayload
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.pybindings import portable_lib


def make_node(op, name, target, val=None, layout=None):
    spec = SimpleNamespace()
    if layout is not None:
        spec.etvk_node_repr = layout
    return SimpleNamespace(
        op=op,
        name=name,
        target=target,
        meta={"val": val, "spec": spec},
    )


def make_edge(nodes, backend_id="VulkanBackend"):
    graph_module = SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
    graph_module.lowered_module_0 = SimpleNamespace(backend_id=backend_id)
    exported_program = SimpleNamespace(graph_module=graph_module)
    return SimpleNamespace(
        methods=["method"],
        exported_program=lambda _: exported_program,
    )


class TestVulkanDelegationAudit(unittest.TestCase):
    def test_accepts_vulkan_delegate_and_dynamic_constructors(self):
        nodes = [
            make_node("get_attr", "lowered_module_0", "lowered_module_0"),
            make_node("call_function", "delegate", executorch_call_delegate),
            make_node("call_function", "getitem", operator.getitem),
            make_node(
                "call_function",
                "arange",
                exir_ops.edge.aten.arange.start_step,
            ),
            make_node("call_function", "full", exir_ops.edge.aten.full.default),
            make_node(
                "call_function",
                "sym_size",
                torch.ops.aten.sym_size.int,
            ),
        ]

        audit_vulkan_delegation(make_edge(nodes))

    def test_rejects_portable_tensor_compute_with_diagnostics(self):
        tensor = SimpleNamespace(dtype=torch.float32, shape=torch.Size([2, 3]))
        nodes = [
            make_node("get_attr", "lowered_module_0", "lowered_module_0"),
            make_node(
                "call_function",
                "add",
                exir_ops.edge.aten.add.Tensor,
                tensor,
                "texture3d/channels",
            ),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            r"method=method.*aten.add.Tensor.*torch.float32.*\[2, 3\].*texture3d/channels",
        ):
            audit_vulkan_delegation(make_edge(nodes))

    def test_rejects_non_vulkan_delegate(self):
        nodes = [make_node("get_attr", "lowered_module_0", "lowered_module_0")]

        with self.assertRaisesRegex(RuntimeError, "delegated to XnnpackBackend"):
            audit_vulkan_delegation(make_edge(nodes, "XnnpackBackend"))


class TestVulkanExportOptions(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "backend": "vulkan",
            "dtype": "fp32",
            "qlinear": None,
            "qlinear_encoder": None,
            "qembedding": None,
            "qlinear_packing_format": None,
            "qlinear_encoder_packing_format": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_accepts_fp32_and_vulkan_quantization(self):
        parser = MagicMock()
        validate_vulkan_options(
            self.make_args(
                qlinear="8da4w",
                qlinear_encoder="8da4w",
                qembedding="4w",
            ),
            parser,
        )
        parser.error.assert_not_called()

    def test_vulkan_quantization_uses_torchao_to_split_tied_weight(self):
        self.assertFalse(
            _requires_explicit_output_weight_clone("vulkan", "8da4w", "4w")
        )
        self.assertTrue(
            _requires_explicit_output_weight_clone("xnnpack", "8da4w", "4w")
        )

    def test_torchao_split_matches_explicit_clone(self):
        from executorch.extension.llm.export.quantize import quantize_model_

        class TiedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 32)
                self.output = torch.nn.Linear(32, 64, bias=False)
                self.output.weight = self.embedding.weight

        torch.manual_seed(20260814)
        tied = TiedModule().eval()
        cloned = copy.deepcopy(tied)
        cloned.output.weight = torch.nn.Parameter(cloned.embedding.weight.clone())
        original_embedding = tied.embedding.weight.detach().clone()

        for module in (tied, cloned):
            quantize_model_(
                module,
                qlinear_config="8da4w",
                qlinear_group_size=32,
            )

        self.assertIsNot(tied.output.weight, tied.embedding.weight)
        self.assertTrue(torch.equal(tied.embedding.weight, original_embedding))
        self.assertIs(type(tied.output.weight), type(cloned.output.weight))
        self.assertEqual(tied.output.weight.shape, cloned.output.weight.shape)
        self.assertEqual(tied.output.weight.block_size, cloned.output.weight.block_size)
        self.assertTrue(
            torch.equal(tied.output.weight.scale, cloned.output.weight.scale)
        )
        self.assertTrue(
            torch.equal(tied.output.weight.zero_point, cloned.output.weight.zero_point)
        )
        self.assertTrue(
            torch.equal(
                tied.output.weight.dequantize(), cloned.output.weight.dequantize()
            )
        )

        for module in (tied, cloned):
            quantize_model_(
                module,
                qembedding_config="4w",
                qembedding_group_size=32,
            )

        self.assertIs(type(tied.embedding.weight), type(cloned.embedding.weight))
        self.assertEqual(tied.embedding.weight.shape, cloned.embedding.weight.shape)
        self.assertEqual(
            tied.embedding.weight.block_size, cloned.embedding.weight.block_size
        )
        self.assertTrue(
            torch.equal(tied.embedding.weight.scale, cloned.embedding.weight.scale)
        )
        self.assertTrue(
            torch.equal(
                tied.embedding.weight.zero_point, cloned.embedding.weight.zero_point
            )
        )
        self.assertTrue(
            torch.equal(
                tied.embedding.weight.dequantize(),
                cloned.embedding.weight.dequantize(),
            )
        )

    def test_rejects_incompatible_dtype_quantization_and_packing(self):
        def parser_error(message):
            raise ValueError(message)

        for overrides, message in (
            ({"dtype": "bf16"}, "requires --dtype=fp32"),
            ({"qlinear": "4w"}, "--qlinear=4w"),
            ({"qlinear_encoder": "8w"}, "--qlinear-encoder=8w"),
            ({"qembedding": "8w"}, "--qembedding=8w"),
            (
                {"qlinear_packing_format": "tile_packed_to_4d"},
                "--qlinear-packing-format",
            ),
        ):
            with self.subTest(overrides=overrides):
                parser = MagicMock()
                parser.error.side_effect = parser_error
                with self.assertRaisesRegex(ValueError, message):
                    validate_vulkan_options(self.make_args(**overrides), parser)

    @patch(
        "executorch.backends.vulkan.partitioner.vulkan_partitioner.VulkanPartitioner"
    )
    @patch(
        "executorch.examples.models.voxtral_realtime.export_voxtral_rt.audit_vulkan_delegation"
    )
    @patch(
        "executorch.examples.models.voxtral_realtime.export_voxtral_rt.to_edge_transform_and_lower"
    )
    def test_vulkan_lowering_is_method_scoped(
        self,
        lower_mock,
        audit_mock,
        partitioner_mock,
    ):
        edge = MagicMock()
        lower_mock.return_value = edge
        programs = {
            "encode_audio_chunk": MagicMock(),
            "text_decoder": MagicMock(),
            "token_embedding": MagicMock(),
        }
        metadata = {"vocab_size": 131072, "dim": 3072}

        lower_to_executorch(programs, metadata, backend="vulkan")

        self.assertEqual(
            partitioner_mock.call_args_list,
            [
                call(
                    compile_options={
                        "require_dynamic_shapes": True,
                        "external_constants_max_data_bytes": (
                            VULKAN_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
                        ),
                        "alias_buffer_mutations": True,
                    }
                ),
                call(
                    compile_options={
                        "require_dynamic_shapes": True,
                        "external_constants_max_data_bytes": (
                            VULKAN_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
                        ),
                        "alias_buffer_mutations": True,
                    }
                ),
                call(
                    compile_options={
                        "require_dynamic_shapes": True,
                        "external_constants_max_data_bytes": (
                            VULKAN_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
                        ),
                        "buffer_limit": 402653184,
                        "storage_type_override": VkStorageType.BUFFER,
                    }
                ),
            ],
        )
        audit_mock.assert_called_once_with(edge)
        edge.to_executorch.assert_called_once()

    @patch(
        "executorch.backends.vulkan.partitioner.vulkan_partitioner.VulkanPartitioner"
    )
    @patch(
        "executorch.examples.models.voxtral_realtime.export_voxtral_rt.audit_vulkan_delegation"
    )
    @patch(
        "executorch.examples.models.voxtral_realtime.export_voxtral_rt.to_edge_transform_and_lower"
    )
    def test_vulkan_force_fp16_is_applied_to_every_method(
        self,
        lower_mock,
        audit_mock,
        partitioner_mock,
    ):
        edge = MagicMock()
        lower_mock.return_value = edge
        programs = {
            "encode_audio_chunk": MagicMock(),
            "text_decoder": MagicMock(),
            "token_embedding": MagicMock(),
        }
        metadata = {"vocab_size": 131072, "dim": 3072}

        lower_to_executorch(
            programs,
            metadata,
            backend="vulkan",
            vulkan_force_fp16=True,
        )

        self.assertEqual(len(partitioner_mock.call_args_list), 3)
        for partitioner_call in partitioner_mock.call_args_list:
            self.assertTrue(partitioner_call.kwargs["compile_options"]["force_fp16"])
        audit_mock.assert_called_once_with(edge)
        edge.to_executorch.assert_called_once()


class TestStreamingEncoderBatching(unittest.TestCase):
    def test_two_chunk_export_and_runtime(self):
        config = VoxtralRealtimeConfig(
            dim=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            hidden_dim=64,
            vocab_size=64,
            ada_rms_norm_t_cond_dim=8,
            enc_dim=32,
            enc_n_layers=1,
            enc_n_heads=4,
            enc_head_dim=8,
            enc_hidden_dim=64,
            num_mel_bins=8,
            downsample_factor=4,
            max_seq_len=8,
            sliding_window=8,
            streaming=True,
            backend="vulkan",
        )
        model = VoxtralRealtimeModel(config).eval()
        model.register_buffer("t_cond", compute_time_embedding(1, config.dim))
        reference_encoder = StreamingAudioEncoderExport(
            copy.deepcopy(model), max_enc_len=10
        ).eval()

        programs, metadata, _ = export_streaming(
            model,
            max_seq_len=config.max_seq_len,
            max_enc_len=10,
            backend="vulkan",
            encoder_batch_chunks=2,
        )

        self.assertEqual(metadata["chunk_mel_len"], 8)
        self.assertEqual(metadata["encoder_batch_chunks"], 2)
        encoder_targets = {
            node.target for node in programs["encode_audio_chunk"].graph.nodes
        }
        self.assertIn(torch.ops.et_vk.ring_sdpa.default, encoder_targets)

        et_program = lower_to_executorch(
            {"encode_audio_chunk": programs["encode_audio_chunk"]},
            metadata,
            backend="vulkan",
        )
        buffers = []
        named_data = {}
        for data_file in et_program._tensor_data.values():
            payload = et_program._data_serializer.deserialize(data_file)
            buffer_index_offset = len(buffers)
            buffers.extend(payload.buffers)
            for name, entry in payload.named_data.items():
                self.assertNotIn(name, named_data)
                named_data[name] = replace(
                    entry,
                    buffer_index=entry.buffer_index + buffer_index_offset,
                )
        data_buffer = bytes(
            et_program._data_serializer.serialize(
                DataPayload(buffers=buffers, named_data=named_data)
            )
        )
        module = portable_lib._load_for_executorch_from_buffer(
            et_program.buffer, data_buffer
        )

        mel_0 = torch.randn(1, config.num_mel_bins, 16)
        mel_1 = torch.randn(1, config.num_mel_bins, 16)
        mel_wrap = torch.randn(1, config.num_mel_bins, 16)
        positions = (torch.arange(8), torch.arange(8, 16), torch.arange(16, 24))
        with torch.inference_mode():
            expected = (
                reference_encoder(mel_0, positions[0]),
                reference_encoder(mel_1, positions[1]),
                reference_encoder(mel_wrap, positions[2]),
            )
            actual = (
                module.run_method("encode_audio_chunk", (mel_0, positions[0]))[0],
                module.run_method("encode_audio_chunk", (mel_1, positions[1]))[0],
                module.run_method("encode_audio_chunk", (mel_wrap, positions[2]))[0],
            )

        for actual_output, expected_output in zip(actual, expected):
            torch.testing.assert_close(
                actual_output, expected_output, atol=1e-3, rtol=1e-3
            )


class TestTinyStreamingVulkan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20260816)
        config = VoxtralRealtimeConfig(
            dim=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            hidden_dim=64,
            vocab_size=64,
            ada_rms_norm_t_cond_dim=8,
            enc_dim=32,
            enc_n_layers=1,
            enc_n_heads=4,
            enc_head_dim=8,
            enc_hidden_dim=64,
            num_mel_bins=8,
            downsample_factor=4,
            max_seq_len=8,
            sliding_window=4,
            streaming=True,
            backend="vulkan",
        )
        model = VoxtralRealtimeModel(config).eval()
        model.register_buffer("t_cond", compute_time_embedding(1, config.dim))
        cls.reference_model = copy.deepcopy(model)

        programs, metadata, _ = export_streaming(
            model,
            max_seq_len=config.max_seq_len,
            max_enc_len=4,
            backend="vulkan",
        )
        for method_name in ("encode_audio_chunk", "text_decoder"):
            targets = {node.target for node in programs[method_name].graph.nodes}
            if torch.ops.et_vk.ring_sdpa.default not in targets:
                raise RuntimeError(
                    f"Expected {method_name} to preserve et_vk.ring_sdpa"
                )
        cls.et_program = lower_to_executorch(
            programs,
            metadata,
            backend="vulkan",
        )
        data_files = cls.et_program._tensor_data
        if not data_files:
            raise RuntimeError("Expected external Vulkan constants for the tiny model")
        # The Python binding accepts one data map; the C++ runner covers PTD vectors.
        buffers = []
        named_data = {}
        for data_file in data_files.values():
            payload = cls.et_program._data_serializer.deserialize(data_file)
            buffer_index_offset = len(buffers)
            buffers.extend(payload.buffers)
            for name, entry in payload.named_data.items():
                if name in named_data:
                    raise RuntimeError(f"Duplicate external constant: {name}")
                named_data[name] = replace(
                    entry,
                    buffer_index=entry.buffer_index + buffer_index_offset,
                )
        cls.data_buffer = bytes(
            cls.et_program._data_serializer.serialize(
                DataPayload(buffers=buffers, named_data=named_data)
            )
        )

    def test_runtime_lifecycle_and_fresh_state(self):
        module = portable_lib._load_for_executorch_from_buffer(
            self.et_program.buffer, self.data_buffer
        )
        reference_encoder = StreamingAudioEncoderExport(
            self.reference_model, max_enc_len=4
        ).eval()
        reference_decoder = TextDecoderExport(self.reference_model).eval()
        reference_embedding = TokenEmbeddingExport(self.reference_model).eval()

        mel_0 = torch.randn(1, 8, 8)
        mel_1 = torch.randn(1, 8, 8)
        mel_wrap = torch.randn(1, 8, 8)
        enc_pos_0 = torch.arange(4)
        enc_pos_1 = torch.arange(4, 8)
        enc_pos_wrap = torch.arange(8, 12)
        decoder_input_0 = torch.randn(1, 1, 32)
        decoder_input_1 = torch.randn(1, 1, 32)
        decoder_input_wrap = torch.randn(1, 1, 32)
        decoder_pos_0 = torch.tensor([0])
        decoder_pos_1 = torch.tensor([1])
        decoder_pos_wrap = torch.tensor([8])
        token_ids = torch.tensor([[1, 2, 3, 4]])

        with torch.inference_mode():
            expected_encoder_0 = reference_encoder(mel_0, enc_pos_0)
            expected_encoder_1 = reference_encoder(mel_1, enc_pos_1)
            expected_encoder_wrap = reference_encoder(mel_wrap, enc_pos_wrap)
            expected_encoder_reset = reference_encoder(mel_0, enc_pos_0)
            expected_decoder_0 = reference_decoder(decoder_input_0, decoder_pos_0)
            expected_decoder_1 = reference_decoder(decoder_input_1, decoder_pos_1)
            expected_decoder_wrap = reference_decoder(
                decoder_input_wrap, decoder_pos_wrap
            )
            expected_decoder_reset = reference_decoder(decoder_input_0, decoder_pos_0)
            expected_embedding = reference_embedding(token_ids)

            actual_encoder_0 = module.run_method(
                "encode_audio_chunk", (mel_0, enc_pos_0)
            )[0]
            actual_encoder_1 = module.run_method(
                "encode_audio_chunk", (mel_1, enc_pos_1)
            )[0]
            actual_encoder_wrap = module.run_method(
                "encode_audio_chunk", (mel_wrap, enc_pos_wrap)
            )[0]
            actual_encoder_reset = module.run_method(
                "encode_audio_chunk", (mel_0, enc_pos_0)
            )[0]
            actual_decoder_0 = module.run_method(
                "text_decoder", (decoder_input_0, decoder_pos_0)
            )[0]
            actual_decoder_1 = module.run_method(
                "text_decoder", (decoder_input_1, decoder_pos_1)
            )[0]
            actual_decoder_wrap = module.run_method(
                "text_decoder", (decoder_input_wrap, decoder_pos_wrap)
            )[0]
            actual_decoder_reset = module.run_method(
                "text_decoder", (decoder_input_0, decoder_pos_0)
            )[0]
            actual_embedding = module.run_method("token_embedding", (token_ids,))[0]

        torch.testing.assert_close(
            actual_encoder_0, expected_encoder_0, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            actual_encoder_1, expected_encoder_1, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            actual_encoder_wrap, expected_encoder_wrap, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            actual_encoder_reset,
            expected_encoder_reset,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            actual_decoder_0, expected_decoder_0, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            actual_decoder_1, expected_decoder_1, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            actual_decoder_wrap, expected_decoder_wrap, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            actual_decoder_reset,
            expected_decoder_reset,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(actual_embedding, expected_embedding)
        self.assertTrue(torch.equal(actual_encoder_0, actual_encoder_reset))
        self.assertTrue(torch.equal(actual_decoder_0, actual_decoder_reset))

        fresh_module = portable_lib._load_for_executorch_from_buffer(
            self.et_program.buffer, self.data_buffer
        )
        with torch.inference_mode():
            fresh_encoder = fresh_module.run_method(
                "encode_audio_chunk", (mel_0, enc_pos_0)
            )[0]
            fresh_decoder = fresh_module.run_method(
                "text_decoder", (decoder_input_0, decoder_pos_0)
            )[0]
            fresh_embedding = fresh_module.run_method("token_embedding", (token_ids,))[
                0
            ]

        self.assertTrue(torch.equal(actual_encoder_0, fresh_encoder))
        self.assertTrue(torch.equal(actual_decoder_0, fresh_decoder))
        self.assertTrue(torch.equal(actual_embedding, fresh_embedding))


if __name__ == "__main__":
    unittest.main()
