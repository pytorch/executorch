# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkStorageType
from executorch.examples.models.voxtral_realtime.export_voxtral_rt import (
    VULKAN_EXTERNAL_CONSTANTS_MAX_DATA_BYTES,
    _requires_explicit_output_weight_clone,
    audit_vulkan_delegation,
    lower_to_executorch,
    validate_vulkan_options,
)
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops


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


if __name__ == "__main__":
    unittest.main()
