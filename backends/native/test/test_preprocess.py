# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import (
    EXTERNAL_CONSTANTS_TAG_KEY,
    NativePartitioner,
    PTN_SERIALIZATION_KEY,
)
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.preprocess import (
    _parse_compile_specs,
    NativeBackend,
    NativeDelegateInfo,
)
from executorch.backends.native.serialization import (
    deserialize_graph,
    deserialize_program,
)
from executorch.backends.native.serialization.schema import OpKind, OutputKind
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec


def _lower(model, example_inputs):
    ep = torch.export.export(model, example_inputs)
    return to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[NativePartitioner()],
        compile_config=get_default_compile_config(),
    )


def _get_delegate_blob(edge):
    """Extract the single delegate's processed bytes from the lowered program."""
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1, f"Expected 1 delegate blob, got {len(delegates)}"
    return bytes(delegates[0].data)


def _call_function_targets(graph):
    return [n.target for n in graph.nodes if n.op_kind == OpKind.CALL_FUNCTION]


class CompileSpecParsingTest(unittest.TestCase):
    def test_ptn_flag_is_parsed(self):
        self.assertEqual(
            _parse_compile_specs([CompileSpec(PTN_SERIALIZATION_KEY, b"1")]),
            (None, True),
        )

    def test_ptn_flag_requires_canonical_value(self):
        with self.assertRaisesRegex(ValueError, "must have value"):
            _parse_compile_specs([CompileSpec(PTN_SERIALIZATION_KEY, b"0")])

    def test_conflicting_constant_channels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _parse_compile_specs(
                [
                    CompileSpec(EXTERNAL_CONSTANTS_TAG_KEY, b"weights"),
                    CompileSpec(PTN_SERIALIZATION_KEY, b"1"),
                ]
            )

    def test_duplicate_recognized_spec_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate compile spec"):
            _parse_compile_specs(
                [
                    CompileSpec(EXTERNAL_CONSTANTS_TAG_KEY, b"first"),
                    CompileSpec(EXTERNAL_CONSTANTS_TAG_KEY, b"second"),
                ]
            )


class PtnConstantHandoffTest(unittest.TestCase):
    def test_preserves_layout_and_storage_for_package_validation(self):
        base = torch.arange(6).view(2, 3)
        view = base.t()
        edge_program = SimpleNamespace(
            graph_module=object(),
            graph_signature=object(),
            state_dict={},
            constants={},
        )

        with patch(
            "executorch.backends.native.preprocess.serialize_graph",
            return_value=(b"NPTG", {"view": view}),
        ):
            result = NativeBackend.preprocess(
                edge_program,
                [CompileSpec(PTN_SERIALIZATION_KEY, b"1")],
            )

        info = result._delegate_info_meta
        self.assertIsInstance(info, NativeDelegateInfo)
        captured = info.constants["view"]
        self.assertFalse(captured.is_contiguous())
        self.assertEqual(captured.stride(), view.stride())
        self.assertEqual(
            captured.untyped_storage().data_ptr(), view.untyped_storage().data_ptr()
        )


class PreprocessSerializationTest(unittest.TestCase):
    def test_payload_has_native_file_identifier(self):
        blob = _get_delegate_blob(_lower(nn.Linear(4, 4), (torch.randn(1, 4),)))
        self.assertEqual(blob[4:8], b"NPTG")

    def test_linear_op_roundtrips(self):
        blob = _get_delegate_blob(_lower(nn.Linear(4, 4), (torch.randn(1, 4),)))
        graph = deserialize_graph(blob)
        targets = _call_function_targets(graph)
        self.assertTrue(
            any(t is not None and "linear" in t for t in targets),
            f"expected linear op, got {targets}",
        )

    def test_add_op_roundtrips(self):
        class AddModel(nn.Module):
            def forward(self, x, y):
                return x + y

        blob = _get_delegate_blob(
            _lower(AddModel(), (torch.randn(2, 3), torch.randn(2, 3)))
        )
        graph = deserialize_graph(blob)
        targets = _call_function_targets(graph)
        self.assertTrue(any(t is not None and "add" in t for t in targets))

    def test_non_persistent_buffer_recorded_as_mutable_buffer(self):
        # A KV-cache-style non-persistent buffer, mutated in place, must survive
        # the full .pte lowering route as a mutable buffer (no shipped data), with
        # its mutation captured as a BUFFER_MUTATION output writeback.
        class KVCacheModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(4), persistent=False)

            def forward(self, x):
                self.cache.add_(x)
                return self.cache + 1.0

        blob = _get_delegate_blob(_lower(KVCacheModel(), (torch.randn(4),)))
        method = deserialize_program(blob).methods[0]
        self.assertIn("cache", {mb.fqn for mb in (method.mutable_buffers or [])})
        self.assertNotIn("cache", {c.data_key for c in (method.constants or [])})
        self.assertTrue(
            any(
                s.kind == OutputKind.BUFFER_MUTATION
                for s in (method.output_specs or [])
            )
        )

    def test_reinplace_produces_inplace_relu(self):
        class ReluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.linear(x))

        blob = _get_delegate_blob(_lower(ReluModel(), (torch.randn(1, 8),)))
        graph = deserialize_graph(blob)
        targets = _call_function_targets(graph)
        self.assertTrue(
            any(t is not None and "relu_" in t for t in targets),
            f"expected in-place relu_, got {targets}",
        )

    def test_constants_shipped_via_named_data(self):
        edge = _lower(nn.Linear(4, 4), (torch.randn(1, 4),))
        blob = _get_delegate_blob(edge)
        method = deserialize_program(blob).methods[0]
        self.assertTrue(method.constants)
        data_keys = {c.data_key for c in method.constants}
        self.assertTrue(any("weight" in k for k in data_keys))
