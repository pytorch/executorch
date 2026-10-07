# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
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
from executorch.backends.native.serialization.graph_serialize import (
    _extract_constants_and_mutable_buffers,
)
from executorch.backends.native.serialization.schema import OpKind, OutputKind
from executorch.backends.native.test.utils import lifted_constant_program
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.exir import load, save, to_edge, to_edge_transform_and_lower
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.program._program import lift_constant_tensor_pass
from torch.export.graph_signature import InputKind, TensorArgument


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
            graph_module=torch.fx.GraphModule(nn.Module(), torch.fx.Graph()),
            graph_signature=object(),
            state_dict={},
            constants={},
            range_constraints={},
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


class LiftedParameterPreprocessTest(unittest.TestCase):
    def setUp(self):
        model = nn.Sequential(nn.Conv2d(4, 8, 3), nn.BatchNorm2d(8)).eval()
        self.ep = to_edge(
            torch.export.export(model, (torch.randn(1, 4, 8, 8),), strict=True),
            compile_config=get_default_compile_config(),
        ).exported_program()
        FuseBatchNormWithConvPass(self.ep)(self.ep.graph_module)
        lift_constant_tensor_pass(self.ep)
        self.lifted_by_name = {
            spec.arg.name: self.ep.state_dict[spec.target]
            for spec in self.ep.graph_signature.input_specs
            if spec.kind == InputKind.BUFFER
            and spec.target.startswith("_lifted_tensor_constant")
        }
        self.assertEqual(len(self.lifted_by_name), 2)
        self.assertTrue(all(t.requires_grad for t in self.lifted_by_name.values()))

    def test_lifted_parameters_shipped_as_named_data(self):
        result = NativeBackend.preprocess(self.ep, [])
        method = deserialize_program(result.processed_bytes).methods[0]
        refs = {ref.name: ref for ref in method.constants}
        store = result.data_store_output
        self.assertIsNotNone(store)
        self.assertEqual({ref.data_key for ref in refs.values()}, set(store.pte_data))
        for name, tensor in self.lifted_by_name.items():
            entry = store.pte_data[refs[name].data_key]
            self.assertEqual(
                store.buffers[entry.buffer_index], tensor.detach().numpy().tobytes()
            )
            self.assertTrue(tensor.requires_grad)

    def test_lifted_parameters_handed_off_for_ptn(self):
        result = NativeBackend.preprocess(
            self.ep, [CompileSpec(PTN_SERIALIZATION_KEY, b"1")]
        )
        method = deserialize_program(result.processed_bytes).methods[0]
        refs = {ref.name: ref for ref in method.constants}
        info = result._delegate_info_meta
        self.assertIsInstance(info, NativeDelegateInfo)
        self.assertEqual({ref.data_key for ref in refs.values()}, set(info.constants))
        for name, tensor in self.lifted_by_name.items():
            captured = info.constants[refs[name].data_key]
            torch.testing.assert_close(captured, tensor)
            self.assertFalse(captured.requires_grad)
            self.assertTrue(tensor.requires_grad)


def _partition_store(fqn, tensor):
    """Serialize the constants of one partition into its own data store."""
    signature = SimpleNamespace(
        input_specs=[
            SimpleNamespace(
                kind=InputKind.BUFFER,
                arg=TensorArgument(name=f"b_{fqn}"),
                target=fqn,
                persistent=True,
            )
        ]
    )
    refs, data, _ = _extract_constants_and_mutable_buffers(
        signature,
        {fqn: tensor},
        None,
        set(),
    )
    store = NamedDataStore()
    for ref in refs:
        store.add_named_data(ref.data_key, data[ref.data_key])
    return refs[0].data_key, store.get_named_data_store_output()


def _merge(*outputs):
    merged = NamedDataStore()
    for output in outputs:
        merged.merge_named_data_store(output)
    return merged


class LiftedConstantDataKeyTest(unittest.TestCase):
    def test_same_local_name_different_data_after_save_load(self):
        expected_keys = set()
        outputs = []
        for start in (0, 16):
            ep = lifted_constant_program(
                (torch.arange(start, start + 16, dtype=torch.float32),)
            )
            before = NativeBackend.preprocess(ep, [])
            [ref] = deserialize_program(before.processed_bytes).methods[0].constants
            expected_keys.add(ref.data_key)

            archive = io.BytesIO()
            save(ep, archive)
            archive.seek(0)
            restored = load(archive)
            outputs.append(NativeBackend.preprocess(restored, []).data_store_output)

        merged = _merge(*outputs)
        self.assertEqual(set(merged.pte_data), expected_keys)
        self.assertEqual(len(merged.buffers), 2)

    def test_same_local_name_different_data_across_partitions(self):
        # Equal-sized, as in the Conformer failure.
        partitions = []
        for start in (0, 16):
            ep = lifted_constant_program(
                (torch.arange(start, start + 16, dtype=torch.float32),)
            )
            result = NativeBackend.preprocess(ep, [])
            [ref] = deserialize_program(result.processed_bytes).methods[0].constants
            partitions.append((ref.data_key, result.data_store_output))
        (key_a, out_a), (key_b, out_b) = partitions
        self.assertNotEqual(key_a, key_b)
        merged = _merge(out_a, out_b)
        self.assertEqual({key_a, key_b}, set(merged.pte_data))
        self.assertEqual(2, len(merged.buffers))

    def test_same_local_name_different_size_across_partitions(self):
        key_a, out_a = _partition_store(
            "_lifted_tensor_constant12", torch.ones(4, dtype=torch.float32)
        )
        key_b, out_b = _partition_store(
            "_lifted_tensor_constant12", torch.ones(8, dtype=torch.float32)
        )
        self.assertNotEqual(key_a, key_b)
        self.assertEqual(2, len(_merge(out_a, out_b).buffers))

    def test_identical_constants_keep_distinct_keys_and_share_buffer(self):
        tensor = torch.arange(16, dtype=torch.float32)
        key_a, out_a = _partition_store("_lifted_tensor_constant4", tensor.clone())
        key_b, out_b = _partition_store("_lifted_tensor_constant7", tensor.clone())
        self.assertNotEqual(key_a, key_b)
        merged = _merge(out_a, out_b)
        self.assertEqual({key_a, key_b}, set(merged.pte_data))
        self.assertEqual(1, len(merged.buffers))

    def test_same_local_name_and_data_share_key_across_partitions(self):
        tensor = torch.arange(16, dtype=torch.float32)
        key_a, out_a = _partition_store("_lifted_tensor_constant4", tensor.clone())
        key_b, out_b = _partition_store("_lifted_tensor_constant4", tensor.clone())
        self.assertEqual(key_a, key_b)
        self.assertEqual(1, len(_merge(out_a, out_b).buffers))

    def test_real_fqn_is_unchanged(self):
        key, _ = _partition_store("linear.weight", torch.randn(4, 4))
        self.assertEqual("linear.weight", key)

    def test_identical_lora_weights_keep_their_fqns(self):
        fqn_a = "layers.0.attention.wq.lora_b.weight"
        fqn_b = "layers.1.attention.wq.lora_b.weight"
        key_a, out_a = _partition_store(fqn_a, torch.zeros(4, 4))
        key_b, out_b = _partition_store(fqn_b, torch.zeros(4, 4))
        self.assertEqual((fqn_a, fqn_b), (key_a, key_b))
        merged = _merge(out_a, out_b)
        self.assertEqual({fqn_a, fqn_b}, set(merged.pte_data))
        self.assertEqual(1, len(merged.buffers))
