# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaWeightCollector writes a weight storage the methods share only once."""

import contextlib
import gc
import tempfile
import unittest
from unittest import mock

import torch
from executorch.backends.cuda import cuda_weight_collector
from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CPU,
    AOTI_DEVICE_TYPE_CUDA,
    CudaWeightCollector,
    decode_cuda_aoti_metadata,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import PreprocessResult
from torch.export.pt2_archive._package_weights import TensorProperties, Weights


def _weights(**tensors: torch.Tensor) -> Weights:
    return Weights({fqn: (t, TensorProperties(t)) for fqn, t in tensors.items()})


def _parent_result(so_key: str) -> PreprocessResult:
    store = NamedDataStore()
    store.add_named_data(so_key, so_key.encode())
    store.add_named_data("empty_weights", b"", external_tag="aoti_cuda_blob")
    return PreprocessResult(
        processed_bytes=f"{so_key}\nempty_weights".encode(),
        data_store_output=store.get_named_data_store_output(),
    )


def _as_bytes(buffer) -> bytes:
    return buffer if isinstance(buffer, bytes) else buffer.to_bytes()


@contextlib.contextmanager
def _counting_writes():
    """Yields the sizes of the storages the collector writes."""
    writes = []
    materialize_storage = cuda_weight_collector._materialize_storage

    def counting(tensor, directory):
        writes.append(tensor.untyped_storage().nbytes())
        return materialize_storage(tensor, directory)

    with mock.patch.object(cuda_weight_collector, "_materialize_storage", counting):
        yield writes


class CudaWeightReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def _method(self, collector, so_key, weights):
        """One method's preprocess: materialize, then add to the shared store."""
        with collector.capture(lambda _: AOTI_DEVICE_TYPE_CUDA) as capture:
            artifact = collector.materialize(weights, self.directory.name, capture)
        result = _parent_result(so_key)
        collector.add_preprocess_result(result, artifact, "cuda")
        return artifact, result

    def test_a_storage_shared_by_methods_is_written_once(self) -> None:
        weight = torch.randn(1024, 33, device="cuda")
        bias = torch.arange(7, dtype=torch.int8, device="cuda")
        collector = CudaWeightCollector()
        with _counting_writes() as writes:
            first, first_result = self._method(
                collector, "so-0", _weights(w=weight, b=bias)
            )
            # Another method's copies: same bytes, other tensors.
            second, second_result = self._method(
                collector, "so-1", _weights(w=weight.clone(), b=bias.clone())
            )
        collector.finish()
        self.assertEqual(len(writes), 2)
        self.assertEqual(set(first.storages), {e.storage_key for e in first.entries})
        self.assertEqual(second.storages, {})
        self.assertEqual(
            set(second.stored), {e.storage_key for e in second.entries}
        )
        first_manifest = decode_cuda_aoti_metadata(first_result.processed_bytes)
        second_manifest = decode_cuda_aoti_metadata(second_result.processed_bytes)
        for entry in first_manifest.entries:
            self.assertIn(entry.storage_key, collector._store.key_to_buffer_idx)
        self.assertEqual(second_manifest.entries, first_manifest.entries)

    def test_the_stored_bytes_match_writing_every_method(self) -> None:
        tensors = {
            "w": torch.randn(4096, 129, device="cuda"),
            "odd": torch.arange(13, dtype=torch.uint8, device="cuda"),
        }

        def stored_data(reuse: bool):
            collector = CudaWeightCollector()
            same_storage = (
                cuda_weight_collector._same_cuda_storage
                if reuse
                # Never equal, so every method writes its storages.
                else (lambda first, second: False)
            )
            with mock.patch.object(
                cuda_weight_collector, "_same_cuda_storage", same_storage
            ):
                for index in range(3):
                    self._method(
                        collector,
                        f"so-{index}",
                        _weights(**{k: v.clone() for k, v in tensors.items()}),
                    )
            collector.finish()
            store = collector._store
            return {
                key: _as_bytes(store.buffers[index])
                for key, index in store.key_to_buffer_idx.items()
            }

        self.assertEqual(stored_data(reuse=True), stored_data(reuse=False))

    def test_changed_bytes_are_stored_per_method(self) -> None:
        weight = torch.randn(256, 8, device="cuda")
        changed = weight.clone()
        changed[3, 5] += 1
        collector = CudaWeightCollector()
        _, first_result = self._method(collector, "so-0", _weights(w=weight))
        _, second_result = self._method(collector, "so-1", _weights(w=changed))
        self._assert_stored_per_method(
            collector,
            first_result,
            second_result,
            first_bytes=weight.cpu().numpy().tobytes(),
            second_bytes=changed.cpu().numpy().tobytes(),
        )

    def test_library_local_constants_are_written_per_library(self) -> None:
        constant = torch.arange(5, dtype=torch.float32, device="cuda")
        collector = CudaWeightCollector()
        first, first_result = self._method(
            collector, "so-0", _weights(_tensor_constant0=constant)
        )
        second, second_result = self._method(
            collector, "so-1", _weights(_tensor_constant0=constant.clone())
        )
        self.assertEqual(len(first.storages), 1)
        self.assertEqual(len(second.storages), 1)
        self.assertEqual(second.stored, {})
        first_key = decode_cuda_aoti_metadata(first_result.processed_bytes).entries[0]
        second_key = decode_cuda_aoti_metadata(second_result.processed_bytes).entries[0]
        self.assertNotEqual(first_key.storage_key, second_key.storage_key)

    def test_storage_differing_in_two_high_bits_is_stored_per_method(self) -> None:
        weight = torch.arange(3 << 20, dtype=torch.int64, device="cuda")
        changed = weight.clone()
        view = changed.view(torch.uint8)
        view[7] ^= 0x80
        view[-1] ^= 0x80
        collector = CudaWeightCollector()
        _, first_result = self._method(collector, "so-0", _weights(w=weight))
        _, second_result = self._method(collector, "so-1", _weights(w=changed))
        self._assert_stored_per_method(collector, first_result, second_result)

    def test_same_cuda_storage_tells_one_flipped_bit_apart(self) -> None:
        base = torch.randint(0, 255, (3 << 20,), dtype=torch.uint8, device="cuda")
        same = cuda_weight_collector._same_cuda_storage
        self.assertTrue(same(base, base.clone()))
        for index in (0, 12345, base.numel() - 1):
            flipped = base.clone()
            flipped[index] ^= 1
            self.assertFalse(same(base, flipped), index)

    def test_same_cuda_storage_rejects_other_sizes_and_empty_storages(self) -> None:
        same = cuda_weight_collector._same_cuda_storage
        base = torch.zeros(16, dtype=torch.uint8, device="cuda")
        self.assertFalse(same(base, base[:8].clone()))
        empty = torch.empty(0, dtype=torch.uint8, device="cuda")
        self.assertTrue(same(empty, empty.clone()))

    def _moved_method(self, collector, so_key, sources, replace=None, update=None):
        """A method whose weights are moved from ``sources`` and freed after it.

        ``replace`` drops the moved copies before AOTI sees them (as a pass
        replacing them would) and returns the tensors to materialize instead.
        ``update`` writes to the moved copies before AOTI sees them.
        """
        moved = {name: tensor.to("cuda") for name, tensor in sources.items()}
        with collector.capture(lambda _: AOTI_DEVICE_TYPE_CUDA) as capture:
            capture.record_moved_sources(sources, moved)
            if update is not None:
                with torch.no_grad():
                    for tensor in moved.values():
                        update(tensor)
            if replace is not None:
                shapes = {name: tensor.shape for name, tensor in moved.items()}
                # Only the capture's records may keep the storages alive.
                moved.clear()
                gc.collect()
                moved = replace(shapes)
            artifact = collector.materialize(
                _weights(**moved), self.directory.name, capture
            )
        result = _parent_result(so_key)
        collector.add_preprocess_result(result, artifact, "cuda")
        for tensor in moved.values():
            tensor.untyped_storage().resize_(0)
        return artifact, result

    def test_freed_copies_of_one_source_are_written_once(self) -> None:
        sources = {
            "w": torch.randn(1024, 33),
            "b": torch.arange(7, dtype=torch.int8),
        }
        collector = CudaWeightCollector()
        with _counting_writes() as writes:
            first, first_result = self._moved_method(collector, "so-0", sources)
            second, second_result = self._moved_method(collector, "so-1", sources)
            third, _ = self._moved_method(collector, "so-2", sources)
        collector.finish()
        self.assertEqual(len(writes), 2)
        self.assertEqual(second.storages, {})
        self.assertEqual(third.storages, {})
        self.assertEqual(
            set(second.stored), {e.storage_key for e in second.entries}
        )
        self.assertEqual(
            decode_cuda_aoti_metadata(second_result.processed_bytes).entries,
            decode_cuda_aoti_metadata(first_result.processed_bytes).entries,
        )
        stored = {
            key: _as_bytes(collector._store.buffers[index])
            for key, index in collector._store.key_to_buffer_idx.items()
        }
        for entry in decode_cuda_aoti_metadata(first_result.processed_bytes).entries:
            self.assertEqual(
                stored[entry.storage_key],
                sources[entry.fqn].contiguous().view(torch.uint8).numpy().tobytes(),
            )

    def _assert_stored_per_method(
        self, collector, first_result, second_result, first_bytes=None, second_bytes=None
    ) -> None:
        collector.finish()
        first_entry = decode_cuda_aoti_metadata(first_result.processed_bytes).entries[0]
        second_entry = decode_cuda_aoti_metadata(
            second_result.processed_bytes
        ).entries[0]
        self.assertNotEqual(first_entry.storage_key, second_entry.storage_key)
        stored = {
            key: _as_bytes(collector._store.buffers[index])
            for key, index in collector._store.key_to_buffer_idx.items()
        }
        self.assertIn(first_entry.storage_key, stored)
        self.assertIn(second_entry.storage_key, stored)
        if first_bytes is not None:
            self.assertEqual(stored[first_entry.storage_key], first_bytes)
        if second_bytes is not None:
            self.assertEqual(stored[second_entry.storage_key], second_bytes)

    def test_another_source_with_the_same_key_is_stored_per_method(self) -> None:
        collector = CudaWeightCollector()
        _, first_result = self._moved_method(
            collector, "so-0", {"w": torch.randn(256, 8)}
        )
        _, second_result = self._moved_method(
            collector, "so-1", {"w": torch.randn(256, 8)}
        )
        self._assert_stored_per_method(collector, first_result, second_result)

    def test_a_source_view_with_other_metadata_is_written_again(self) -> None:
        base = torch.randn(64, 16)
        for first, second in (
            (base, base[:32]),  # shape
            (base[:32], base[32:]),  # storage offset
            (base[:, :8], base[:, ::2]),  # strides
        ):
            with self.subTest(second=(second.shape, second.storage_offset())):
                collector = CudaWeightCollector()
                with _counting_writes() as writes:
                    _, first_result = self._moved_method(
                        collector, "so-0", {"w": first}
                    )
                    _, second_result = self._moved_method(
                        collector, "so-1", {"w": second}
                    )
                self.assertEqual(len(writes), 2)
                self._assert_stored_per_method(collector, first_result, second_result)

    def test_a_source_updated_in_place_is_stored_per_method(self) -> None:
        sources = {"w": torch.nn.Parameter(torch.randn(256, 8))}
        collector = CudaWeightCollector()
        _, first_result = self._moved_method(collector, "so-0", sources)
        with torch.no_grad():
            sources["w"].add_(1)
        _, second_result = self._moved_method(collector, "so-1", sources)
        self._assert_stored_per_method(collector, first_result, second_result)

    def test_a_moved_copy_written_before_materialize_is_stored_per_method(
        self,
    ) -> None:
        sources = {"w": torch.randn(256, 8)}
        collector = CudaWeightCollector()
        _, first_result = self._moved_method(collector, "so-0", sources)
        _, second_result = self._moved_method(
            collector, "so-1", sources, update=lambda tensor: tensor.add_(1)
        )
        self._assert_stored_per_method(collector, first_result, second_result)

    def test_a_storage_freed_before_materialize_is_not_matched(self) -> None:
        sources = {"w": torch.randn(256, 8)}
        collector = CudaWeightCollector()
        _, first_result = self._moved_method(collector, "so-0", sources)
        # Same source, but the moved copy is freed before AOTI sees it and a
        # storage with other bytes is allocated, possibly at its old address:
        # it must be read and stored as this method's own data.
        _, second_result = self._moved_method(
            collector,
            "so-1",
            sources,
            replace=lambda shapes: {
                name: torch.randn(shape, device="cuda")
                for name, shape in shapes.items()
            },
        )
        self._assert_stored_per_method(collector, first_result, second_result)

    def test_a_new_source_with_the_same_bytes_is_reused_by_source(self) -> None:
        collector = CudaWeightCollector()
        first = torch.randn(256, 8)
        with _counting_writes() as writes:
            self._moved_method(collector, "so-0", {"w": first})
            shared = first.clone()
            self._moved_method(collector, "so-1", {"w": shared})
            self._moved_method(collector, "so-2", {"w": shared})
        self.assertEqual(len(writes), 2)

    def test_reuse_after_divergence_binds_to_the_matched_write(self) -> None:
        # so-1 writes different data than so-0; so-2 reuses so-1's
        # source and must bind to so-1's write, not so-0's.
        collector = CudaWeightCollector()
        first = torch.randn(256, 8)
        with _counting_writes() as writes:
            _, first_result = self._moved_method(collector, "so-0", {"w": first})
            shared = first + 1
            _, second_result = self._moved_method(collector, "so-1", {"w": shared})
            _, third_result = self._moved_method(collector, "so-2", {"w": shared})
        self.assertEqual(len(writes), 2)
        entries = [
            decode_cuda_aoti_metadata(result.processed_bytes).entries[0]
            for result in (first_result, second_result, third_result)
        ]
        self.assertNotEqual(entries[0].storage_key, entries[1].storage_key)
        self.assertEqual(entries[1].storage_key, entries[2].storage_key)
        collector.finish()
        stored = {
            key: _as_bytes(collector._store.buffers[index])
            for key, index in collector._store.key_to_buffer_idx.items()
        }
        self.assertEqual(
            stored[entries[2].storage_key],
            shared.contiguous().view(torch.uint8).numpy().tobytes(),
        )

    def test_a_third_method_reuses_a_non_first_variant(self) -> None:
        weight = torch.randn(256, 8, device="cuda")
        changed = weight + 1
        collector = CudaWeightCollector()
        with _counting_writes() as writes:
            _, first_result = self._method(collector, "so-0", _weights(w=weight))
            _, second_result = self._method(collector, "so-1", _weights(w=changed))
            _, third_result = self._method(
                collector, "so-2", _weights(w=changed.clone())
            )
            _, fourth_result = self._method(
                collector, "so-3", _weights(w=weight.clone())
            )
        self.assertEqual(len(writes), 2)
        entries = [
            decode_cuda_aoti_metadata(result.processed_bytes).entries[0]
            for result in (
                first_result,
                second_result,
                third_result,
                fourth_result,
            )
        ]
        self.assertNotEqual(entries[0].storage_key, entries[1].storage_key)
        self.assertEqual(entries[1].storage_key, entries[2].storage_key)
        self.assertEqual(entries[0].storage_key, entries[3].storage_key)


class _TwoMethodLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(64, 48)
        # A persistent buffer keeps the per-method handling.
        self.register_buffer("scale", (torch.rand(48) + 0.5).to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


class CudaWeightReuseExportTest(unittest.TestCase):
    """Lowers two methods through CudaBackend, as a real export does."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def _lower(self, low_memory: bool):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        module = _TwoMethodLinear().eval()
        programs = {
            name: torch.export.export(module, (torch.randn(rows, 64),), strict=True)
            for name, rows in (("forward_1", 1), ("forward_4", 4))
        }

        def partitioner(name):
            specs = [CudaBackend.generate_method_name_compile_spec(name)]
            if low_memory:
                specs.append(CompileSpec("low_memory_mode", b"ON"))
            return [CudaPartitioner(specs)]

        with _counting_writes() as writes:
            to_edge_transform_and_lower(
                programs,
                partitioner={name: partitioner(name) for name in programs},
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
        return writes

    # The linear weight, its bias and the bf16 buffer.
    _SIZES = (64 * 48 * 4, 48 * 4, 48 * 2)

    def test_low_memory_methods_write_shared_weights_once(self) -> None:
        writes = self._lower(low_memory=True)
        self.assertEqual([writes.count(size) for size in self._SIZES], [1, 1, 2])

    def test_default_mode_writes_each_methods_own_copy(self) -> None:
        writes = self._lower(low_memory=False)
        self.assertEqual([writes.count(size) for size in self._SIZES[:2]], [2, 2])

    def _lower_programs(self, programs):
        """Lower ``programs``; return (write sizes, per-method manifests)."""
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        manifests = []
        add_result = CudaWeightCollector.add_preprocess_result

        def recording(collector, result, artifact, *args, **kwargs):
            out = add_result(collector, result, artifact, *args, **kwargs)
            manifests.append(decode_cuda_aoti_metadata(result.processed_bytes))
            return out

        def partitioner(name):
            return [
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec(name)]
                )
            ]

        with _counting_writes() as writes, mock.patch.object(
            CudaWeightCollector, "add_preprocess_result", recording
        ):
            to_edge_transform_and_lower(
                programs,
                partitioner={name: partitioner(name) for name in programs},
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
        return writes, manifests

    @staticmethod
    def _entry(manifest, suffix):
        return next(e for e in manifest.entries if e.fqn.endswith(suffix))

    def test_buffer_resized_per_method_writes_its_own_storage(self) -> None:
        class _AddBuffer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("pos", torch.randn(16))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.pos

        module = _AddBuffer().eval()
        small = torch.export.export(module, (torch.randn(16),), strict=True)
        module.register_buffer("pos", torch.randn(32))
        large = torch.export.export(module, (torch.randn(32),), strict=True)

        writes, manifests = self._lower_programs({"small": small, "large": large})
        self.assertEqual(sorted(writes), [16 * 4, 32 * 4])
        small_entry = self._entry(manifests[0], "pos")
        large_entry = self._entry(manifests[1], "pos")
        self.assertEqual(
            (small_entry.storage_nbytes, large_entry.storage_nbytes), (64, 128)
        )
        self.assertNotEqual(small_entry.storage_key, large_entry.storage_key)

    def test_mutated_buffer_must_agree_on_initial_data(self) -> None:
        class _StateWriter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(16))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.state.add_(1)
                return x + self.state

        class _StateReader(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(16))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.state

        def export(module):
            return torch.export.export(module.eval(), (torch.randn(16),), strict=True)

        # Same initial data: both manifests name one content key.
        writer = _StateWriter()
        _, manifests = self._lower_programs(
            {"a": export(writer), "b": export(writer)}
        )
        self.assertEqual(
            self._entry(manifests[0], "state").storage_key,
            self._entry(manifests[1], "state").storage_key,
        )

        # Different initial data between two mutating methods: rejected.
        writer = _StateWriter()
        first = export(writer)
        writer.register_buffer("state", torch.ones(16))
        with self.assertRaisesRegex(Exception, "same initial data"):
            self._lower_programs({"a": first, "b": export(writer)})

        # A reader paired with a writer is checked too, in either order.
        for names in (("writer", "reader"), ("reader", "writer")):
            with self.subTest(order=names):
                writer = _StateWriter()
                reader = _StateReader()
                reader.register_buffer("state", torch.ones(16))
                programs = {
                    "writer": export(writer),
                    "reader": export(reader),
                }
                ordered = {name: programs[name] for name in names}
                with self.assertRaisesRegex(Exception, "same initial data"):
                    self._lower_programs(ordered)


class CudaWeightPerMethodDataTest(unittest.TestCase):
    """Methods of one program may hold different data under one FQN."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def _method(self, collector, so_key, tensor):
        with collector.capture(lambda _: AOTI_DEVICE_TYPE_CPU) as capture:
            artifact = collector.materialize(
                _weights(w=tensor), self.directory.name, capture
            )
        result = _parent_result(so_key)
        collector.add_preprocess_result(result, artifact, "cuda")
        return result

    def test_mutated_buffer_with_different_data_is_rejected(self) -> None:
        first = torch.arange(6, dtype=torch.float32)
        second = first + 100
        collector = CudaWeightCollector(mutated_fqns={"w"})
        self._method(collector, "so-0", first)
        with self.assertRaisesRegex(ValueError, "must have the same initial data"):
            self._method(collector, "so-1", second)

    def test_same_fqn_with_different_data_is_stored_per_method(self) -> None:
        first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        second = first.clone()
        second[0, 0] += 100
        collector = CudaWeightCollector()
        first_result = self._method(collector, "so-0", first)
        second_result = self._method(collector, "so-1", second)
        third_result = self._method(collector, "so-2", second.clone())
        collector.finish()

        entries = [
            decode_cuda_aoti_metadata(result.processed_bytes).entries[0]
            for result in (first_result, second_result, third_result)
        ]
        self.assertNotEqual(entries[0].storage_key, entries[1].storage_key)
        self.assertEqual(entries[1].storage_key, entries[2].storage_key)
        stored = {
            key: _as_bytes(collector._store.buffers[index])
            for key, index in collector._store.key_to_buffer_idx.items()
        }
        self.assertEqual(
            stored[entries[0].storage_key], bytes(first.untyped_storage())
        )
        self.assertEqual(
            stored[entries[1].storage_key], bytes(second.untyped_storage())
        )
