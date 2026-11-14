# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaWeightCollector writes a weight storage the methods share only once."""

import contextlib
import tempfile
import unittest
from unittest import mock

import torch
from executorch.backends.cuda import cuda_weight_collector
from executorch.backends.cuda.cuda_weight_collector import (
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
        self.assertEqual(second.stored, {e.storage_key for e in second.entries})
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

    def test_changed_bytes_are_written_and_rejected_as_before(self) -> None:
        weight = torch.randn(256, 8, device="cuda")
        changed = weight.clone()
        changed[3, 5] += 1
        collector = CudaWeightCollector()
        self._method(collector, "so-0", _weights(w=weight))
        with self.assertRaisesRegex(ValueError, "with different data"):
            self._method(collector, "so-1", _weights(w=changed))

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
        self.assertEqual(second.stored, set())
        first_key = decode_cuda_aoti_metadata(first_result.processed_bytes).entries[0]
        second_key = decode_cuda_aoti_metadata(second_result.processed_bytes).entries[0]
        self.assertNotEqual(first_key.storage_key, second_key.storage_key)

    def test_storage_differing_in_two_high_bits_is_rejected(self) -> None:
        weight = torch.arange(3 << 20, dtype=torch.int64, device="cuda")
        changed = weight.clone()
        view = changed.view(torch.uint8)
        view[7] ^= 0x80
        view[-1] ^= 0x80
        collector = CudaWeightCollector()
        self._method(collector, "so-0", _weights(w=weight))
        with self.assertRaisesRegex(ValueError, "with different data"):
            self._method(collector, "so-1", _weights(w=changed))

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

    def _moved_method(self, collector, so_key, sources, replace=None):
        """A method whose weights are moved from ``sources`` and freed after it.

        ``replace`` frees the moved copies before AOTI sees them (as a pass
        replacing them would) and returns the tensors to materialize instead.
        """
        moved = {name: tensor.to("cuda") for name, tensor in sources.items()}
        with collector.capture(lambda _: AOTI_DEVICE_TYPE_CUDA) as capture:
            capture.track_moved_tensors(sources, moved)
            weights = moved
            if replace is not None:
                shapes = {name: tensor.shape for name, tensor in moved.items()}
                for tensor in moved.values():
                    tensor.untyped_storage().resize_(0)
                weights = replace(shapes)
            artifact = collector.materialize(
                _weights(**weights), self.directory.name, capture
            )
        result = _parent_result(so_key)
        collector.add_preprocess_result(result, artifact, "cuda")
        for tensor in [*moved.values(), *weights.values()]:
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
        self.assertEqual(second.stored, {e.storage_key for e in second.entries})
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

    def test_another_source_with_the_same_key_is_written_and_checked(self) -> None:
        collector = CudaWeightCollector()
        self._moved_method(collector, "so-0", {"w": torch.randn(256, 8)})
        with self.assertRaisesRegex(ValueError, "with different data"):
            self._moved_method(collector, "so-1", {"w": torch.randn(256, 8)})

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
                    self._moved_method(collector, "so-0", {"w": first})
                    with self.assertRaisesRegex(ValueError, "with different data"):
                        self._moved_method(collector, "so-1", {"w": second})
                self.assertEqual(len(writes), 2)

    def test_a_source_updated_in_place_is_written_and_checked(self) -> None:
        sources = {"w": torch.nn.Parameter(torch.randn(256, 8))}
        collector = CudaWeightCollector()
        self._moved_method(collector, "so-0", sources)
        with torch.no_grad():
            sources["w"].add_(1)
        with self.assertRaisesRegex(ValueError, "with different data"):
            self._moved_method(collector, "so-1", sources)

    def test_a_storage_freed_before_materialize_is_not_matched(self) -> None:
        sources = {"w": torch.randn(256, 8)}
        collector = CudaWeightCollector()
        self._moved_method(collector, "so-0", sources)
        # Same source, but the moved copy is freed before AOTI sees it and a
        # storage with other bytes takes its place: it must be read and checked.
        with self.assertRaisesRegex(ValueError, "with different data"):
            self._moved_method(
                collector,
                "so-1",
                sources,
                replace=lambda shapes: {
                    name: torch.randn(shape, device="cuda")
                    for name, shape in shapes.items()
                },
            )

    def test_a_freed_written_source_lets_a_live_source_take_over(self) -> None:
        collector = CudaWeightCollector()
        first = torch.randn(256, 8)
        with _counting_writes() as writes:
            self._moved_method(collector, "so-0", {"w": first})
            shared = first.clone()
            del first
            self._moved_method(collector, "so-1", {"w": shared})
            self._moved_method(collector, "so-2", {"w": shared})
        self.assertEqual(len(writes), 2)


class _TwoMethodLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(64, 48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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

    # The linear weight and its bias.
    _SIZES = (64 * 48 * 4, 48 * 4)

    def test_low_memory_methods_write_shared_weights_once(self) -> None:
        writes = self._lower(low_memory=True)
        self.assertEqual([writes.count(size) for size in self._SIZES], [1, 1])

    def test_default_mode_writes_each_methods_own_copy(self) -> None:
        writes = self._lower(low_memory=False)
        self.assertEqual([writes.count(size) for size in self._SIZES], [2, 2])
