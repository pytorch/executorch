# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaWeightCollector writes a weight storage the methods share only once."""

import tempfile
import unittest
from unittest import mock

import torch
from executorch.backends.cuda import cuda_weight_collector
from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CUDA,
    CudaWeightCollector,
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


class CudaWeightReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def _method(self, collector, so_key, weights):
        """One method's preprocess: materialize, then add to the shared store."""
        artifact = collector.materialize(
            weights, self.directory.name, lambda _: AOTI_DEVICE_TYPE_CUDA
        )
        collector.add_preprocess_result(_parent_result(so_key), artifact, "cuda")
        return artifact

    def test_a_storage_shared_by_methods_is_written_once(self) -> None:
        weight = torch.randn(1024, 33, device="cuda")
        bias = torch.arange(7, dtype=torch.int8, device="cuda")
        collector = CudaWeightCollector()
        writes = []
        materialize_storage = cuda_weight_collector._materialize_storage

        def counting(tensor, directory):
            writes.append(tensor.untyped_storage().nbytes())
            return materialize_storage(tensor, directory)

        with mock.patch.object(cuda_weight_collector, "_materialize_storage", counting):
            first = self._method(collector, "so-0", _weights(w=weight, b=bias))
            # Another method's copies: same bytes, other tensors.
            second = self._method(
                collector, "so-1", _weights(w=weight.clone(), b=bias.clone())
            )
        collector.finish()
        self.assertEqual(len(writes), 2)
        self.assertEqual(set(first.storages), {e.storage_key for e in first.entries})
        self.assertEqual(second.storages, {})
        self.assertEqual(second.stored, {e.storage_key for e in second.entries})
        for entry in first.entries:
            self.assertIn(entry.storage_key, collector._store.key_to_buffer_idx)

    def test_the_stored_bytes_match_writing_every_method(self) -> None:
        tensors = {
            "w": torch.randn(4096, 129, device="cuda"),
            "odd": torch.arange(13, dtype=torch.uint8, device="cuda"),
        }

        def stored_data(reuse: bool):
            collector = CudaWeightCollector()
            fingerprint = (
                cuda_weight_collector._cuda_storage_fingerprint
                if reuse
                # Never equal, so every method writes its storages.
                else (lambda tensor: object())
            )
            with mock.patch.object(
                cuda_weight_collector, "_cuda_storage_fingerprint", fingerprint
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
        first = self._method(collector, "so-0", _weights(_tensor_constant0=constant))
        second = self._method(
            collector, "so-1", _weights(_tensor_constant0=constant.clone())
        )
        self.assertEqual(len(first.storages), 1)
        self.assertEqual(len(second.storages), 1)
        self.assertEqual(second.stored, set())

    def test_fingerprint_tells_one_flipped_bit_apart(self) -> None:
        base = torch.randint(0, 255, (3 << 20,), dtype=torch.uint8, device="cuda")
        fingerprint = cuda_weight_collector._cuda_storage_fingerprint
        self.assertEqual(fingerprint(base), fingerprint(base.clone()))
        for index in (0, 12345, base.numel() - 1):
            flipped = base.clone()
            flipped[index] ^= 1
            self.assertNotEqual(fingerprint(base), fingerprint(flipped), index)


if __name__ == "__main__":
    unittest.main()
