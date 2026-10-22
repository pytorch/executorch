# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CUDA,
    CUDA_AOTI_METADATA_MAGIC,
    CudaAotiVariant,
    CudaWeightEntry,
    decode_cuda_aoti_metadata,
    encode_cuda_aoti_metadata,
)


class TestCudaWeightMetadata(unittest.TestCase):
    @staticmethod
    def _entry() -> CudaWeightEntry:
        return CudaWeightEntry(
            fqn="model.weight",
            storage_key="cuda_fqn_weight:cuda:model.weight",
            storage_nbytes=24,
            dtype=6,
            device_type=AOTI_DEVICE_TYPE_CUDA,
            storage_offset=0,
            sizes=(2, 3),
            strides=(3, 1),
        )

    def test_targeted_metadata_has_shared_weights(self) -> None:
        entry = self._entry()
        encoded = encode_cuda_aoti_metadata(
            [
                CudaAotiVariant(80, 0, "sm80-so"),
                CudaAotiVariant(120, 0, "sm120-so"),
            ],
            [entry],
        )
        self.assertTrue(encoded.startswith(CUDA_AOTI_METADATA_MAGIC))
        decoded = decode_cuda_aoti_metadata(encoded)
        self.assertEqual(
            decoded.variants,
            [
                CudaAotiVariant(80, 0, "sm80-so"),
                CudaAotiVariant(120, 0, "sm120-so"),
            ],
        )
        self.assertEqual(decoded.entries, [entry])

    def test_metadata_rejects_duplicate_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate CUDA target SM"):
            encode_cuda_aoti_metadata(
                [
                    CudaAotiVariant(80, 0, "first"),
                    CudaAotiVariant(80, 0, "second"),
                ],
                [self._entry()],
            )

    def test_fallback_metadata_allows_matching_regular_target(self) -> None:
        entry = self._entry()
        encoded = encode_cuda_aoti_metadata(
            [
                CudaAotiVariant(80, 0, "sm80-so"),
                CudaAotiVariant(80, 80, "fallback-so", fallback_only=True),
            ],
            [entry],
        )
        self.assertTrue(encoded.startswith(CUDA_AOTI_METADATA_MAGIC))
        decoded = decode_cuda_aoti_metadata(encoded)
        self.assertEqual(
            decoded.variants,
            [
                CudaAotiVariant(80, 0, "sm80-so"),
                CudaAotiVariant(80, 80, "fallback-so", fallback_only=True),
            ],
        )

    def test_fallback_metadata_rejects_multiple_fallbacks(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one fallback"):
            encode_cuda_aoti_metadata(
                [
                    CudaAotiVariant(80, 80, "first", fallback_only=True),
                    CudaAotiVariant(75, 75, "second", fallback_only=True),
                ],
                [self._entry()],
            )

    def test_multi_variant_metadata_rejects_implicit_ptx_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit PTX fallback"):
            encode_cuda_aoti_metadata(
                [
                    CudaAotiVariant(80, 80, "sm80-so"),
                    CudaAotiVariant(120, 0, "sm120-so"),
                ],
                [self._entry()],
            )

    def test_untargeted_metadata_for_rocm(self) -> None:
        encoded = encode_cuda_aoti_metadata(
            [CudaAotiVariant(0, 0, "rocm-so")], [self._entry()]
        )
        self.assertTrue(encoded.startswith(CUDA_AOTI_METADATA_MAGIC))
        decoded = decode_cuda_aoti_metadata(encoded)
        self.assertEqual(decoded.variants, [CudaAotiVariant(0, 0, "rocm-so")])
        self.assertEqual(decoded.entries, [self._entry()])

    def test_untargeted_metadata_cannot_mix_with_targeted_variants(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires one non-fallback variant"):
            encode_cuda_aoti_metadata(
                [
                    CudaAotiVariant(0, 0, "rocm-so"),
                    CudaAotiVariant(80, 0, "sm80-so"),
                ],
                [self._entry()],
            )

    def test_metadata_rejects_trailing_data(self) -> None:
        encoded = encode_cuda_aoti_metadata(
            [CudaAotiVariant(80, 80, "sm80-so")], [self._entry()]
        )
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            decode_cuda_aoti_metadata(encoded + b"\0")


if __name__ == "__main__":
    unittest.main()
