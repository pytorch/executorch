# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CUDA,
    CudaAotiVariant,
    CudaWeightEntry,
    decode_cuda_aoti_metadata,
    encode_cuda_multi_arch_metadata,
    encode_cuda_weight_metadata,
)
from executorch.backends.cuda.merge_ptes import (
    CudaPteInput,
    main as merge_main,
    merge_cuda_pte_files,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._serialize._program import (
    deserialize_pte_binary,
    PTEFile,
    serialize_pte_binary,
)
from executorch.exir._serialize.data_serializer import DataEntry, DataPayload
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    ContainerMetadata,
    DataLocation,
    ExecutionPlan,
    Program,
    SubsegmentOffsets,
)
from executorch.extension.flat_tensor.serialize.serialize import FlatTensorSerializer


@patch("executorch.backends.cuda.merge_ptes.torch.version.hip", None)
class TestMergeCudaPtes(unittest.TestCase):
    def _write_artifact(
        self,
        directory: Path,
        target_sm: int,
        so_data: bytes,
        weight_data: bytes,
        *,
        fqn: str = "model.weight",
        weight_key: str = "cuda_fqn_weight:cuda:model.weight",
        ptx_compute: int | None = None,
        legacy: bool = False,
    ) -> CudaPteInput:
        if ptx_compute is None:
            ptx_compute = target_sm
        so_key = hashlib.sha256(so_data).hexdigest() + "_so_blob"
        entry = CudaWeightEntry(
            fqn=fqn,
            storage_key=weight_key,
            storage_nbytes=len(weight_data),
            dtype=1,
            device_type=AOTI_DEVICE_TYPE_CUDA,
            storage_offset=0,
            sizes=(len(weight_data),),
            strides=(1,),
        )
        metadata = (
            encode_cuda_weight_metadata(so_key, [entry])
            if legacy
            else encode_cuda_multi_arch_metadata(
                [CudaAotiVariant(target_sm, ptx_compute, so_key)], [entry]
            )
        )
        compile_specs = [CompileSpec("method_name", b"forward")]
        compile_specs.append(
            CompileSpec("cuda_include_ptx", b"ON" if ptx_compute else b"OFF")
        )
        delegate = BackendDelegate(
            id="CudaBackend",
            processed=BackendDelegateDataReference(DataLocation.INLINE, 0),
            compile_specs=compile_specs,
        )
        program = Program(
            version=0,
            execution_plan=[
                ExecutionPlan(
                    name="forward",
                    container_meta_type=ContainerMetadata("", ""),
                    values=[],
                    inputs=[],
                    outputs=[],
                    chains=[],
                    operators=[],
                    delegates=[delegate],
                    non_const_buffer_sizes=[],
                )
            ],
            constant_buffer=[],
            backend_delegate_data=[BackendDelegateInlineData(metadata)],
            segments=[],
            constant_segment=SubsegmentOffsets(0, []),
        )
        store = NamedDataStore()
        store.add_named_data(so_key, so_data)
        pte_path = directory / "model.pte"
        with pte_path.open("wb") as output:
            serialize_pte_binary(
                PTEFile(
                    program=program, named_data=store.get_named_data_store_output()
                ),
                extract_delegate_segments=True,
            ).write_to_file(output)

        ptd_path = directory / "aoti_cuda_blob.ptd"
        serializer = FlatTensorSerializer()
        with ptd_path.open("wb") as output:
            serializer.serialize(
                DataPayload(
                    buffers=[weight_data],
                    named_data={weight_key: DataEntry(0, 1, None)},
                )
            ).write_to_file(output)
        return CudaPteInput(
            pte_path=pte_path,
            ptd_path=ptd_path,
            legacy_target_sm=target_sm if legacy else None,
        )

    def test_merges_variants_and_keeps_one_weight_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            sm80.mkdir()
            sm120.mkdir()
            inputs = [
                self._write_artifact(sm80, 80, b"sm80-so", b"shared-weight"),
                self._write_artifact(sm120, 120, b"sm120-so", b"shared-weight"),
            ]

            merged = deserialize_pte_binary(bytes(merge_cuda_pte_files(inputs)))
            delegate = merged.program.execution_plan[0].delegates[0]
            payload = merged.program.backend_delegate_data[
                delegate.processed.index
            ].data
            metadata = decode_cuda_aoti_metadata(payload)
            self.assertEqual(
                [variant.target_sm for variant in metadata.variants], [80, 120]
            )
            self.assertEqual(
                [variant.ptx_compute for variant in metadata.variants], [80, 0]
            )
            self.assertEqual(len(metadata.entries), 1)
            self.assertEqual(metadata.entries[0].fqn, "model.weight")
            self.assertEqual(
                set(merged.named_data.pte_data),
                {
                    hashlib.sha256(b"sm80-so").hexdigest() + "_so_blob",
                    hashlib.sha256(b"sm120-so").hexdigest() + "_so_blob",
                },
            )
            self.assertNotIn(
                "cuda_include_ptx",
                {
                    spec.key
                    for spec in merged.program.execution_plan[0]
                    .delegates[0]
                    .compile_specs
                },
            )

    def test_allows_only_lowest_source_to_contain_ptx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            sm80.mkdir()
            sm120.mkdir()
            inputs = [
                self._write_artifact(sm80, 80, b"sm80-so", b"weight"),
                self._write_artifact(
                    sm120,
                    120,
                    b"sm120-so",
                    b"weight",
                    ptx_compute=0,
                ),
            ]

            merged = deserialize_pte_binary(bytes(merge_cuda_pte_files(inputs)))
            delegate = merged.program.execution_plan[0].delegates[0]
            metadata = decode_cuda_aoti_metadata(
                merged.program.backend_delegate_data[delegate.processed.index].data
            )
            self.assertEqual(
                [variant.ptx_compute for variant in metadata.variants], [80, 0]
            )

    def test_preserves_no_ptx_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            sm80.mkdir()
            sm120.mkdir()
            inputs = [
                self._write_artifact(sm80, 80, b"sm80-so", b"weight", ptx_compute=0),
                self._write_artifact(
                    sm120,
                    120,
                    b"sm120-so",
                    b"weight",
                    ptx_compute=0,
                ),
            ]

            merged = deserialize_pte_binary(bytes(merge_cuda_pte_files(inputs)))
            delegate = merged.program.execution_plan[0].delegates[0]
            metadata = decode_cuda_aoti_metadata(
                merged.program.backend_delegate_data[delegate.processed.index].data
            )
            self.assertEqual(
                [variant.ptx_compute for variant in metadata.variants], [0, 0]
            )

    def test_merges_legacy_fqn3_source_with_explicit_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            sm80.mkdir()
            sm120.mkdir()
            inputs = [
                self._write_artifact(sm80, 80, b"sm80-so", b"weight", legacy=True),
                self._write_artifact(sm120, 120, b"sm120-so", b"weight"),
            ]

            merged = deserialize_pte_binary(bytes(merge_cuda_pte_files(inputs)))
            delegate = merged.program.execution_plan[0].delegates[0]
            metadata = decode_cuda_aoti_metadata(
                merged.program.backend_delegate_data[delegate.processed.index].data
            )
            self.assertEqual(
                [variant.target_sm for variant in metadata.variants], [80, 120]
            )
            self.assertEqual(
                [variant.ptx_compute for variant in metadata.variants], [80, 0]
            )

    def test_rejects_different_weight_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            sm80.mkdir()
            sm120.mkdir()
            inputs = [
                self._write_artifact(sm80, 80, b"sm80-so", b"first-weight"),
                self._write_artifact(sm120, 120, b"sm120-so", b"other-weight"),
            ]
            with self.assertRaisesRegex(ValueError, "weight content differs"):
                merge_cuda_pte_files(inputs)

    def test_library_local_weight_keys_are_normalized_to_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            sm80.mkdir()
            sm120.mkdir()
            inputs = [
                self._write_artifact(
                    sm80,
                    80,
                    b"sm80-so",
                    b"constant",
                    fqn="_tensor_constant0",
                    weight_key="cuda_fqn_weight:cuda:sm80-so:_tensor_constant0",
                ),
                self._write_artifact(
                    sm120,
                    120,
                    b"sm120-so",
                    b"constant",
                    fqn="_tensor_constant0",
                    weight_key="cuda_fqn_weight:cuda:sm120-so:_tensor_constant0",
                ),
            ]

            merged = deserialize_pte_binary(bytes(merge_cuda_pte_files(inputs)))
            delegate = merged.program.execution_plan[0].delegates[0]
            metadata = decode_cuda_aoti_metadata(
                merged.program.backend_delegate_data[delegate.processed.index].data
            )
            self.assertEqual(
                metadata.entries[0].storage_key,
                "cuda_fqn_weight:cuda:sm80-so:_tensor_constant0",
            )

    def test_rejects_duplicate_target_sm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            inputs = [
                self._write_artifact(first, 80, b"first-so", b"weight"),
                self._write_artifact(second, 80, b"second-so", b"weight"),
            ]
            with self.assertRaisesRegex(ValueError, "Duplicate CUDA target sm80"):
                merge_cuda_pte_files(inputs)

    def test_rejects_rocm(self) -> None:
        with patch(
            "executorch.backends.cuda.merge_ptes.torch.version.hip", "6.3"
        ), self.assertRaisesRegex(RuntimeError, "only NVIDIA CUDA"):
            merge_cuda_pte_files([])

    def test_cli_writes_merged_pte_and_reuses_base_ptd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sm80 = root / "sm80"
            sm120 = root / "sm120"
            output = root / "output"
            sm80.mkdir()
            sm120.mkdir()
            first = self._write_artifact(sm80, 80, b"sm80-so", b"weight")
            second = self._write_artifact(sm120, 120, b"sm120-so", b"weight")
            self.assertIsNotNone(first.ptd_path)
            output_pte = output / "model.pte"
            output_ptd = output / "aoti_cuda_blob.ptd"

            with patch(
                "sys.argv",
                [
                    "merge_ptes",
                    "--input-pte",
                    str(first.pte_path),
                    "--input-pte",
                    str(second.pte_path),
                    "--input-ptd",
                    str(first.ptd_path),
                    "--input-ptd",
                    str(second.ptd_path),
                    "--output-pte",
                    str(output_pte),
                    "--output-ptd",
                    str(output_ptd),
                ],
            ):
                merge_main()

            self.assertTrue(output_pte.is_file())
            assert first.ptd_path is not None
            self.assertEqual(output_ptd.read_bytes(), first.ptd_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
