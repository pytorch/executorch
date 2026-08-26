# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from executorch.backends.cuda.cuda_weight_collector import (
    CudaAotiMetadata,
    CudaAotiVariant,
    CudaWeightEntry,
    decode_cuda_aoti_metadata,
    encode_cuda_multi_arch_metadata,
)
from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._named_data_store import (
    NamedDataStore,
    NamedDataStoreOutput,
)
from executorch.exir._serialize._program import (
    deserialize_pte_binary,
    PTEFile,
    serialize_pte_binary,
)
from executorch.exir.schema import (
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    DataLocation,
    Program,
)
from executorch.extension.flat_tensor.serialize.serialize import (
    _deserialize_to_flat_tensor,
    FlatTensorHeader,
)


CUDA_BACKEND_ID = "CudaBackend"


@dataclass(frozen=True)
class CudaPteInput:
    """One native CUDA export and its optional external tensor data."""

    pte_path: Path
    ptd_path: Optional[Path]
    legacy_target_sm: Optional[int] = None


@dataclass(frozen=True)
class _PtdBlob:
    offset: int
    size: int


class _PtdIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("rb") as source:
            prefix = source.read(8 + FlatTensorHeader.EXPECTED_LENGTH)
            header = FlatTensorHeader.from_bytes(prefix[8:])
            if not header.is_valid():
                raise ValueError(f"Invalid PTD header in {path}")
            source.seek(0)
            flatbuffer_size = header.flatbuffer_offset + header.flatbuffer_size
            flat_tensor = _deserialize_to_flat_tensor(source.read(flatbuffer_size))

        file_size = path.stat().st_size
        self._blobs: Dict[str, _PtdBlob] = {}
        for named_data in flat_tensor.named_data:
            if named_data.key in self._blobs:
                raise ValueError(f"PTD contains duplicate key {named_data.key!r}")
            if named_data.segment_index >= len(flat_tensor.segments):
                raise ValueError(
                    f"PTD key {named_data.key!r} has an invalid segment index"
                )
            segment = flat_tensor.segments[named_data.segment_index]
            offset = header.segment_base_offset + segment.offset
            if offset + segment.size > file_size:
                raise ValueError(f"PTD key {named_data.key!r} extends past end of file")
            self._blobs[named_data.key] = _PtdBlob(offset, segment.size)
        self._digests: Dict[str, bytes] = {}

    def keys(self) -> set[str]:
        return set(self._blobs)

    def size(self, key: str) -> int:
        try:
            return self._blobs[key].size
        except KeyError as error:
            raise ValueError(f"PTD {self.path} does not contain key {key!r}") from error

    def sha256(self, key: str) -> bytes:
        digest = self._digests.get(key)
        if digest is not None:
            return digest
        try:
            blob = self._blobs[key]
        except KeyError as error:
            raise ValueError(f"PTD {self.path} does not contain key {key!r}") from error

        hasher = hashlib.sha256()
        remaining = blob.size
        with self.path.open("rb") as source:
            source.seek(blob.offset)
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"PTD {self.path} is truncated at key {key!r}")
                hasher.update(chunk)
                remaining -= len(chunk)
        digest = hasher.digest()
        self._digests[key] = digest
        return digest


@dataclass
class _CudaDelegate:
    identity: Tuple[str, int]
    metadata: CudaAotiMetadata


@dataclass
class _Artifact:
    source: CudaPteInput
    pte: PTEFile
    pte_named_data: Dict[str, bytes]
    ptd: Optional[_PtdIndex]
    delegates: List[_CudaDelegate]

    def blob_size(self, key: str) -> int:
        data = self.pte_named_data.get(key)
        if data is not None:
            return len(data)
        if self.ptd is not None:
            return self.ptd.size(key)
        raise ValueError(f"{self.source.pte_path} does not contain named data {key!r}")

    def blob_sha256(self, key: str) -> bytes:
        data = self.pte_named_data.get(key)
        if data is not None:
            return hashlib.sha256(data).digest()
        if self.ptd is not None:
            return self.ptd.sha256(key)
        raise ValueError(f"{self.source.pte_path} does not contain named data {key!r}")


def _named_data_bytes(output: Optional[NamedDataStoreOutput]) -> Dict[str, bytes]:
    if output is None:
        return {}
    return {
        key: bytes(output.buffers[entry.buffer_index])
        for key, entry in output.pte_data.items()
    }


def _delegate_payload(program: Program, delegate) -> bytes:
    if delegate.processed.location != DataLocation.INLINE:
        raise ValueError("PTE deserialization did not restore delegate data inline")
    try:
        return bytes(program.backend_delegate_data[delegate.processed.index].data)
    except IndexError as error:
        raise ValueError("CUDA delegate references invalid processed data") from error


def _load_artifact(source: CudaPteInput) -> _Artifact:
    pte = deserialize_pte_binary(source.pte_path.read_bytes())
    delegates = []
    for plan in pte.program.execution_plan:
        for delegate_index, delegate in enumerate(plan.delegates):
            if delegate.id != CUDA_BACKEND_ID:
                continue
            metadata = decode_cuda_aoti_metadata(
                _delegate_payload(pte.program, delegate)
            )
            if metadata.variants[0].target_sm == 0:
                if source.legacy_target_sm is None:
                    raise ValueError(
                        f"Legacy CUDA metadata in {source.pte_path} requires "
                        "--legacy-target-sm"
                    )
                metadata = CudaAotiMetadata(
                    variants=[
                        CudaAotiVariant(
                            target_sm=source.legacy_target_sm,
                            ptx_compute=source.legacy_target_sm,
                            so_blob_key=metadata.variants[0].so_blob_key,
                        )
                    ],
                    entries=metadata.entries,
                )
            delegates.append(_CudaDelegate((plan.name, delegate_index), metadata))
    if not delegates:
        raise ValueError(f"{source.pte_path} contains no CUDA delegates")
    ptd = _PtdIndex(source.ptd_path) if source.ptd_path is not None else None
    return _Artifact(source, pte, _named_data_bytes(pte.named_data), ptd, delegates)


def _normalized_program(program: Program) -> Program:
    normalized = copy.deepcopy(program)
    payloads = []
    for plan in normalized.execution_plan:
        for delegate in plan.delegates:
            payload = _delegate_payload(normalized, delegate)
            if delegate.id == CUDA_BACKEND_ID:
                payload = b"CUDA_AOTI_VARIANTS"
                delegate.compile_specs = [
                    spec
                    for spec in delegate.compile_specs
                    if spec.key != "cuda_include_ptx"
                ]
            delegate.processed = BackendDelegateDataReference(
                location=DataLocation.INLINE, index=len(payloads)
            )
            payloads.append(BackendDelegateInlineData(data=payload))
    normalized.backend_delegate_data = payloads
    return normalized


def _entry_map(entries: Iterable[CudaWeightEntry]) -> Dict[str, CudaWeightEntry]:
    result = {}
    for entry in entries:
        if entry.fqn in result:
            raise ValueError(f"Duplicate CUDA weight FQN {entry.fqn!r}")
        result[entry.fqn] = entry
    return result


def _entry_without_storage_key(entry: CudaWeightEntry) -> Tuple[object, ...]:
    return (
        entry.fqn,
        entry.storage_nbytes,
        entry.dtype,
        entry.device_type,
        entry.storage_offset,
        entry.sizes,
        entry.strides,
    )


def _validate_shared_weights(
    reference: _Artifact,
    reference_metadata: CudaAotiMetadata,
    candidate: _Artifact,
    candidate_metadata: CudaAotiMetadata,
    identity: Tuple[str, int],
) -> None:
    reference_entries = _entry_map(reference_metadata.entries)
    candidate_entries = _entry_map(candidate_metadata.entries)
    if reference_entries.keys() != candidate_entries.keys():
        raise ValueError(f"CUDA weights differ for delegate {identity}: FQN mismatch")

    for fqn, reference_entry in reference_entries.items():
        candidate_entry = candidate_entries[fqn]
        if _entry_without_storage_key(reference_entry) != _entry_without_storage_key(
            candidate_entry
        ):
            raise ValueError(
                f"CUDA weight metadata differs for delegate {identity}, FQN {fqn!r}"
            )
        if (
            reference.blob_size(reference_entry.storage_key)
            != reference_entry.storage_nbytes
        ):
            raise ValueError(
                f"CUDA weight {fqn!r} has an invalid size in {reference.source.pte_path}"
            )
        if (
            candidate.blob_size(candidate_entry.storage_key)
            != candidate_entry.storage_nbytes
        ):
            raise ValueError(
                f"CUDA weight {fqn!r} has an invalid size in {candidate.source.pte_path}"
            )
        if reference.blob_sha256(reference_entry.storage_key) != candidate.blob_sha256(
            candidate_entry.storage_key
        ):
            raise ValueError(
                f"CUDA weight content differs for delegate {identity}, FQN {fqn!r}"
            )


def _cuda_so_keys(artifact: _Artifact) -> set[str]:
    return {
        variant.so_blob_key
        for delegate in artifact.delegates
        for variant in delegate.metadata.variants
    }


def _cuda_weight_keys(artifact: _Artifact) -> set[str]:
    return {
        entry.storage_key
        for delegate in artifact.delegates
        for entry in delegate.metadata.entries
    }


def _validate_programs(reference: _Artifact, candidate: _Artifact) -> None:
    if _normalized_program(reference.pte.program) != _normalized_program(
        candidate.pte.program
    ):
        raise ValueError(
            f"ExecuTorch programs differ between {reference.source.pte_path} and "
            f"{candidate.source.pte_path}"
        )
    if reference.pte.mutable_data != candidate.pte.mutable_data:
        raise ValueError(
            f"Mutable program data differs between {reference.source.pte_path} and "
            f"{candidate.source.pte_path}"
        )

    reference_non_cuda = {
        key: value
        for key, value in reference.pte_named_data.items()
        if key not in _cuda_so_keys(reference)
        and key not in _cuda_weight_keys(reference)
    }
    candidate_non_cuda = {
        key: value
        for key, value in candidate.pte_named_data.items()
        if key not in _cuda_so_keys(candidate)
        and key not in _cuda_weight_keys(candidate)
    }
    if reference_non_cuda != candidate_non_cuda:
        raise ValueError(
            f"Non-CUDA named data differs between {reference.source.pte_path} and "
            f"{candidate.source.pte_path}"
        )

    reference_external = (
        reference.ptd.keys() - _cuda_weight_keys(reference)
        if reference.ptd is not None
        else set()
    )
    candidate_external = (
        candidate.ptd.keys() - _cuda_weight_keys(candidate)
        if candidate.ptd is not None
        else set()
    )
    if reference_external != candidate_external:
        raise ValueError(
            f"Non-CUDA external data differs between {reference.source.pte_path} and "
            f"{candidate.source.pte_path}"
        )
    for key in reference_external:
        if reference.blob_size(key) != candidate.blob_size(
            key
        ) or reference.blob_sha256(key) != candidate.blob_sha256(key):
            raise ValueError(
                f"External data {key!r} differs between "
                f"{reference.source.pte_path} and {candidate.source.pte_path}"
            )


def _compact_delegate_data(program: Program) -> None:
    payloads = []
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            payload = _delegate_payload(program, delegate)
            delegate.processed = BackendDelegateDataReference(
                location=DataLocation.INLINE, index=len(payloads)
            )
            payloads.append(BackendDelegateInlineData(data=payload))
    program.backend_delegate_data = payloads


def _merge_delegate_variants(
    artifacts: Sequence[_Artifact],
    artifact_delegates: Sequence[Dict[Tuple[str, int], CudaAotiMetadata]],
    reference_metadata: CudaAotiMetadata,
    identity: Tuple[str, int],
    merged_store: NamedDataStore,
) -> List[CudaAotiVariant]:
    variants = []
    target_sms = set()
    reference = artifacts[0]
    for artifact, delegates in zip(artifacts, artifact_delegates):
        metadata = delegates[identity]
        _validate_shared_weights(
            reference, reference_metadata, artifact, metadata, identity
        )
        for variant in metadata.variants:
            if variant.target_sm in target_sms:
                raise ValueError(f"Duplicate CUDA target sm{variant.target_sm}")
            target_sms.add(variant.target_sm)
            try:
                so_data = artifact.pte_named_data[variant.so_blob_key]
            except KeyError as error:
                raise ValueError(
                    f"{artifact.source.pte_path} does not contain CUDA SO "
                    f"{variant.so_blob_key!r}"
                ) from error
            merged_store.add_named_data(variant.so_blob_key, so_data)
            variants.append(variant)

    variants.sort(key=lambda variant: variant.target_sm)
    fallback = next((variant for variant in variants if variant.ptx_compute), None)
    if fallback is not None:
        variants = [
            variant if variant is fallback else replace(variant, ptx_compute=0)
            for variant in variants
        ]
    return variants


def merge_cuda_pte_files(inputs: Sequence[CudaPteInput]) -> Cord:
    """Merge native CUDA exports into a PTE containing one SO per target SM."""
    if len(inputs) < 2:
        raise ValueError("At least two CUDA PTE inputs are required")
    artifacts = [_load_artifact(source) for source in inputs]
    reference = artifacts[0]
    reference_identities = [delegate.identity for delegate in reference.delegates]

    for candidate in artifacts[1:]:
        _validate_programs(reference, candidate)
        if [
            delegate.identity for delegate in candidate.delegates
        ] != reference_identities:
            raise ValueError(
                f"CUDA delegate layout differs between {reference.source.pte_path} "
                f"and {candidate.source.pte_path}"
            )

    merged_program = copy.deepcopy(reference.pte.program)
    for plan in merged_program.execution_plan:
        for delegate in plan.delegates:
            if delegate.id == CUDA_BACKEND_ID:
                delegate.compile_specs = [
                    spec
                    for spec in delegate.compile_specs
                    if spec.key != "cuda_include_ptx"
                ]
    merged_store = NamedDataStore()
    if reference.pte.named_data is not None:
        merged_store.merge_named_data_store(reference.pte.named_data)

    artifact_delegates = [
        {delegate.identity: delegate.metadata for delegate in artifact.delegates}
        for artifact in artifacts
    ]
    expected_target_sms = None
    for identity_index, identity in enumerate(reference_identities):
        reference_metadata = reference.delegates[identity_index].metadata
        variants = _merge_delegate_variants(
            artifacts,
            artifact_delegates,
            reference_metadata,
            identity,
            merged_store,
        )
        current_target_sms = tuple(variant.target_sm for variant in variants)
        if expected_target_sms is None:
            expected_target_sms = current_target_sms
        elif current_target_sms != expected_target_sms:
            raise ValueError(
                f"CUDA target variants differ across delegates at {identity}"
            )
        merged_payload = encode_cuda_multi_arch_metadata(
            variants, reference_metadata.entries
        )
        plan_name, delegate_index = identity
        plan = next(
            plan for plan in merged_program.execution_plan if plan.name == plan_name
        )
        delegate = plan.delegates[delegate_index]
        delegate.processed = BackendDelegateDataReference(
            location=DataLocation.INLINE,
            index=len(merged_program.backend_delegate_data),
        )
        merged_program.backend_delegate_data.append(
            BackendDelegateInlineData(data=merged_payload)
        )

    _compact_delegate_data(merged_program)
    return serialize_pte_binary(
        PTEFile(
            program=merged_program,
            mutable_data=reference.pte.mutable_data,
            named_data=merged_store.get_named_data_store_output(),
        ),
        extract_delegate_segments=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge native CUDA PTE exports into one multi-SM PTE"
    )
    parser.add_argument(
        "--input-pte",
        action="append",
        required=True,
        type=Path,
        help="Native CUDA PTE to merge; the first input supplies common data",
    )
    parser.add_argument(
        "--input-ptd",
        action="append",
        type=Path,
        default=[],
        help="PTD paired by position with --input-pte",
    )
    parser.add_argument(
        "--legacy-target-sm",
        action="append",
        type=int,
        default=[],
        help="Target SM for an FQN3 input; assumes that input carries PTX",
    )
    parser.add_argument("--output-pte", required=True, type=Path)
    parser.add_argument("--output-ptd", type=Path)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point for CUDA PTE merging."""
    args = _parse_args()
    if args.input_ptd and len(args.input_ptd) != len(args.input_pte):
        raise ValueError("--input-ptd must be provided once per --input-pte")
    if args.legacy_target_sm and len(args.legacy_target_sm) != len(args.input_pte):
        raise ValueError("--legacy-target-sm must be provided once per --input-pte")
    if args.input_ptd and args.output_ptd is None:
        raise ValueError("--output-ptd is required when --input-ptd is provided")

    sources = [
        CudaPteInput(
            pte_path=pte_path,
            ptd_path=args.input_ptd[index] if args.input_ptd else None,
            legacy_target_sm=(
                args.legacy_target_sm[index] if args.legacy_target_sm else None
            ),
        )
        for index, pte_path in enumerate(args.input_pte)
    ]
    output = merge_cuda_pte_files(sources)
    args.output_pte.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output_pte.parent, prefix=f".{args.output_pte.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        output.write_to_file(temporary)
    os.replace(temporary_path, args.output_pte)

    if args.output_ptd is not None:
        if not args.input_ptd:
            raise ValueError("--output-ptd requires --input-ptd")
        args.output_ptd.parent.mkdir(parents=True, exist_ok=True)
        if args.input_ptd[0].resolve() != args.output_ptd.resolve():
            shutil.copyfile(args.input_ptd[0], args.output_ptd)


if __name__ == "__main__":
    main()
