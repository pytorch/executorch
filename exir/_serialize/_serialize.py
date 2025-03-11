# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, Optional, Set, Tuple

from executorch.exir._serialize import _serialize_pte_binary

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._named_data_store import NamedDataStoreOutput
from executorch.exir._serialize.data_serializer import (
    DataEntry,
    DataPayload,
    DataSerializer,
    TensorEntry,
    TensorLayout,
)

from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.emit import EmitterOutput
from executorch.exir.schema import Tensor, TensorDataLocation


def serialize_for_executorch(
    emitter_output: EmitterOutput,
    config: ExecutorchBackendConfig,
    data_serializer: DataSerializer,
    named_data: Optional[NamedDataStoreOutput] = None,
) -> Tuple[Cord, Dict[str, Cord]]:
    """Serialize the output from Emitter into ExecuTorch artifacts; PTE and PTD files."""

    # Serialize PTE file.
    pte_named_data = None
    if (
        named_data is not None
        and len(named_data.buffers) > 0
        and len(named_data.pte_data) > 0
    ):
        # Create a separate NamedDataStoreOutput with only pte_data; exclude
        # external_data, which shouldn't be serialized with the PTE file.
        pte_named_data = NamedDataStoreOutput(
            buffers=named_data.buffers,
            pte_data=named_data.pte_data,
            external_data={},
        )
    pte: Cord = _serialize_pte_binary(
        program=emitter_output.program,
        mutable_data=emitter_output.mutable_data,
        extract_delegate_segments=config.extract_delegate_segments,
        segment_alignment=config.segment_alignment,
        constant_tensor_alignment=config.constant_tensor_alignment,
        delegate_alignment=config.delegate_alignment,
        named_data=pte_named_data,
    )

    # Serialize PTD files.
    ptd_files: Dict[str, Cord] = {}

    # Find all external tensors and organize into {fqn: TensorLayout}.
    fqn_to_tensor_layout: Dict[str, TensorLayout] = {}
    for plan in emitter_output.program.execution_plan:
        for evalue in plan.values:
            if isinstance(evalue.val, Tensor):
                tensor = evalue.val
                if (
                    tensor.extra_tensor_info is not None
                    and tensor.extra_tensor_info.fully_qualified_name is not None
                    and tensor.extra_tensor_info.location is TensorDataLocation.EXTERNAL
                ):
                    fqn_to_tensor_layout[
                        tensor.extra_tensor_info.fully_qualified_name
                    ] = TensorLayout(tensor.scalar_type, tensor.sizes, tensor.dim_order)

    if len(fqn_to_tensor_layout) == 0 and (
        named_data is None or len(named_data.external_data) == 0
    ):
        return pte, ptd_files

    all_external_files: Set[str] = set()
    if named_data is not None and len(named_data.external_data) > 0:
        assert (
            len(named_data.buffers) > 0
        ), "External data exists, but there are no buffers provided."
        all_external_files = set(named_data.external_data.keys())

    if len(fqn_to_tensor_layout) > 0:
        # emitter_output.external_constant_map contains the mapping from
        # {file: {fqn: index into external_constant_buffer}}
        # Contains the locations of the tensor buffers, and must be non-empty
        # if there are external tensors to serialize.
        assert (
            emitter_output.external_constant_map is not None
        ), "External exists, but there are no buffers provided."
        all_external_files = all_external_files | set(
            emitter_output.external_constant_map.keys()
        )

    for filename in all_external_files:
        fqn_to_tensor_entry: Dict[str, TensorEntry] = {}
        fqn_to_index = emitter_output.external_constant_map.get(filename, {})
        # Create a TensorEntry for each external tensor.
        for fqn, index in fqn_to_index.items():
            assert fqn in fqn_to_tensor_layout
            fqn_to_tensor_entry[fqn] = TensorEntry(
                buffer_index=index,
                layout=fqn_to_tensor_layout[fqn],
            )

        # Extract external data.
        key_to_data: Dict[str, DataEntry] = {}
        key_to_buffer_index = named_data.external_data.get(filename, {})
        for key, index in key_to_buffer_index.items():
            key_to_data[key] = DataEntry(index, named_data.buffers[index].alignment)

        # Serialize into PTD file.
        ptd_files[filename] = data_serializer.serialize(
            DataPayload(
                buffers=emitter_output.external_constant_buffer,
                fqn_to_tensor=fqn_to_tensor_entry,
                key_to_data=key_to_data,
            )
        )

    return pte, ptd_files
