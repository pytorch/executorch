# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, Optional, Tuple

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._named_data_store import (
    NamedDataStore,
    NamedDataStoreOutput,
)
from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
from executorch.exir._serialize.data_serializer import DataPayload, DataSerializer
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.emit import EmitterOutput
from executorch.exir.schema import Program, Tensor, TensorDataLocation
from executorch.exir.tensor_layout import TensorLayout


def _extract_external_tensor_layouts(program: Program) -> Dict[str, TensorLayout]:
    # Find all external tensors and organize into {fqn: TensorLayout}.
    fqn_to_tensor_layout: Dict[str, TensorLayout] = {}
    for plan in program.execution_plan:
        for evalue in plan.values:
            if isinstance(evalue.val, Tensor):
                tensor = evalue.val
                if (
                    tensor.extra_tensor_info is not None
                    and tensor.extra_tensor_info.fully_qualified_name is not None
                    and tensor.extra_tensor_info.location is TensorDataLocation.EXTERNAL
                ):
                    fqn_to_tensor_layout[
                        # pyre-ignore Undefined attribute [16]: `Optional` has no attribute `fully_qualified_name`
                        tensor.extra_tensor_info.fully_qualified_name
                    ] = TensorLayout(tensor.scalar_type, tensor.sizes, tensor.dim_order)

    return fqn_to_tensor_layout


def serialize_for_executorch(
    emitter_output: EmitterOutput,
    config: ExecutorchBackendConfig,
    data_serializer: DataSerializer,
    named_data_store: Optional[NamedDataStoreOutput] = None,
) -> Tuple[Cord, Dict[str, Cord]]:
    """Serialize the output from Emitter into ExecuTorch artifacts; PTE and PTD files."""

    # Serialize PTE file.
    pte_named_data = None
    if (
        named_data_store is not None
        and len(named_data_store.buffers) > 0
        and len(named_data_store.pte_data) > 0
    ):
        # Create a separate NamedDataStoreOutput with only pte_data; exclude
        # external_data, which shouldn't be serialized with the PTE file.
        if len(named_data_store.external_data) == 0:
            pte_named_data = named_data_store
        else:
            pte_named_data = NamedDataStoreOutput(
                buffers=named_data_store.buffers,
                pte_data=named_data_store.pte_data,
                external_data={},
            )
    pte: Cord = serialize_pte_binary(
        pte_file=PTEFile(
            program=emitter_output.program,
            mutable_data=emitter_output.mutable_data,
            named_data=pte_named_data,
        ),
        extract_delegate_segments=config.extract_delegate_segments,
        segment_alignment=config.segment_alignment,
        constant_tensor_alignment=config.constant_tensor_alignment,
        delegate_alignment=config.delegate_alignment,
    )

    # Early exit if no external weights.
    if len(emitter_output.external_constant_map) == 0 and (
        named_data_store is None or len(named_data_store.external_data) == 0
    ):
        return pte, {}

    ptd_files: Dict[str, Cord] = {}

    # If there are no emitter constants, use named_data_store directly.
    if len(emitter_output.external_constant_map) == 0:
        for tag in named_data_store.external_data.keys():
            ptd_files[tag] = data_serializer.serialize(
                DataPayload(
                    buffers=named_data_store.buffers,
                    named_data=named_data_store.external_data[tag],
                )
            )
        return pte, ptd_files

    # Collect external weights from emitter output and merge them.
    fqn_to_tensor_layout = _extract_external_tensor_layouts(emitter_output.program)
    updated_named_data_store = NamedDataStore()
    # Add tensor constants from the emitter to the NamedDataStore.
    for tag, fqn_to_index in emitter_output.external_constant_map.items():
        for fqn, index in fqn_to_index.items():
            updated_named_data_store.add_named_data(
                fqn,
                emitter_output.external_constant_buffer[index],
                tensor_layout=fqn_to_tensor_layout[fqn],
                external_tag=tag,
            )
    updated_named_data_store.merge_named_data_store(named_data_store)

    # Serialize each tag into a PTD file.
    for tag in updated_named_data_store.external_data.keys():
        ptd_files[tag] = data_serializer.serialize(
            DataPayload(
                buffers=updated_named_data_store.buffers,
                named_data=updated_named_data_store.external_data[tag],
            )
        )

    return pte, ptd_files
