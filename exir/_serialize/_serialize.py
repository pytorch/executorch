# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Dict, Tuple

from executorch.exir._serialize import _serialize_pte_binary

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize.data_serializer import (
    DataSerializer,
    SerializationInfo,
    TensorLayout,
)

from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.emit import EmitterOutput
from executorch.exir.schema import Tensor, TensorDataLocation


def serialize(
    emitter_output: EmitterOutput,
    config: ExecutorchBackendConfig,
    data_serializer: DataSerializer,
) -> Tuple[Cord, Dict[str, Cord]]:
    """Serialize the output from Emitter into ExecuTorch artifacts; PTE and PTD files."""
    # Serialize PTE file.
    pte: Cord = _serialize_pte_binary(
        program=emitter_output.program,
        mutable_data=emitter_output.mutable_data,
        extract_delegate_segments=config.extract_delegate_segments,
        segment_alignment=config.segment_alignment,
        constant_tensor_alignment=config.constant_tensor_alignment,
        delegate_alignment=config.delegate_alignment,
    )

    # Serialize PTD files.
    ptd_files: Dict[str, Cord] = {}

    # Find all external tensors and organize into {fqn: Tensor}.
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
                        tensor.extra_tensor_info.fully_qualified_name  # pyre-ignore Undefined attribute [16]
                    ] = TensorLayout(tensor.scalar_type, tensor.sizes, tensor.dim_order)
    if len(fqn_to_tensor_layout) > 0:
        assert emitter_output.external_constant_map is not None
        for (
            file,
            fqn_map,
        ) in (
            # pyre-ignore Undefined attribute [16]: Optional type has no attribute `items`.
            emitter_output.external_constant_map.items()
        ):
            ptd_files[file] = data_serializer.serialize_tensors(
                SerializationInfo(
                    emitter_output.external_constant_buffer,
                    fqn_map,
                    fqn_to_tensor_layout,
                )
            )

    return pte, ptd_files
