# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NativeBackend — AOT preprocess for the native portable runtime.

Serializes the delegated fx subgraph into the native backend's generic
flatbuffer format (a topological list of fx nodes; see serialization/).
Constant tensor data is shipped via the NamedDataStore, referenced by
fully-qualified name -- never embedded in the graph flatbuffer.

When the partitioner sets PTN_SERIALIZATION_KEY, the constants are handed back on
NativeDelegateInfo instead of being copied into a NamedDataStore, and to_native
packs them once, globally deduped, across every method. processed_bytes is the
same graph flatbuffer either way; only the constant channel differs.

Such a program must be consumed by to_native. _delegate_info_meta is never
serialized into a PTE, so lowering in this mode and then calling to_executorch
yields a program whose constants are simply absent -- see to_native.

Graph cleanup (CSE, reinplace, view_copy collapsing) is expected to run before
lowering via the ``transform_passes`` argument of
``to_edge_transform_and_lower`` (see ``passes.get_default_passes``).
"""

from dataclasses import dataclass, field
from typing import Dict, final, List, Optional

import torch

from executorch.backends.native.partitioner import (
    EXTERNAL_CONSTANTS_TAG_KEY,
    PTN_SERIALIZATION_KEY,
)

from executorch.backends.native.serialization import serialize_graph

from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)

from torch._subclasses.fake_tensor import FakeTensor


@dataclass(frozen=True)
class NativeDelegateInfo:
    """What preprocess hands to to_native alongside the graph blob.

    Carried on ``lowered_module.meta["_delegate_info_meta"]``, which ExecuTorch
    never serializes into a PTE. Tensors are passed by reference rather than
    copied, so to_native must consume them before anything mutates the edge
    program; the sanctioned sequence lowers and packs with nothing in between.

    A container rather than a bare dict so later additions do not have to fight
    over the single _delegate_info_meta slot.
    """

    constants: Dict[str, torch.Tensor] = field(default_factory=dict)


def _parse_compile_specs(
    module_compile_spec: List[CompileSpec],
) -> tuple[Optional[str], bool]:
    """Parse the native specs, rejecting ambiguous recognized configuration."""
    external_tag = None
    serialize_as_ptn = False
    seen: set[str] = set()
    for spec in module_compile_spec:
        if spec.key not in (EXTERNAL_CONSTANTS_TAG_KEY, PTN_SERIALIZATION_KEY):
            continue
        if spec.key in seen:
            raise ValueError(
                f"NativeBackend: duplicate compile spec {spec.key!r} is ambiguous."
            )
        seen.add(spec.key)

        value = bytes(spec.value)
        if spec.key == EXTERNAL_CONSTANTS_TAG_KEY:
            external_tag = value.decode("utf-8")
        else:
            if value != b"1":
                raise ValueError(
                    f"NativeBackend: {PTN_SERIALIZATION_KEY!r} must have value b'1'."
                )
            serialize_as_ptn = True

    if serialize_as_ptn and external_tag is not None:
        raise ValueError(
            "NativeBackend: PTN serialization cannot be combined with "
            "external_constants_tag; the two modes use different constant channels."
        )
    return external_tag, serialize_as_ptn


@final
class NativeBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        flatbuffer_bytes, constant_data = serialize_graph(
            edge_program.graph_module,
            edge_program.graph_signature,
            edge_program.state_dict,
            edge_program.constants,
        )

        external_tag, serialize_as_ptn = _parse_compile_specs(module_compile_spec)

        constants = {
            fqn: tensor.detach()
            for fqn, tensor in constant_data.items()
            if not isinstance(tensor, FakeTensor)
        }

        if serialize_as_ptn:
            # No dedup here: nothing is serialized yet, so hashing would cost a
            # full pass over every weight and free nothing (the tensors are owned
            # by the edge program). Preserve source layout/storage as well, so the
            # PTN writer can reject mutable alias topology before normalization.
            # to_native dedups once, across all methods.
            return PreprocessResult(
                processed_bytes=flatbuffer_bytes,
                data_store_output=None,
                _delegate_info_meta=NativeDelegateInfo(constants=constants),
            )

        named_data_store = NamedDataStore()
        for fqn, tensor in constants.items():
            named_data_store.add_named_data(
                fqn,
                tensor.contiguous(),
                external_tag=external_tag,
            )

        return PreprocessResult(
            processed_bytes=flatbuffer_bytes,
            data_store_output=named_data_store.get_named_data_store_output(),
        )
