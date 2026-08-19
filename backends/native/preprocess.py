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

Graph cleanup (CSE, reinplace, view_copy collapsing) is expected to run before
lowering via the ``transform_passes`` argument of
``to_edge_transform_and_lower`` (see ``passes.get_default_passes``).
"""

from typing import final, List

from executorch.backends.native.serialization import serialize_graph

from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)

from torch._subclasses.fake_tensor import FakeTensor


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

        external_tag = None
        for spec in module_compile_spec:
            if spec.key == "external_constants_tag":
                external_tag = bytes(spec.value).decode("utf-8")

        named_data_store = NamedDataStore()
        for fqn, tensor in constant_data.items():
            if isinstance(tensor, FakeTensor):
                continue
            named_data_store.add_named_data(
                fqn,
                tensor.detach().contiguous(),
                external_tag=external_tag,
            )

        return PreprocessResult(
            processed_bytes=flatbuffer_bytes,
            data_store_output=named_data_store.get_named_data_store_output(),
        )
