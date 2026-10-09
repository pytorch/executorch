# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
from executorch.backends.cpu.partitioner import CPU_DELEGATE_VERSION
from executorch.backends.native.serialization import serialize_graph
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)
from torch.utils._pytree import tree_leaves, tree_map


def _validate_boundary_layouts(graph_module: torch.fx.GraphModule) -> None:
    for node in graph_module.graph.nodes:
        if node.op != "placeholder" and not any(
            user.op == "output" for user in node.users
        ):
            continue
        if any(
            isinstance(value, torch.Tensor) and not value.is_contiguous()
            for value in tree_leaves(node.meta.get("val"))
        ):
            raise ValueError(
                f"CpuBackend requires contiguous boundary tensor {node.name}"
            )


class CpuBackend(BackendDetails):
    """Serialize provider-neutral Native semantics with contiguous FP32 storage.

    The delegate version fixes 64-byte alignment and 64 initialized readable
    tail bytes. Named constants include the tail in their recorded buffer capacity.
    Provider choices and implementation eligibility belong to the runtime.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, module_compile_spec: list[CompileSpec]
    ) -> PreprocessResult:
        specs = [(spec.key, spec.value) for spec in module_compile_spec]
        expected_specs = [
            ("cpu_delegate_version", CPU_DELEGATE_VERSION.to_bytes(4, "little"))
        ]
        if specs != expected_specs:
            raise ValueError(
                f"CpuBackend requires delegate version {CPU_DELEGATE_VERSION}: "
                f"expected exactly {expected_specs!r}, got {specs!r}"
            )
        _validate_boundary_layouts(edge_program.graph_module)
        graph_module = deepcopy(edge_program.graph_module)
        for node in graph_module.graph.nodes:
            if "val" in node.meta:
                node.meta["val"] = tree_map(
                    lambda value: (
                        value.contiguous() if isinstance(value, torch.Tensor) else value
                    ),
                    node.meta["val"],
                )
        graph, constants = serialize_graph(
            graph_module,
            edge_program.graph_signature,
            edge_program.state_dict,
            edge_program.constants,
        )
        data = NamedDataStore()
        for name, tensor in constants.items():
            if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                raise ValueError(
                    f"CpuBackend requires FP32 CPU constant {name}, "
                    f"got dtype {tensor.dtype} on {tensor.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"CpuBackend requires contiguous constant {name}")
            data.add_named_data(
                name, tensor.detach().numpy().tobytes() + bytes(64), alignment=64
            )
        return PreprocessResult(
            processed_bytes=graph,
            data_store_output=data.get_named_data_store_output(),
        )
