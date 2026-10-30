# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
from executorch.backends.cpu.partitioner import (
    CPU_DELEGATE_VERSION,
    CPU_SUPPORTED_DTYPES,
)
from executorch.backends.native.serialization import serialize_graph
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.transforms.fuse_batch_norm_with_linear import (
    FuseBatchNormWithLinearPass,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.program._program import (
    _transform,
    _update_exported_program_graph_module,
    lift_constant_tensor_pass,
)
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_leaves, tree_map

_CONSTANT_INPUT_KINDS = (
    InputKind.PARAMETER,
    InputKind.BUFFER,
    InputKind.CONSTANT_TENSOR,
)


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


def _remove_unused_constants(edge_program: ExportedProgram) -> None:
    signature = edge_program.graph_signature
    mutated = set(signature.buffers_to_mutate.values())
    placeholders = {
        node.name: node for node in edge_program.graph.nodes if node.op == "placeholder"
    }
    kept_specs = []
    for spec in signature.input_specs:
        node = placeholders.get(spec.arg.name)
        if (
            node is not None
            and spec.kind in _CONSTANT_INPUT_KINDS
            and spec.target not in mutated
            and not node.users
        ):
            edge_program.graph.erase_node(node)
            edge_program.state_dict.pop(spec.target, None)
            edge_program.constants.pop(spec.target, None)
        else:
            kept_specs.append(spec)
    signature.input_specs = kept_specs
    edge_program.graph_module.recompile()


# Ordered passes applied to every program in preprocess, mirroring
# XNNPACKPassManager.passes. Each entry takes the current program, so later
# passes observe earlier passes' rewrites; extend by appending new pass types.
CPU_PREPROCESS_PASSES = [
    FuseBatchNormWithConvPass,
    FuseBatchNormWithLinearPass,
]


class CpuBackend(BackendDetails):
    """Serialize provider-neutral Native semantics with contiguous tensor storage.

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
        edge_program = deepcopy(edge_program)
        edge_program = _update_exported_program_graph_module(
            edge_program, edge_program.graph_module
        )
        for pass_type in CPU_PREPROCESS_PASSES:
            edge_program = _transform(edge_program, pass_type(edge_program))
        # Passes can introduce tensor attributes that must become program inputs.
        edge_program = lift_constant_tensor_pass(edge_program)
        # Fusion leaves the original weights and batch norm stats unused.
        _remove_unused_constants(edge_program)
        graph_module = edge_program.graph_module
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
            if tensor.dtype not in CPU_SUPPORTED_DTYPES or tensor.device.type != "cpu":
                raise ValueError(
                    f"CpuBackend requires supported CPU constant {name}, "
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
