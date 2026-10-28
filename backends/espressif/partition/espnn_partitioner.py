# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable, Optional

import torch

from executorch.backends.espressif.espnn_preprocess import (
    EspNNBackend,
    INT8_FORMAT_SPEC,
)
from executorch.backends.espressif.serialization.int8_ops import (
    consumers,
    is_dequantize,
    is_movement_terminal,
    is_quantize,
    MOVEMENT_KINDS,
    node_kind,
    SPLIT_KINDS,
)
from executorch.backends.espressif.serialization.xts3 import (
    _target_kind,
    ExportedTensorResolver,
    INT8_PASSTHROUGH_MARKERS,
    lower_int8_node,
    LUT_KINDS,
    TensorRegistry,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.utils.fuser_utils import validate_partition


SUPPORTED_KINDS = {
    "conv2d",
    "convolution",
    "conv_transpose2d",
    "hardtanh",
    "hardtanh_",
    "relu",
    "relu6",
    "add",
    "adaptive_avg_pool2d",
    "mean_dim",
    "max_pool2d",
    "max_pool2d_with_indices",
    "avg_pool2d",
    "cat",
    "stack",
    "flatten",
    "reshape",
    "linear",
    "identity",
    "dropout",
    "getitem",
    "mul",
    "softmax",
    "pad",
    "upsample_nearest2d",
    "upsample_nearest2d_vec",
    "upsample_bilinear2d",
    "upsample_bilinear2d_vec",
    *MOVEMENT_KINDS,
    *SPLIT_KINDS,
    *LUT_KINDS,
}

PRESERVED_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.upsample_nearest2d.vec,
    torch.ops.aten.upsample_bilinear2d.vec,
]


def is_supported_int8_node(node) -> bool:
    if node.op != "call_function":
        return False
    text = str(node.target)
    if any(marker in text for marker in INT8_PASSTHROUGH_MARKERS):
        return True
    return _target_kind(node.target) in SUPPORTED_KINDS


def _keep_if_quantized(node: torch.fx.Node) -> bool:
    """Only quantized instances stay whole; float ones decompose for the CPU."""
    tensor_inputs = [arg for arg in node.args if isinstance(arg, torch.fx.Node)]
    if not tensor_inputs or not is_dequantize(tensor_inputs[0]):
        return False
    if not any(is_quantize(user) for user in node.users) and not any(
        _target_kind(user.target) in ("relu", "relu6", "hardtanh", "hardtanh_")
        and any(is_quantize(u) for u in user.users)
        for user in node.users
        if user.op == "call_function"
    ):
        return False
    if _target_kind(node.target) == "linear":
        value = tensor_inputs[0].meta.get("val")
        if value is None or value.dim() == 4 or value.dim() < 1:
            return False
    return True


def _operator_name(target) -> str:
    import re

    match = re.search(
        r"([A-Za-z_]+(?:\.[A-Za-z0-9_]+){1,2})(?=[>:\s]|$)",
        str(target).replace("<EdgeOpOverload: ", ""),
    )
    return match.group(1) if match else str(target)


def _is_passthrough(node) -> bool:
    return node.op == "call_function" and any(
        marker in str(node.target) for marker in INT8_PASSTHROUGH_MARKERS
    )


class StructuralResolver(ExportedTensorResolver):
    """
    ExecuTorch partitions a copy of the program whose constants are
    FakeTensors. Support is judged on shapes, arguments, and quantization
    parameters, so fake constants are replaced by neutral stand-ins of the
    same shape and dtype (scales of one, zero points and weights of zero).
    """

    def resolve(self, node):
        from torch._subclasses.fake_tensor import FakeTensor

        value = super().resolve(node)
        if not isinstance(value, FakeTensor):
            return value
        shape = tuple(int(d) for d in value.shape)
        if value.dtype.is_floating_point:
            return torch.ones(shape, dtype=value.dtype)
        return torch.zeros(shape, dtype=value.dtype)


def find_supported_nodes(exported_program: ExportedProgram) -> tuple[set, dict]:
    graph_module = exported_program.graph_module
    registry = TensorRegistry(StructuralResolver(exported_program), graph_module)
    supported = set()
    reasons = {}
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_passthrough(node):
            supported.add(node)
            continue
        if _target_kind(node.target) not in SUPPORTED_KINDS:
            reasons[node] = "no int8 lowering for this operator"
            continue
        fused_before = set(registry.fused_nodes)
        try:
            registry.extra_operations.clear()
            lowered = lower_int8_node(node, registry)
            if lowered is not None or node in registry.fused_nodes:
                supported.add(node)
            else:
                reasons[node] = "operator was not lowered"
        except Exception as error:
            registry.fused_nodes = fused_before
            reasons[node] = str(error).splitlines()[0][:160]
    registry.extra_operations.clear()

    changed = True
    while changed:
        changed = False
        for node in list(supported):
            kind = node_kind(node)
            deferred = (
                kind in MOVEMENT_KINDS and not is_movement_terminal(node)
            ) or kind in SPLIT_KINDS
            if not deferred:
                continue
            downstream = (
                consumers(node) if kind not in SPLIT_KINDS else list(node.users)
            )
            if any(user not in supported for user in downstream):
                supported.discard(node)
                reasons[node] = "data movement feeds an operator that runs on the CPU"
                changed = True
    return supported, reasons


class _SupportSet(OperatorSupportBase):
    def __init__(self, nodes: set):
        self.nodes = nodes

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node in self.nodes


def _is_constant(node, resolver: ExportedTensorResolver) -> bool:
    return node.op in ("placeholder", "get_attr") and resolver.resolve(node) is not None


def _boundary_is_valid(nodes: set, resolver: ExportedTensorResolver) -> bool:
    for node in nodes:
        for user in node.users:
            if user not in nodes and not (is_quantize(node) or is_dequantize(node)):
                return False
        for source in node.all_input_nodes:
            if source in nodes or _is_constant(source, resolver):
                continue
            if not (is_quantize(node) or is_dequantize(node)):
                return False
    return True


FUSABLE_ACTIVATION_KINDS = ("relu", "relu6", "hardtanh", "hardtanh_")


def _attach_outputs_to_producers(partitions: list[set]) -> None:
    """
    A quantize node (and an activation fused into its producer) must live in
    the producer's partition so the partition exports int8 or dequantized
    values only, never a raw accumulator. A move that would make either
    partition non-convex is skipped; the affected partition then fails
    _boundary_is_valid and stays on the CPU instead of breaking fusion.
    """
    owner = {}
    for index, nodes in enumerate(partitions):
        for node in nodes:
            owner[node] = index
    changed = True
    while changed:
        changed = False
        for index, nodes in enumerate(partitions):
            for node in list(nodes):
                if _is_passthrough(node):
                    continue
                for user in node.users:
                    attach = is_quantize(user) or (
                        user.op == "call_function"
                        and _target_kind(user.target) in FUSABLE_ACTIVATION_KINDS
                        and user in owner
                    )
                    if not attach or owner.get(user) == index or user not in owner:
                        continue
                    donor = partitions[owner[user]]
                    if not validate_partition(
                        list(nodes | {user})
                    ) or not validate_partition(list(donor - {user})):
                        continue
                    donor.discard(user)
                    nodes.add(user)
                    owner[user] = index
                    changed = True


def _has_compute(nodes: set) -> bool:
    return any(not _is_passthrough(node) for node in nodes)


class EspNNPartitioner(Partitioner):
    """
    Delegates PT2E-quantized regions of a graph to the ESP32 ESP-NN backend.

    With allow_cpu_fallback (the default) operators the int8 backend cannot
    lower stay in the ExecuTorch program and run on the CPU through the
    portable kernels; partition boundaries carry float or int8 tensors.
    With allow_cpu_fallback=False any operator left outside a delegate is an
    error, whether it is unsupported or its region could not be delegated;
    the standalone XTS3 firmware path requires this.
    """

    def __init__(
        self,
        compile_specs: Optional[list[CompileSpec]] = None,
        allow_cpu_fallback: bool = True,
    ):
        specs = list(compile_specs or [])
        if not any(spec.key == INT8_FORMAT_SPEC.key for spec in specs):
            specs.append(INT8_FORMAT_SPEC)
        self.delegation_spec = DelegationSpec(EspNNBackend.__name__, specs)
        self.allow_cpu_fallback = allow_cpu_fallback
        self.report: dict = {}

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> tuple[list[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return PRESERVED_OPS, _keep_if_quantized

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        graph_module = exported_program.graph_module
        with unset_fake_temporarily():
            supported, reasons = find_supported_nodes(exported_program)
        unsupported = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node not in supported
        ]
        if unsupported and not self.allow_cpu_fallback:
            raise RuntimeError(
                "The ESP32 int8 backend cannot run these operators: "
                + ", ".join(
                    f"{node.name} ({node.target}): {reasons.get(node, '')}"
                    for node in unsupported
                )
            )

        with unset_fake_temporarily():
            resolver = StructuralResolver(exported_program)
        proposer = CapabilityBasedPartitioner(
            graph_module, _SupportSet(supported), allows_single_node_partition=True
        )
        partitions = [
            set(partition.nodes) for partition in proposer.propose_partitions()
        ]
        _attach_outputs_to_producers(partitions)
        tags = {}
        rejected = 0
        for nodes in partitions:
            if not _has_compute(nodes):
                continue
            with unset_fake_temporarily():
                valid = _boundary_is_valid(nodes, resolver)
            if not valid:
                rejected += 1
                continue
            tag = f"espnn_{len(tags)}"
            for node in nodes:
                node.meta["delegation_tag"] = tag
            tags[tag] = self.delegation_spec
        delegated = {
            node
            for node in graph_module.graph.nodes
            if "delegation_tag" in node.meta and node.op == "call_function"
        }
        cpu_ops = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node not in delegated
            and not _is_passthrough(node)
        ]
        leftover = [node for node in cpu_ops if "getitem" not in str(node.target)]
        if leftover and not self.allow_cpu_fallback:
            raise RuntimeError(
                f"{len(leftover)} operator(s) would run on the CPU "
                f"({rejected} region(s) could not be delegated): "
                + ", ".join(
                    f"{node.name} ({_operator_name(node.target)}): "
                    f"{reasons.get(node, 'outside the delegated regions')}"
                    for node in leftover[:20]
                )
            )
        if not tags and not self.allow_cpu_fallback:
            raise RuntimeError("No ESP32 int8 partition could be formed")
        tag_constant_data(exported_program)
        self.report = {
            "partitions": len(tags),
            "rejected_partitions": rejected,
            "delegated_ops": sum(1 for node in delegated if not _is_passthrough(node)),
            "cpu_ops": [
                {
                    "name": node.name,
                    "target": _operator_name(node.target),
                    "reason": reasons.get(node, "outside the delegated regions"),
                }
                for node in leftover
            ],
        }
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=tags
        )


__all__ = [
    "StructuralResolver",
    "EspNNPartitioner",
    "find_supported_nodes",
    "is_supported_int8_node",
]
