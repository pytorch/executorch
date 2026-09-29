# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten._to_copy.default` modules for the WebGPU op-test framework.

`ToCopyIntToFloatModule` / `ToCopyFloatModule` drive the export-delegation smoke
test (mirroring `test_view_copy.py`). The int32 -> fp32 numeric convert — input
int `[1, 2, 3]` -> float `[1.0, 2.0, 3.0]`, NOT the byte-reinterpretation
`0x1 -> 1.4e-45` — and the fp32 same-dtype passthrough are value-checked by the
lvp golden (yolo11n / Depth-Anything-V2).
"""

import math
import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import get_lowered_submodules


class ToCopyIntToFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # int32 -> fp32 dtype cast (the numeric-convert path).
        return x.to(torch.float32)


class ToCopyFloatToIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int32)


class ToCopyFloatToIntToFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int32).to(torch.float32)


class ToCopyFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same-dtype copy (flat byte-copy path); copy=True keeps the op from
        # being elided as a no-op.
        return x.to(torch.float32, copy=True)


class ToCopyBoolToFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class ToCopyInt8ToFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CompareToCopyBoolToFloatModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a > b).to(torch.float32)


def to_copy_int_input(shape: tuple[int, ...]) -> torch.Tensor:
    n = math.prod(shape)
    return (torch.arange(n, dtype=torch.int32) - n // 2).reshape(shape)


def to_copy_float_input(shape: tuple[int, ...]) -> torch.Tensor:
    n = math.prod(shape)
    pattern = torch.tensor(
        [-8.75, -3.0, -1.5, -0.25, 0.0, 0.25, 1.5, 3.0, 8.75],
        dtype=torch.float32,
    )
    repeats = (n + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[:n].reshape(shape)


def bool_tail_input(shape: tuple[int, ...]) -> torch.Tensor:
    n = math.prod(shape)
    pattern = torch.tensor([True, False, True, True, False, False, True])
    repeats = (n + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[:n].reshape(shape)


def compare_to_copy_input_a(shape: tuple[int, ...]) -> torch.Tensor:
    n = math.prod(shape)
    pattern = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0])
    repeats = (n + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[:n].reshape(shape)


def compare_to_copy_input_b(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def _lower(model: torch.nn.Module, *inputs: torch.Tensor):
    ep = torch.export.export(model.eval(), inputs)
    edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    return ep, edge


def _export(model: torch.nn.Module, *inputs: torch.Tensor):
    _, edge = _lower(model, *inputs)
    return edge.to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _prepartition_cast_dtypes(ep) -> list[torch.dtype]:
    return [
        node.args[1]
        for node in ep.graph_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.to.dtype
    ]


def _delegated_cast_dtypes(edge) -> list[torch.dtype]:
    graph_module = edge.exported_program().graph_module
    if any(
        "_to_dim_order_copy" in str(getattr(node, "target", ""))
        for node in graph_module.graph.nodes
    ):
        return []
    return [
        node.kwargs["dtype"]
        for _, lowered, _ in get_lowered_submodules(graph_module)
        for node in lowered.original_module.graph_module.graph.nodes
        if "_to_dim_order_copy" in str(getattr(node, "target", ""))
    ]


class ToCopyTest(unittest.TestCase):
    def test_int_to_float_delegates(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        et = _export(ToCopyIntToFloatModule(), x)
        self.assertTrue(
            _delegated(et), "Expected a VulkanBackend delegate (to_copy int->float)"
        )

    def test_float_to_int_delegates(self) -> None:
        x = torch.tensor([-3.75, -1.0, 0.0, 1.9, 63.0], dtype=torch.float32)
        et = _export(ToCopyFloatToIntModule(), x)
        self.assertTrue(
            _delegated(et), "Expected a VulkanBackend delegate (to_copy float->int)"
        )

    def test_roundtrip_keeps_both_casts_in_delegate(self) -> None:
        x = torch.tensor([-3.75, -1.0, 0.0, 1.9, 63.0], dtype=torch.float32)
        ep, edge = _lower(ToCopyFloatToIntToFloatModule(), x)
        expected = [torch.int32, torch.float32]
        self.assertEqual(_prepartition_cast_dtypes(ep), expected)
        self.assertEqual(_delegated_cast_dtypes(edge), expected)
        self.assertTrue(_delegated(edge.to_executorch()))

        for module, one_direction_input in (
            (ToCopyFloatToIntModule(), x),
            (
                ToCopyIntToFloatModule(),
                torch.tensor([-3, -1, 0, 1, 63], dtype=torch.int32),
            ),
        ):
            _, one_direction_edge = _lower(module, one_direction_input)
            self.assertNotEqual(_delegated_cast_dtypes(one_direction_edge), expected)

    def test_float_passthrough_delegates(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        et = _export(ToCopyFloatModule(), x)
        self.assertTrue(
            _delegated(et), "Expected a VulkanBackend delegate (to_copy float->float)"
        )

    def test_bool_to_float_delegates(self) -> None:
        x = bool_tail_input((5,))
        ep, edge = _lower(ToCopyBoolToFloatModule(), x)
        self.assertEqual(_prepartition_cast_dtypes(ep), [torch.float32])
        self.assertEqual(_delegated_cast_dtypes(edge), [torch.float32])
        self.assertTrue(_delegated(edge.to_executorch()))

    def test_compare_bool_to_float_delegates(self) -> None:
        a = compare_to_copy_input_a((5,))
        b = compare_to_copy_input_b((5,))
        ep, edge = _lower(CompareToCopyBoolToFloatModule(), a, b)
        self.assertEqual(_prepartition_cast_dtypes(ep), [torch.float32])
        self.assertEqual(_delegated_cast_dtypes(edge), [torch.float32])
        self.assertTrue(_delegated(edge.to_executorch()))

    def test_int8_to_float_does_not_delegate(self) -> None:
        x = torch.tensor([-2, 0, 3], dtype=torch.int8)
        self.assertFalse(_delegated(_export(ToCopyInt8ToFloatModule(), x)))
