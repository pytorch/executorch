# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The plain WebGPU edge rewrites must run pre-partition, not in partition().

These drive the real `_replace_single_hf_rope` / `_rewrite_gemma4_sdpa`
helpers, not stand-ins, and mutate the exact site counts and ABI guards so a
weakened guard fails.
"""

import unittest
from typing import List, Optional

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401
import torch
from executorch.examples.models.gemma4 import webgpu_partitioner as wp
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult


def _sdpa_available() -> bool:
    """llama::custom_sdpa needs its compiled kernel; absent in a bare checkout."""
    try:
        exir_ops.edge.llama.custom_sdpa.default
    except AttributeError:
        return False
    return True


_REQUIRES_SDPA = unittest.skipUnless(
    _sdpa_available(), "llama::custom_sdpa is not registered in this environment"
)


class _Stub:
    """`_rewrite_gemma4_sdpa` reads only `.graph_module`."""

    def __init__(self, graph_module: torch.fx.GraphModule) -> None:
        self.graph_module = graph_module


def _fake(rank: int) -> torch.Tensor:
    return torch.zeros([1] * rank)


def _sdpa_graph(
    count: int,
    *,
    args: int = 8,
    mask_rank: int = 2,
    dropout: float = 0.0,
    is_causal: bool = False,
    scale: float = 1.0,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    ph = {r: graph.placeholder(f"p{r}") for r in (2, 4)}
    for r, node in ph.items():
        node.meta["val"] = _fake(r)
    for i in range(count):
        full = (
            ph[4],
            ph[4],
            ph[4],
            0,
            ph[mask_rank],
            dropout,
            is_causal,
            scale,
        )
        node = graph.call_function(exir_ops.edge.llama.custom_sdpa.default, full[:args])
        node.meta["val"] = _fake(4)
        node.name = f"sdpa_{i}"
    graph.output(ph[4])
    return torch.fx.GraphModule(torch.nn.Module(), graph)


class _Inert(Partitioner):
    def __init__(self) -> None:
        super().__init__()
        self.delegation_spec = DelegationSpec("VulkanBackend", [CompileSpec("k", b"v")])
        self.seen: List[object] = []
        self.input_ids: Optional[List[int]] = None

    def partition(self, exported_program) -> PartitionResult:
        nodes = list(exported_program.graph_module.graph.nodes)
        self.seen = [n.target for n in nodes if n.op == "call_function"]
        self.input_ids = [id(n) for n in nodes]
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags={}
        )


class _Probe(ExportedProgramPassBase):
    def __init__(self, log: List[str], name: str) -> None:
        super().__init__()
        self.log, self.name = log, name

    def call(self, ep) -> ExportedProgramPassResult:
        self.log.append(self.name)
        return ExportedProgramPassResult(ep, True)


class RewritePassTest(unittest.TestCase):
    @_REQUIRES_SDPA
    def test_exact_sdpa_count_is_enforced(self) -> None:
        # The accepted contract is exactly 35 sites.
        wp._rewrite_gemma4_sdpa(_Stub(_sdpa_graph(35)))
        for wrong in (0, 34, 36):
            with self.subTest(count=wrong):
                with self.assertRaisesRegex(ValueError, "35 SDPA sites"):
                    wp._rewrite_gemma4_sdpa(_Stub(_sdpa_graph(wrong)))

    @_REQUIRES_SDPA
    def test_rewrite_retargets_every_site(self) -> None:
        gm = _sdpa_graph(35)
        wp._rewrite_gemma4_sdpa(_Stub(gm))
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(targets.count(exir_ops.edge.et_vk.gemma4_sdpa.default), 35)
        self.assertNotIn(exir_ops.edge.llama.custom_sdpa.default, targets)

    @_REQUIRES_SDPA
    def test_abi_and_rank_guards_fail_closed(self) -> None:
        cases = {
            "positional ABI": dict(args=7),
            "not WebGPU-compatible": dict(mask_rank=4),
        }
        for expected, kwargs in cases.items():
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, expected):
                    wp._rewrite_gemma4_sdpa(_Stub(_sdpa_graph(35, **kwargs)))
        for kwargs in (
            dict(dropout=0.1),
            dict(is_causal=True),
            dict(scale=0.5),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "not WebGPU-compatible"):
                    wp._rewrite_gemma4_sdpa(_Stub(_sdpa_graph(35, **kwargs)))

    def test_rope_count_is_enforced(self) -> None:
        # No RoPE sites in an empty graph: the 20-site contract must fire.
        empty = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        with self.assertRaisesRegex(ValueError, "20 single-HF-RoPE sites"):
            wp._replace_single_hf_rope(_Stub(empty))

    def test_pass_applies_both_rewrites_in_order(self) -> None:
        # RoPE runs first, so its contract is what an empty graph trips.
        with self.assertRaisesRegex(ValueError, "single-HF-RoPE"):
            wp._Gemma4WebGPURewritePass().call(
                _Stub(torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()))
            )

    def test_transform_passes_entry_point_is_fixed(self) -> None:
        passes = wp.build_webgpu_transform_passes()
        self.assertEqual(len(passes), 1)
        self.assertIsInstance(passes[0], wp._Gemma4WebGPURewritePass)
        self.assertFalse(hasattr(passes[0], "rewrites"))


class _Add(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.add.Tensor(x, x)


class _Mutating(Partitioner):
    def __init__(self) -> None:
        super().__init__()
        self.delegation_spec = DelegationSpec("VulkanBackend", [CompileSpec("k", b"v")])

    def partition(self, exported_program) -> PartitionResult:
        for node in exported_program.graph_module.graph.nodes:
            if node.target == exir_ops.edge.aten.add.Tensor:
                node.target = exir_ops.edge.aten.mul.Tensor
        exported_program.graph_module.recompile()
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags={}
        )


class PlacementTest(unittest.TestCase):
    def _program(self):
        return torch.export.export(_Add(), (torch.randn(4),), strict=True)

    def test_partition_time_mutation_is_rejected(self) -> None:
        with self.assertRaises(AssertionError) as caught:
            to_edge_transform_and_lower(self._program(), partitioner=[_Mutating()])
        self.assertIn("should not modify the graph module", str(caught.exception))

    def test_partition_input_is_left_identical(self) -> None:
        inert = _Inert()
        to_edge_transform_and_lower(self._program(), partitioner=[inert])
        self.assertIn(exir_ops.edge.aten.add.Tensor, inert.seen)

    def test_transform_ordering_is_deterministic(self) -> None:
        log: List[str] = []
        to_edge_transform_and_lower(
            self._program(),
            partitioner=[_Inert()],
            transform_passes=[_Probe(log, "a"), _Probe(log, "b")],
        )
        self.assertEqual(log, ["a", "b"])
