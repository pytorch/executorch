# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import pytest
import torch

from executorch.backends.transforms.make_inputs_channels_last import (
    MakeInputsChannelsLast,
)
from executorch.backends.transforms.replace_channels_last_input_clones import (
    ReplaceChannelsLastInputClones,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.node import Target

_CLONE_DIM_ORDER = exir_ops.edge.dim_order_ops._clone_dim_order.default
_TO_DIM_ORDER_COPY = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
_ATEN_PERMUTE_COPY = exir_ops.edge.aten.permute_copy.default
_CHANNELS_LAST_PERMUTE_COPY = exir_ops.edge.channels_last.permute_copy.default


# ---------------------------------------------------------------------------
# Small test modules
# ---------------------------------------------------------------------------


class SingleInputConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class MultiInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.avg_pool(x)


class NonSpatialModule(torch.nn.Module):
    def forward(self, x):
        return x + x


class MixedRankModule(torch.nn.Module):
    """One 4-D input and one 3-D input."""

    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, x4d, x3d):
        y = self.avg_pool(x4d)
        y3d = y.reshape(1, 2, -1)
        return y3d + x3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_to_edge(module: torch.nn.Module, inputs: tuple) -> ExportedProgram:
    ep = torch.export.export(module.eval(), inputs)
    return to_edge(ep).exported_program()


def _find_nodes(gm: GraphModule, target: Target) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _count(gm: GraphModule, target: Target) -> int:
    return len(_find_nodes(gm, target))


def _run_pass(ep: ExportedProgram):
    result = MakeInputsChannelsLast(ep)(ep.graph_module)
    return result.graph_module, result.modified


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reseed():
    torch.manual_seed(0)
    yield


class TestMakeInputsChannelsLast:
    # ------------------------------------------------------------------
    # Basic transformation correctness
    # ------------------------------------------------------------------

    def test_single_input(self):
        example_input = torch.randn(1, 3, 8, 8)
        model = SingleInputConvModule()
        ep = _export_to_edge(model, (example_input,))

        # Before the pass the input must be contiguous.
        input_before = list(ep.graph_module.graph.nodes)[2]
        assert input_before.name == "x"
        assert input_before.meta["val"].dim_order() == (0, 1, 2, 3)
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 0

        output_before = ep.module()(example_input)

        gm, modified = _run_pass(ep)
        assert modified

        # The input must now be channels-last.
        input_after = [n for n in gm.graph.nodes if n.name == "x"][0]
        assert input_after.meta["val"].dim_order() == (0, 2, 3, 1)

        # Exactly one clone must have been inserted.
        assert len(clone_nodes := _find_nodes(gm, _CLONE_DIM_ORDER)) == 1
        assert (clone_node := clone_nodes[0]).kwargs["dim_order"] == [0, 1, 2, 3]
        assert (input_node := clone_node.args[0]).op == "placeholder"
        assert input_node.meta["val"].dim_order() == (
            0,
            2,
            3,
            1,
        ), "The input after the pass is not channels last."

        # Numerical output must be unchanged when we supply channels last data.
        output_after = gm(
            model.conv.weight,
            model.conv.bias,
            example_input.to(memory_format=torch.channels_last),
        )[0]
        assert torch.allclose(output_before, output_after)

    def test_multiple_inputs(self):
        num_inputs = 3
        example_inputs = tuple(torch.randn(1, 3, 8, 8) for _ in range(num_inputs))
        model = MultiInputModule()
        ep = _export_to_edge(model, example_inputs)

        # Before the pass, all inputs must be contiguous.
        inputs_before = [n for n in ep.graph.nodes if "inputs_" in n.name]
        for input_ in inputs_before:
            assert input_.meta["val"].dim_order() == (0, 1, 2, 3)
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 0

        output_before = ep.module()(*example_inputs)

        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, _CLONE_DIM_ORDER) == num_inputs
        # After the pass, all inputs must be channels last.
        inputs_after = [n for n in gm.graph.nodes if "inputs_" in n.name]
        for input_ in inputs_after:
            assert input_.meta["val"].dim_order() == (0, 2, 3, 1)

        channels_last_inputs = [
            i.to(memory_format=torch.channels_last) for i in example_inputs
        ]
        outputs_after = gm(*channels_last_inputs)[0]
        assert torch.allclose(output_before, outputs_after)

    def test_not_applied_to_3d_input(self):
        example_inputs = (torch.randn(2, 4, 8),)
        ep = _export_to_edge(NonSpatialModule(), example_inputs)

        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, _CLONE_DIM_ORDER) == 0

    def test_only_4d_inputs_modified_in_mixed_rank_model(self):
        example_inputs = (torch.randn(1, 3, 8, 8), torch.randn(1, 2, 6))
        ep = _export_to_edge(MixedRankModule(), example_inputs)

        gm, modified = _run_pass(ep)

        assert modified
        # Only the 4-D placeholder should be channels-last.
        assert _count(gm, _CLONE_DIM_ORDER) == 1

    def test_not_applied_to_already_channels_last_input(self):
        example_inputs = (
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),
        )
        ep = _export_to_edge(SingleInputConvModule(), example_inputs)

        input_node = [n for n in ep.graph.nodes if n.name == "x"][0]
        assert input_node.meta["val"].dim_order() == (0, 2, 3, 1)

        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, _CLONE_DIM_ORDER) == 0

    def test_idempotent(self):
        """Running the pass twice must not insert extra clones."""
        example_inputs = (torch.randn(1, 3, 8, 8),)
        ep = _export_to_edge(SingleInputConvModule(), example_inputs)

        pass_ = MakeInputsChannelsLast(ep)
        gm = ep.graph_module

        gm, modified_first = pass_(gm)
        gm, modified_second = pass_(gm)

        assert modified_first
        assert not modified_second
        assert _count(gm, _CLONE_DIM_ORDER) == 1

    def test_composed_with_replace_channels_last_input_clones(self):
        """After MakeInputsChannelsLast the ReplaceChannelsLastInputClones pass should be able to replace the inserted
        clone with a permute pair.
        """
        example_inputs = (torch.randn(1, 3, 8, 8),)
        module = SingleInputConvModule()
        ep = _export_to_edge(module, example_inputs)
        output_before = ep.module()(*example_inputs)

        gm, modified1 = _run_pass(ep)
        assert modified1
        assert _count(gm, _CLONE_DIM_ORDER) == 1

        result2 = ReplaceChannelsLastInputClones()(gm)
        gm2 = result2.graph_module
        assert result2.modified
        assert _count(gm2, _CLONE_DIM_ORDER) == 0
        assert _count(gm2, _ATEN_PERMUTE_COPY) == 1
        assert _count(gm2, _CHANNELS_LAST_PERMUTE_COPY) == 1

        channels_last_input = example_inputs[0].to(memory_format=torch.channels_last)
        output_after = gm2(module.conv.weight, module.conv.bias, channels_last_input)[0]
        assert torch.allclose(output_before, output_after)
