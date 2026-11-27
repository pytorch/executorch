# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.transforms.channels_last_ops  # noqa: F401
import pytest
import torch

from executorch.backends.transforms.enforce_contiguous_dim_order import (
    EnforceContiguousDimOrder,
)
from executorch.backends.transforms.replace_channels_last_input_clones import (
    _is_4d_channels_last,
    _is_4d_contiguous,
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


class SingleInputToDimOrderCopyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, x):
        # Expecting `x` to use the channels last memory format.
        x = x.to(memory_format=torch.contiguous_format)
        x = self.avg_pool(x)
        return x


class MultiInputToDimOrderCopyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, *inputs):
        contiguous_inputs = [
            input_.to(memory_format=torch.contiguous_format) for input_ in inputs
        ]
        x = torch.concatenate(contiguous_inputs)
        x = self.avg_pool(x)
        return x


class ToDimOrderCopyAfterAddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, x):
        x = (
            x + x
        )  # Make sure the `_to_dim_order_copy` is not consuming the model input.
        x = x.to(memory_format=torch.contiguous_format)
        x = self.avg_pool(x)
        return x


class SingleInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.relu(x)
        return x


class MultiInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.single_input_module = SingleInputModule()

    def forward(self, *inputs):
        x = torch.concatenate(inputs)
        x = self.single_input_module(x)
        return x


class IncompatibleDimOrderModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)  # Incompatible dim order.
        x = self.avg_pool(x)
        return x


class DtypeCastAndLayoutChangeModule(torch.nn.Module):
    """Module whose forward casts dtype AND changes memory format via `_to_dim_order_copy`.
    The pass must not replace such a node, since doing so would silently drop the cast.
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3)

    def forward(self, x):
        # Both a memory-format change and a dtype cast happen here.
        x = x.to(dtype=torch.float64, memory_format=torch.contiguous_format)
        x = x.to(dtype=torch.float32)  # Cast back so avg_pool accepts it.
        x = self.avg_pool(x)
        return x


def _export_to_edge(module: torch.nn.Module, inputs: tuple) -> ExportedProgram:
    ep = torch.export.export(module.eval(), inputs)
    return to_edge(ep).exported_program()


def _find_nodes(gm: GraphModule, target: Target) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _count(gm: GraphModule, target: Target) -> int:
    return len(_find_nodes(gm, target))


def _run_pass(ep_or_result) -> tuple[GraphModule, bool]:
    gm = ep_or_result.graph_module
    result = ReplaceChannelsLastInputClones()(gm)
    return result.graph_module, result.modified


def _assert_expected_result_pattern(
    input_: torch.fx.Node,
    aten_permute: torch.fx.Node,
    channels_last_permute: torch.fx.Node,
):
    assert input_.op == "placeholder"
    assert input_.meta["val"].dim_order() == (0, 2, 3, 1)
    assert aten_permute.args[0] == input_
    assert aten_permute.target == _ATEN_PERMUTE_COPY
    assert aten_permute.args[1] == [0, 2, 3, 1]
    assert aten_permute.meta["val"].dim_order() == (0, 1, 2, 3)
    assert channels_last_permute.args[0] == aten_permute
    assert channels_last_permute.target == _CHANNELS_LAST_PERMUTE_COPY
    assert channels_last_permute.args[1] == [0, 3, 1, 2]
    assert channels_last_permute.meta["val"].dim_order() == (0, 1, 2, 3)


@pytest.fixture(autouse=True)
def _reseed():
    torch.manual_seed(42)
    yield


class TestReplaceChannelsLastInputDimOrderCopies:
    """These tests use models with an explicit dim order change in their `forward()` method, which results in a
    `_to_dim_order_copy` operator in edge dialect.
    """

    def test_single_input(self):
        example_inputs = (
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),
        )

        ep = _export_to_edge(SingleInputToDimOrderCopyModule(), example_inputs)
        assert _count(ep.graph_module, _TO_DIM_ORDER_COPY) == 1
        output_before = ep.module()(*example_inputs)

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, _TO_DIM_ORDER_COPY) == 0
        assert _count(gm, _CLONE_DIM_ORDER) == 0

        nodes = list(gm.graph.nodes)
        _assert_expected_result_pattern(*nodes[:3])

        outputs_after = gm(*example_inputs)[0]
        assert torch.allclose(output_before, outputs_after)

    def test_multiple_inputs(self):
        num_inputs = 3
        example_inputs = tuple(
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last)
            for _ in range(num_inputs)
        )

        ep = _export_to_edge(MultiInputToDimOrderCopyModule(), example_inputs)
        assert _count(ep.graph_module, _TO_DIM_ORDER_COPY) == num_inputs
        output_before = ep.module()(*example_inputs)

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, _TO_DIM_ORDER_COPY) == 0
        assert _count(gm, _CLONE_DIM_ORDER) == 0

        nodes = list(gm.graph.nodes)
        for i in range(num_inputs):
            # Each input should have the expected pattern
            input_ = nodes[i]
            start_idx = num_inputs + i * (num_inputs - 1)
            end_idx = start_idx + 2
            pattern = nodes[start_idx:end_idx]
            _assert_expected_result_pattern(input_, *pattern)

        outputs_after = gm(*example_inputs)[0]
        assert torch.allclose(output_before, outputs_after)

    def test__not_applied__not_consuming_model_input(self):
        example_inputs = (
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),
        )

        ep = _export_to_edge(ToDimOrderCopyAfterAddModule(), example_inputs)
        assert _count(ep.graph_module, _TO_DIM_ORDER_COPY) == 1

        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, _TO_DIM_ORDER_COPY) == 1

    def test__not_applied__incompatible_dim_order(self):
        # This test uses a `to_dim_order_copy` which goes from contiguous to channels_last, which is not what the pass
        #  was made for.
        example_inputs = (torch.randn(1, 3, 8, 8),)

        ep = _export_to_edge(IncompatibleDimOrderModule(), example_inputs)
        assert _count(ep.graph_module, _TO_DIM_ORDER_COPY) == 1

        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, _TO_DIM_ORDER_COPY) == 1

    def test__not_applied__dtype_cast(self):
        """A `_to_dim_order_copy` that also changes dtype must not be replaced.
        Replacing it with permutes would silently drop the cast, changing the graph's
        semantics.
        """
        example_inputs = (
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),
        )

        ep = _export_to_edge(DtypeCastAndLayoutChangeModule(), example_inputs)
        # The first `_to_dim_order_copy` carries a dtype change; the pass must leave it.
        to_dim_order_copies_before = _count(ep.graph_module, _TO_DIM_ORDER_COPY)

        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, _TO_DIM_ORDER_COPY) == to_dim_order_copies_before

    @pytest.mark.parametrize(
        "extra_kwarg",
        [
            # dtype differs from the source tensor's dtype -> cast must not be silently dropped.
            {"dtype": torch.float64},
            # layout, device, and pin_memory are in _ALLOWED_KWARGS and are allowed through
            # when their value matches the source tensor (which is always the case on CPU
            # hardware). The whitelist guards against UNKNOWN kwargs; value checks guard
            # against CHANGED TensorOptions. These cases verify that a differing dtype is
            # caught; layout/device/pin_memory rejection can only be tested by providing a
            # value that actually differs from the source tensor (not possible on CPU-only).
        ],
    )
    def test__not_applied__dtype_cast_kwarg(self, extra_kwarg):
        # Verify that a `_to_dim_order_copy` carrying a dtype change is not replaced.
        # The guard must not drop the cast silently, so replacement is blocked.
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        ph.meta["val"] = torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last)
        kwargs = {"dim_order": [0, 1, 2, 3], **extra_kwarg}
        clone = g.call_function(_TO_DIM_ORDER_COPY, args=(ph,), kwargs=kwargs)
        clone.meta["val"] = torch.randn(1, 3, 8, 8)
        g.output((clone,))
        gm = torch.fx.GraphModule({}, g)

        result = ReplaceChannelsLastInputClones()(gm)

        assert not result.modified
        assert _count(result.graph_module, _TO_DIM_ORDER_COPY) == 1


class TestReplaceChannelsLastInputCloneDimOrders:
    """These tests use channels last example inputs for export and apply the `EnforceContiguousDimOrder` pass which
    inserts a `clone_dim_order` operator right after the model inputs to make the dim order contiguous. This is
    precisely the intended use-case for the `ReplaceChannelsLastInputClones`.
    """

    def test_single_input(self):
        example_inputs = (
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),
        )

        ep = _export_to_edge(SingleInputModule(), example_inputs)
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 0

        # Turn the model contiguous and create the input `clone_dim_order` operator.
        # SingleInputModule preserves channels-last format (avg_pool + relu), so ECDO inserts
        # both an input boundary clone and an output boundary clone.
        res1 = EnforceContiguousDimOrder()(ep.graph_module)
        assert res1.modified
        assert (
            _count(res1.graph_module, _CLONE_DIM_ORDER) == 2
        )  # 1 input + 1 output boundary

        output_before = ep.module()(*example_inputs)
        gm, modified = _run_pass(res1)

        assert modified
        assert _count(gm, _TO_DIM_ORDER_COPY) == 0
        # Input boundary clone is replaced by permutes; the output boundary clone remains.
        assert _count(gm, _CLONE_DIM_ORDER) == 1

        nodes = list(gm.graph.nodes)
        _assert_expected_result_pattern(*nodes[:3])

        outputs_after = gm(*example_inputs)[0]
        assert torch.allclose(output_before, outputs_after)

    def test_multi_input(self):
        num_inputs = 3
        example_inputs = tuple(
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last)
            for _ in range(num_inputs)
        )

        ep = _export_to_edge(MultiInputModule(), example_inputs)
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 0

        # Turn the model contiguous and create the input `clone_dim_order` operator.
        # MultiInputModule (avg_pool + relu) preserves channels-last, so ECDO inserts one
        # input boundary clone per input PLUS one output boundary clone.
        res1 = EnforceContiguousDimOrder()(ep.graph_module)
        assert res1.modified
        assert _count(res1.graph_module, _CLONE_DIM_ORDER) == num_inputs + 1

        output_before = ep.module()(*example_inputs)
        res2 = ReplaceChannelsLastInputClones()(res1.graph_module)

        assert res2.modified
        assert _count(res2.graph_module, _TO_DIM_ORDER_COPY) == 0
        # Input boundary clones are replaced by permutes; the output boundary clone remains.
        assert _count(res2.graph_module, _CLONE_DIM_ORDER) == 1

        nodes = list(res2.graph_module.graph.nodes)
        for i in range(num_inputs):
            # Each input should have the expected pattern.
            _assert_expected_result_pattern(*nodes[i * num_inputs : i * num_inputs + 3])

        outputs_after = res2.graph_module(*example_inputs)[0]
        assert torch.allclose(output_before, outputs_after)

    def test_idempotency(self):
        """Running the pass twice must not alter the graph on the second run."""
        example_inputs = (
            torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),
        )

        ep = _export_to_edge(SingleInputModule(), example_inputs)
        res1 = EnforceContiguousDimOrder()(ep.graph_module)
        assert res1.modified

        res2 = ReplaceChannelsLastInputClones()(res1.graph_module)
        assert res2.modified

        # Second pass: input boundary clones are gone; only the output boundary clone remains.
        res3 = ReplaceChannelsLastInputClones()(res2.graph_module)
        assert not res3.modified
        # Input boundary clones have been replaced; the output boundary clone is untouched.
        assert _count(res3.graph_module, _CLONE_DIM_ORDER) == 1
        assert _count(res3.graph_module, _TO_DIM_ORDER_COPY) == 0
        assert _count(res3.graph_module, _ATEN_PERMUTE_COPY) == _count(
            res2.graph_module, _ATEN_PERMUTE_COPY
        )
        assert _count(res3.graph_module, _CHANNELS_LAST_PERMUTE_COPY) == _count(
            res2.graph_module, _CHANNELS_LAST_PERMUTE_COPY
        )


class TestGuardPredicates:
    """Unit tests for the `call_operator` guard conditions using hand-built graphs."""

    def test_is_4d_channels_last(self):
        assert _is_4d_channels_last([0, 2, 3, 1])
        assert not _is_4d_channels_last([0, 1, 2, 3])
        assert not _is_4d_channels_last([0, 2, 1])
        assert not _is_4d_channels_last([0, 2, 3, 4, 1])

    def test_is_4d_contiguous(self):
        assert _is_4d_contiguous([0, 1, 2, 3])
        assert not _is_4d_contiguous([0, 2, 3, 1])
        assert not _is_4d_contiguous([0, 1, 2])
        assert not _is_4d_contiguous([0, 1, 2, 3, 4])
