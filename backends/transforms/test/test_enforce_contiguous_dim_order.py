# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.transforms.enforce_contiguous_dim_order import (
    _is_contiguous,
    _node_is_input_boundary_clone,
    _node_is_output_boundary_clone,
    EnforceContiguousDimOrder,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.node import Target


_CLONE_DIM_ORDER = exir_ops.edge.dim_order_ops._clone_dim_order.default
_EMPTY_DIM_ORDER = exir_ops.edge.dim_order_ops._empty_dim_order.default

_NCHW = torch.randn(1, 4, 8, 8)
_NHWC = torch.randn(1, 4, 8, 8).to(memory_format=torch.channels_last)

_DIM_ORDER_CONTIGUOUS = (0, 1, 2, 3)
_DIM_ORDER_CHANNELS_LAST = (0, 2, 3, 1)


class AddThenContiguousModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + x).contiguous()


class ChannelsLastThenContiguousCloneModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone(memory_format=torch.channels_last).clone(
            memory_format=torch.contiguous_format
        )


class ChannelsLastCloneThenReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.clone(memory_format=torch.channels_last))


class ChannelsLastCloneModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone(memory_format=torch.channels_last)


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class ChainedChannelsLastCloneModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone(memory_format=torch.channels_last).clone(
            memory_format=torch.channels_last
        )


class EmptyDimOrderModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, memory_format=torch.channels_last) + x


class CifarNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))


def _export_to_edge(module: torch.nn.Module, inputs: tuple) -> ExportedProgram:
    return to_edge(torch.export.export(module.eval(), inputs)).exported_program()


def _run_pass(ep: ExportedProgram) -> tuple[GraphModule, bool]:
    result = EnforceContiguousDimOrder()(ep.graph_module)
    return result.graph_module, result.modified


def _find(gm: GraphModule, target: Target) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _count(gm: GraphModule, target: Target) -> int:
    return len(_find(gm, target))


def _input_dim_order(gm: GraphModule) -> tuple[int, ...]:
    """Return the dim_order of the first placeholder's meta['val']."""
    ph = next(n for n in gm.graph.nodes if n.op == "placeholder")
    return tuple(ph.meta["val"].dim_order())


def _output_dim_order(gm: GraphModule) -> tuple[int, ...]:
    """Return the dim_order of the first return value's meta['val']."""
    out = next(n for n in gm.graph.nodes if n.op == "output")
    return tuple(out.args[0][0].meta["val"].dim_order())


class TestEnforceContiguousDimOrderPass:

    def test_nhwc_input_boundary_clone_inserted(self):
        """Non-contiguous placeholder gets an input boundary clone; the existing
        internal clone is replaced rather than duplicated."""
        ep = _export_to_edge(AddThenContiguousModule(), (_NHWC,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 1

        gm, modified = _run_pass(ep)

        assert modified
        clones = _find(gm, _CLONE_DIM_ORDER)
        assert len(clones) == 1
        assert _node_is_input_boundary_clone(clones[0])
        assert clones[0].args[0].op == "placeholder"
        assert clones[0].meta["val"].is_contiguous()
        assert _input_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        assert _output_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        gm.graph.lint()

    def test_chained_internal_clones_removed(self):
        ep = _export_to_edge(ChannelsLastThenContiguousCloneModule(), (_NCHW,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 2

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, _CLONE_DIM_ORDER) == 0
        assert _input_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        assert _output_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        gm.graph.lint()

    def test_internal_clone_removed_output_boundary_inserted(self):
        """Internal channels-last clone is removed; an output boundary clone is
        inserted to restore the non-contiguous output dim order."""
        ep = _export_to_edge(ChannelsLastCloneThenReluModule(), (_NCHW,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 1

        gm, modified = _run_pass(ep)

        assert modified
        clones = _find(gm, _CLONE_DIM_ORDER)
        assert len(clones) == 1
        assert _node_is_output_boundary_clone(clones[0])
        assert _input_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        assert _output_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        gm.graph.lint()

    def test_already_normalized_output_boundary_not_modified(self):
        """A graph already consisting solely of an output boundary clone must
        not be modified (modified=False, clone count unchanged)."""
        ep = _export_to_edge(ChannelsLastCloneModule(), (_NCHW,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 1

        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, _CLONE_DIM_ORDER) == 1
        assert _input_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        assert _output_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        gm.graph.lint()

    def test_nhwc_io_boundary_clones_inserted(self):
        """Non-contiguous input and non-contiguous output both require boundary
        clones; all interior nodes must have contiguous meta['val']."""
        ep = _export_to_edge(AddModule(), (_NHWC,))

        gm, modified = _run_pass(ep)

        assert modified
        clones = _find(gm, _CLONE_DIM_ORDER)
        assert len(clones) == 2
        assert sum(1 for n in clones if _node_is_input_boundary_clone(n)) == 1
        assert sum(1 for n in clones if _node_is_output_boundary_clone(n)) == 1
        for node in gm.graph.nodes:
            if node.op == "placeholder" or _node_is_output_boundary_clone(node):
                continue
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                assert (
                    val.is_contiguous()
                ), f"Node {node.name!r} has non-contiguous meta['val']"
        assert _input_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        assert _output_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        gm.graph.lint()

    def test_chained_nhwc_output_clones_no_duplicate_boundary(self):
        """When chained non-contiguous clones appear at the output, the first is
        removed (non-contiguous source) and the second becomes the output boundary
        clone. Step 4 must not insert a duplicate on top of it."""
        ep = _export_to_edge(ChainedChannelsLastCloneModule(), (_NCHW,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 2

        gm, modified = _run_pass(ep)

        assert modified
        clones = _find(gm, _CLONE_DIM_ORDER)
        assert len(clones) == 1
        assert _node_is_output_boundary_clone(clones[0])
        assert clones[0].args[0].op == "placeholder"
        assert _input_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        assert _output_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        gm.graph.lint()

    def test_empty_dim_order_kwarg_rewritten_to_contiguous(self):
        """The channels-last empty allocation becomes contiguous internally; the
        channels-last output dim order is preserved via an output boundary clone."""
        ep = _export_to_edge(EmptyDimOrderModule(), (_NCHW,))
        empty_before = _find(ep.graph_module, _EMPTY_DIM_ORDER)
        assert len(empty_before) == 1
        assert not _is_contiguous(empty_before[0].kwargs["dim_order"])

        gm, modified = _run_pass(ep)

        assert modified
        empty_after = _find(gm, _EMPTY_DIM_ORDER)
        assert len(empty_after) == 1
        assert _is_contiguous(empty_after[0].kwargs["dim_order"])
        clones = _find(gm, _CLONE_DIM_ORDER)
        assert len(clones) == 1
        assert _node_is_output_boundary_clone(clones[0])
        assert _input_dim_order(gm) == _DIM_ORDER_CONTIGUOUS
        assert _output_dim_order(gm) == _DIM_ORDER_CHANNELS_LAST
        gm.graph.lint()

    @pytest.mark.parametrize(
        "module,inp,seed",
        [
            (AddThenContiguousModule(), _NHWC, 0),
            (ChannelsLastThenContiguousCloneModule(), _NCHW, 1),
            (ChannelsLastCloneThenReluModule(), _NCHW, 2),
            (ChannelsLastCloneModule(), _NCHW, 3),
            (AddModule(), _NHWC, 4),
            (ChainedChannelsLastCloneModule(), _NCHW, 5),
        ],
    )
    def test_numerical_correctness(self, module, inp, seed):
        torch.manual_seed(seed)
        x = inp.clone()
        reference = module(x)
        ep = _export_to_edge(module, (x,))
        _run_pass(ep)
        out = ep.module()(x)[0]
        assert torch.allclose(out.reshape(reference.shape), reference, atol=1e-5)

    @pytest.mark.parametrize(
        "module,inp",
        [
            (AddThenContiguousModule(), _NHWC),
            (ChannelsLastThenContiguousCloneModule(), _NCHW),
            (ChannelsLastCloneThenReluModule(), _NCHW),
            (ChannelsLastCloneModule(), _NCHW),
            (AddModule(), _NHWC),
            (ChainedChannelsLastCloneModule(), _NCHW),
        ],
    )
    def test_pass_is_idempotent(self, module, inp):
        ep = _export_to_edge(module, (inp,))
        gm, _ = _run_pass(ep)
        clone_count = _count(gm, _CLONE_DIM_ORDER)
        result2 = EnforceContiguousDimOrder()(gm)
        assert not result2.modified
        assert _count(result2.graph_module, _CLONE_DIM_ORDER) == clone_count


class TestCifarNet:

    def test_cifarnet_nhwc_input(self):
        """Exactly one input boundary clone after the NHWC placeholder; no other
        dim-order ops remain; all non-placeholder nodes have contiguous meta['val'];
        output values match the NCHW baseline."""
        torch.manual_seed(42)
        model = CifarNet().eval()
        x_nchw = torch.randn(1, 3, 32, 32)
        x_nhwc = x_nchw.to(memory_format=torch.channels_last)

        ep = _export_to_edge(model, (x_nhwc,))
        gm, modified = _run_pass(ep)

        assert modified
        clones = _find(gm, _CLONE_DIM_ORDER)
        assert len(clones) == 1
        assert _node_is_input_boundary_clone(clones[0])

        non_contiguous = [
            n
            for n in gm.graph.nodes
            if n.op != "placeholder"
            and isinstance(n.meta.get("val"), torch.Tensor)
            and not n.meta["val"].is_contiguous()
        ]
        assert non_contiguous == []
        gm.graph.lint()

        reference = model(x_nchw).flatten()
        ep_nhwc = _export_to_edge(model, (x_nhwc,))
        _run_pass(ep_nhwc)
        out = ep_nhwc.module()(x_nhwc)[0].flatten()
        assert torch.allclose(out, reference, atol=1e-5)


class TestBoundaryClonePredicates:

    def test_input_boundary_clone_requires_contiguous_dim_order(self):
        """Clone on non-contiguous placeholder with non-contiguous dim_order is not
        an input boundary clone -- it does not normalize the input."""
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        ph.meta["val"] = _NHWC.clone()
        clone = g.call_function(
            _CLONE_DIM_ORDER, args=(ph,), kwargs={"dim_order": [0, 2, 3, 1]}
        )
        clone.meta["val"] = _NHWC.clone()
        g.output((clone,))
        assert not _node_is_input_boundary_clone(clone)

    def test_input_boundary_clone_accepted(self):
        """Clone on non-contiguous placeholder with contiguous dim_order is valid."""
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        ph.meta["val"] = _NHWC.clone()
        clone = g.call_function(
            _CLONE_DIM_ORDER, args=(ph,), kwargs={"dim_order": [0, 1, 2, 3]}
        )
        clone.meta["val"] = _NCHW.clone()
        g.output((clone,))
        assert _node_is_input_boundary_clone(clone)

    def test_output_boundary_clone_requires_contiguous_source(self):
        """Clone with non-contiguous source and non-contiguous dim_order is not an
        output boundary clone."""
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        ph.meta["val"] = _NHWC.clone()
        clone = g.call_function(
            _CLONE_DIM_ORDER, args=(ph,), kwargs={"dim_order": [0, 2, 3, 1]}
        )
        clone.meta["val"] = _NHWC.clone()
        g.output((clone,))
        assert not _node_is_output_boundary_clone(clone)

    def test_output_boundary_clone_requires_only_output_users(self):
        """Clone that also feeds an internal op must not be an output boundary clone --
        the internal consumer would receive a non-contiguous tensor."""
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        ph.meta["val"] = _NCHW.clone()
        clone = g.call_function(
            _CLONE_DIM_ORDER, args=(ph,), kwargs={"dim_order": [0, 2, 3, 1]}
        )
        clone.meta["val"] = _NHWC.clone()
        relu = g.call_function(torch.ops.aten.relu.default, args=(clone,))
        relu.meta["val"] = _NHWC.clone()
        g.output((clone, relu))
        assert not _node_is_output_boundary_clone(clone)

    def test_output_boundary_clone_accepted(self):
        """Clone with non-contiguous dim_order, contiguous source, feeding only the
        output node is a valid output boundary clone."""
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        ph.meta["val"] = _NCHW.clone()
        clone = g.call_function(
            _CLONE_DIM_ORDER, args=(ph,), kwargs={"dim_order": [0, 2, 3, 1]}
        )
        clone.meta["val"] = _NHWC.clone()
        g.output((clone,))
        assert _node_is_output_boundary_clone(clone)
