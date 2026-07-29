# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.enforce_contiguous_dim_order import (
    EnforceContiguousDimOrder,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.node import Target


_TO_DIM_ORDER_COPY = (
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default
)  # noqa: F405
_CLONE_DIM_ORDER = exir_ops.edge.dim_order_ops._clone_dim_order.default  # noqa: F405
_EMPTY_DIM_ORDER = exir_ops.edge.dim_order_ops._empty_dim_order.default  # noqa: F405

_NCHW_INPUT = torch.randn(1, 4, 8, 8)
_NHWC_INPUT = torch.randn(1, 4, 8, 8).to(memory_format=torch.channels_last)
_CIFARNET_INPUT_NHWC = torch.randn(1, 3, 32, 32).to(memory_format=torch.channels_last)
_CIFARNET_INPUT_NCHW = torch.randn(1, 3, 32, 32)


# ── helpers ────────────────────────────────────────────────────────────────────


def _export_to_edge(module: torch.nn.Module, inputs: tuple) -> ExportedProgram:
    ep = torch.export.export(module.eval(), inputs)
    return to_edge(ep).exported_program()


def _run_pass(ep: ExportedProgram) -> tuple[GraphModule, bool]:
    result = EnforceContiguousDimOrder()(ep.graph_module)
    return result.graph_module, result.modified


def _find_nodes(gm: GraphModule, target: Target) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _count(gm: GraphModule, target: Target) -> int:
    return len(_find_nodes(gm, target))


def _placeholder_nodes(gm: GraphModule) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _non_contiguous_placeholders(gm: GraphModule) -> list[torch.fx.Node]:
    return [
        n
        for n in _placeholder_nodes(gm)
        if isinstance(n.meta.get("val"), torch.Tensor)
        and not n.meta["val"].is_contiguous()
    ]


# ── Type A modules ─────────────────────────────────────────────────────────────
# Exported with a channels last input and the model calls .contiguous() internally. Primary purpose: verify that the
#  pass inserts a boundary clone after the non-contiguous placeholder and removes the internal _clone_dim_order ops.


class TypeAContiguousCallModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        return x.contiguous()


class TypeAContiguousBeforeAddReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        x = x.contiguous()
        return torch.relu(x)


class TypeAConv2dModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.contiguous()


# ── Type B modules ─────────────────────────────────────────────────────────────
# Exported with a contiguous input and the model applies a channels-last clone internally. Primary purpose: verify how
#  to_edge() lowers memory-format ops and that the pass removes internal _clone_dim_order nodes.


class TypeBSingleCloneModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone(memory_format=torch.channels_last)


class TypeBCloneBeforeReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone(memory_format=torch.channels_last)
        return torch.relu(x)


class TypeBMultipleIndependentClonesModule(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clone(memory_format=torch.channels_last)
        y = y.clone(memory_format=torch.channels_last)
        return x + y


class TypeBChainedClonesModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone(memory_format=torch.channels_last)
        return x.clone(memory_format=torch.contiguous_format)


class TypeBConv2dModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.clone(memory_format=torch.channels_last)


# ── Other modules ───────────────────────────────────────────────────


class EmptyDimOrderModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, memory_format=torch.channels_last) + x


class CifarNet(torch.nn.Module):
    """Lightweight CNN matching the CifarNet topology used for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ── tests ──────────────────────────────────────────────────────────────────────


class TestEnforceContiguousDimOrderPass:

    # ── preconditions: how to_edge() lowers memory-format ops ─────────────────
    # If these fail the lowering behavior has changed and the dependent tests below are invalid.

    def test_to_edge_lowers_clone_channels_last_to_clone_dim_order(self):
        """.clone(channels_last) must produce _clone_dim_order in edge dialect."""
        ep = _export_to_edge(TypeBSingleCloneModule(), (_NCHW_INPUT,))
        assert (
            _count(ep.graph_module, _CLONE_DIM_ORDER) == 1
        ), "Expected _clone_dim_order in the edge graph. Lowering may have changed."

    def test_to_edge_lowers_contiguous_call_to_clone_dim_order(self):
        """.contiguous() on a channels-last input must produce _clone_dim_order."""
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))
        assert (
            _count(ep.graph_module, _CLONE_DIM_ORDER) == 1
        ), "Expected _clone_dim_order from .contiguous() in the edge graph."

    def test_to_edge_produces_multiple_clone_dim_order_nodes(self):
        ep = _export_to_edge(
            TypeBMultipleIndependentClonesModule(), (_NCHW_INPUT, _NCHW_INPUT)
        )
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 2

    def test_to_edge_produces_chained_clone_dim_order_nodes(self):
        ep = _export_to_edge(TypeBChainedClonesModule(), (_NCHW_INPUT,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 2

    def test_empty_dim_order_precondition(self):
        """to_edge() must lower torch.empty_like(..., channels_last) to _empty_dim_order with a non-contiguous
        dim_order kwarg.
        """
        ep = _export_to_edge(EmptyDimOrderModule(), (_NCHW_INPUT,))
        empty_nodes = _find_nodes(ep.graph_module, _EMPTY_DIM_ORDER)
        assert len(empty_nodes) == 1, (
            "Expected at least one _empty_dim_order node. "
            "The lowering pass may have changed."
        )
        for node in empty_nodes:
            dim_order = node.kwargs.get("dim_order")
            assert dim_order is not None, f"Node {node.name!r} has no dim_order kwarg."
            assert not list(dim_order) == list(range(len(dim_order))), (
                f"Expected a non-contiguous dim_order before the pass, "
                f"got {dim_order}."
            )

    # ── Type A: boundary clone insertion (channels-last input) ─────────────────

    def test_type_a_boundary_clone_inserted_after_non_contiguous_placeholder(self):
        """After the pass, exactly one _clone_dim_order must remain, and it must have the non-contiguous placeholder as
        its direct input.
        """
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))
        gm, modified = _run_pass(ep)

        assert modified
        clone_nodes = _find_nodes(gm, _CLONE_DIM_ORDER)
        assert (
            len(clone_nodes) == 1
        ), f"Expected exactly one boundary clone, got {len(clone_nodes)}"
        nhwc_placeholders = _non_contiguous_placeholders(gm)
        assert len(nhwc_placeholders) == 1
        assert (
            clone_nodes[0].args[0] is nhwc_placeholders[0]
        ), "Boundary clone must consume the non-contiguous placeholder directly."

    def test_type_a_internal_clone_removed_boundary_clone_inserted(self):
        """The internal _clone_dim_order (from .contiguous()) must be removed and replaced by exactly one boundary
        clone.
        """
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))
        clone_nodes = _find_nodes(ep.graph_module, _CLONE_DIM_ORDER)
        assert len(clone_nodes) == 1
        assert clone_nodes[0].args[0].op == "call_function"
        assert (
            clone_nodes[0].args[0].target == exir_ops.edge.aten.add.Tensor
        )  # internal clone exists

        gm, modified = _run_pass(ep)

        assert modified
        clone_nodes = _find_nodes(gm, _CLONE_DIM_ORDER)
        assert len(clone_nodes) == 1
        assert (
            clone_nodes[0].args[0].op == "placeholder"
        )  # only the boundary clone remains

    def test_type_a_placeholder_meta_val_remains_non_contiguous(self):
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))
        gm, _ = _run_pass(ep)

        nhwc_placeholders = _non_contiguous_placeholders(gm)
        assert (
            len(nhwc_placeholders) == 1
        ), "The channels-last placeholder meta['val'] must remain non-contiguous."

    def test_type_a_downstream_ops_consume_boundary_clone_not_placeholder(self):
        """After the pass, relu must be connected to the add, and the add must be connected to the boundary clone, not
        directly to the non-contiguous placeholder.
        """
        ep = _export_to_edge(TypeAContiguousBeforeAddReluModule(), (_NHWC_INPUT,))
        gm, modified = _run_pass(ep)

        assert modified
        relu_nodes = _find_nodes(gm, exir_ops.edge.aten.relu.default)
        assert len(relu_nodes) == 1
        relu_input = relu_nodes[0].args[0]
        assert relu_input.target == exir_ops.edge.aten.add.Tensor, (
            f"relu must consume the add operator, "
            f"got target={getattr(relu_input, 'target', None)!r}"
        )

        add_nodes = _find_nodes(gm, exir_ops.edge.aten.add.Tensor)
        assert len(add_nodes) == 1
        add_input = add_nodes[0].args[0]
        assert add_input.target == _CLONE_DIM_ORDER, (
            f"add must consume the boundary _clone_dim_order, "
            f"got target={getattr(relu_input, 'target', None)!r}"
        )

    def test_type_a_boundary_clone_meta_val_is_contiguous(self):
        """The boundary clone's meta['val'] must be contiguous so that
        downstream metadata propagation produces correct results."""
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))
        gm, _ = _run_pass(ep)

        clone_nodes = _find_nodes(gm, _CLONE_DIM_ORDER)
        assert len(clone_nodes) == 1
        assert clone_nodes[0].args[0].op == "placeholder"  # Boundary clone.
        val = clone_nodes[0].meta.get("val")
        assert isinstance(val, torch.Tensor)
        assert val.is_contiguous(), "Boundary clone meta['val'] must be contiguous."

    def test_type_a_graph_structurally_valid(self):
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))
        gm, _ = _run_pass(ep)
        gm.graph.lint()

    def test_type_a_graph_structurally_valid_conv2d(self):
        ep = _export_to_edge(TypeAConv2dModule(), (_NHWC_INPUT,))
        gm, _ = _run_pass(ep)
        gm.graph.lint()

    def test_type_a_all_non_placeholder_meta_val_contiguous(self):
        """Every non-placeholder node must have contiguous meta['val'] after the
        pass, including format-preserving ops such as relu."""
        ep = _export_to_edge(TypeAContiguousBeforeAddReluModule(), (_NHWC_INPUT,))
        gm, _ = _run_pass(ep)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                assert (
                    val.is_contiguous()
                ), f"Node {node.name!r} has non-contiguous meta['val'] after pass."

    def test_type_a_pass_is_idempotent(self):
        """After inserting the boundary clone (Type A), a second run must
        preserve it and report modified=False."""
        ep = _export_to_edge(TypeAContiguousCallModule(), (_NHWC_INPUT,))

        gm, modified_first = _run_pass(ep)
        assert modified_first
        assert _count(gm, _CLONE_DIM_ORDER) == 1  # boundary clone present

        result_second = EnforceContiguousDimOrder()(gm)
        assert not result_second.modified
        # Boundary clone must be preserved, not re-removed.
        assert _count(result_second.graph_module, _CLONE_DIM_ORDER) == 1

    # ── Type B: internal clone removal (contiguous input) ─────────────────────

    def test_type_b_remove_single_internal_clone(self):
        ep = _export_to_edge(TypeBSingleCloneModule(), (_NCHW_INPUT,))
        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, _CLONE_DIM_ORDER) == 0  # The clone was removed.

    def test_type_b_remove_multiple_independent_internal_clones(self):
        ep = _export_to_edge(
            TypeBMultipleIndependentClonesModule(), (_NCHW_INPUT, _NCHW_INPUT)
        )
        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, _CLONE_DIM_ORDER) == 0  # Both clones were removed.

    def test_type_b_remove_chained_internal_clones(self):
        ep = _export_to_edge(TypeBChainedClonesModule(), (_NCHW_INPUT,))
        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, _CLONE_DIM_ORDER) == 0  # Both clones were removed.

    def test_type_b_downstream_op_reconnected_to_placeholder_after_removal(self):
        """After the internal clone is removed, relu must consume the
        placeholder directly (no intermediate node)."""
        ep = _export_to_edge(TypeBCloneBeforeReluModule(), (_NCHW_INPUT,))
        assert _count(ep.graph_module, _CLONE_DIM_ORDER) == 1

        gm, modified = _run_pass(ep)

        assert modified
        relu_nodes = _find_nodes(gm, exir_ops.edge.aten.relu.default)
        assert len(relu_nodes) == 1
        relu_input = relu_nodes[0].args[0]
        assert relu_input.op == "placeholder", (
            f"relu input must be the placeholder after clone removal, "
            f"got op={relu_input.op!r}"
        )

    def test_type_b_graph_structurally_valid(self):
        ep = _export_to_edge(TypeBSingleCloneModule(), (_NCHW_INPUT,))
        gm, _ = _run_pass(ep)
        gm.graph.lint()

    def test_type_b_graph_structurally_valid_after_chained_removal(self):
        ep = _export_to_edge(TypeBChainedClonesModule(), (_NCHW_INPUT,))
        gm, _ = _run_pass(ep)
        gm.graph.lint()

    def test_type_b_all_non_placeholder_meta_val_contiguous(self):
        ep = _export_to_edge(TypeBCloneBeforeReluModule(), (_NCHW_INPUT,))
        gm, _ = _run_pass(ep)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                assert (
                    val.is_contiguous()
                ), f"Node {node.name!r} has non-contiguous meta['val'] after pass."

    def test_type_b_pass_is_idempotent(self):
        """After removing internal clones (Type B), a second run must report modified=False and leave the graph
        unchanged.
        """
        ep = _export_to_edge(TypeBSingleCloneModule(), (_NCHW_INPUT,))

        gm, modified_first = _run_pass(ep)
        assert modified_first
        assert _count(gm, _CLONE_DIM_ORDER) == 0

        result_second = EnforceContiguousDimOrder()(gm)
        assert not result_second.modified
        assert _count(result_second.graph_module, _CLONE_DIM_ORDER) == 0

    def test_empty_dim_order_kwarg_rewritten_to_contiguous(self):
        """After the pass, every _empty_dim_order node must have a contiguous dim_order kwarg, modified must be True,
        and graph.lint() must pass.
        """
        ep = _export_to_edge(EmptyDimOrderModule(), (_NCHW_INPUT,))
        assert _count(ep.graph_module, _EMPTY_DIM_ORDER) == 1

        gm, modified = _run_pass(ep)

        assert modified
        empty_nodes = _find_nodes(gm, _EMPTY_DIM_ORDER)
        assert (
            len(empty_nodes) == 1
        ), "_empty_dim_order nodes must survive the pass (only their kwarg is rewritten)."
        for node in empty_nodes:
            dim_order = node.kwargs.get("dim_order")
            ndim = len(dim_order)
            assert list(dim_order) == list(range(ndim)), (
                f"Node {node.name!r} still has non-contiguous dim_order "
                f"{dim_order} after the pass."
            )

        gm.graph.lint()

    # ── numerical correctness ─────────────────────────────────────────────────

    def test_type_a_numerical_correctness_contiguous_call(self):
        """.contiguous() is the identity function — the pass must not alter values."""
        torch.manual_seed(4)
        x = _NHWC_INPUT.clone()
        model = TypeAContiguousCallModule().eval()
        reference = model(x)

        ep = _export_to_edge(model, (x,))
        _run_pass(ep)

        assert torch.allclose(ep.module()(x)[0], reference)

    def test_type_a_numerical_correctness_conv2d(self):
        torch.manual_seed(5)
        x = _NHWC_INPUT.clone()
        model = TypeAConv2dModule().eval()
        reference = model(x)

        ep = _export_to_edge(model, (x,))
        _run_pass(ep)

        assert torch.allclose(ep.module()(x)[0], reference, atol=1e-5)

    def test_type_b_numerical_correctness_single_clone(self):
        torch.manual_seed(0)
        x = _NCHW_INPUT.clone()
        model = TypeBSingleCloneModule().eval()
        reference = model(x)

        ep = _export_to_edge(model, (x,))
        _run_pass(ep)

        assert torch.allclose(ep.module()(x)[0], reference)

    def test_type_b_numerical_correctness_chained_clones(self):
        torch.manual_seed(1)
        x = _NCHW_INPUT.clone()
        model = TypeBChainedClonesModule().eval()
        reference = model(x)

        ep = _export_to_edge(model, (x,))
        _run_pass(ep)

        assert torch.allclose(ep.module()(x)[0], reference)

    def test_type_b_numerical_correctness_multiple_clones(self):
        torch.manual_seed(2)
        x = _NCHW_INPUT.clone()
        y = torch.randn(1, 4, 8, 8)
        model = TypeBMultipleIndependentClonesModule().eval()
        reference = model(x, y)

        ep = _export_to_edge(model, (x, y))
        _run_pass(ep)

        assert torch.allclose(ep.module()(x, y)[0], reference)

    def test_type_b_numerical_correctness_conv2d(self):
        torch.manual_seed(3)
        x = _NCHW_INPUT.clone()
        model = TypeBConv2dModule().eval()
        reference = model(x)

        ep = _export_to_edge(model, (x,))
        _run_pass(ep)

        assert torch.allclose(ep.module()(x)[0], reference, atol=1e-5)

    def test_empty_dim_order_numerical_correctness(self):
        """Rewriting the dim_order kwarg must not alter the output values.
        The addition result depends only on x, so the outputs must match
        regardless of the allocation layout."""
        torch.manual_seed(0)
        x = _NCHW_INPUT.clone()
        # empty_like produces uninitialised memory; fill it with zeros so the
        # reference and exported outputs are deterministically comparable.
        model = EmptyDimOrderModule().eval()

        ep = _export_to_edge(model, (x,))
        _run_pass(ep)

        out = ep.module()(x)[0].reshape([1, 4, 8, 8])
        assert out.shape == x.shape
        assert out.is_contiguous(), "Output tensor must be contiguous after the pass."


class TestEnforceContiguousDimOrderPassCifarNet:

    def test_cifarnet_precondition_channels_last_input_has_non_contiguous_placeholder(
        self,
    ):
        """Exporting with NHWC input must produce exactly one non-contiguous
        placeholder (the model input).  Lifted parameters are always contiguous
        and must not appear here."""
        model = CifarNet().eval()
        ep = _export_to_edge(model, (_CIFARNET_INPUT_NHWC,))
        assert len(_non_contiguous_placeholders(ep.graph_module)) == 1

    def test_cifarnet_graph_structure_after_pass(self):
        """After the pass:
        - Exactly one boundary _clone_dim_order sits directly after the NHWC
          placeholder with a contiguous dim_order kwarg.
        - No other _clone_dim_order or _to_dim_order_copy nodes remain.
        - graph.lint() passes.
        """
        model = CifarNet().eval()
        ep = _export_to_edge(model, (_CIFARNET_INPUT_NHWC,))

        gm, modified = _run_pass(ep)

        assert modified

        nhwc_placeholders = _non_contiguous_placeholders(gm)
        assert len(nhwc_placeholders) == 1
        nhwc_placeholder = nhwc_placeholders[0]

        # Every surviving _clone_dim_order must be the boundary clone.
        clone_nodes = _find_nodes(gm, _CLONE_DIM_ORDER)
        assert len(clone_nodes) == 1, (
            f"Expected exactly one _clone_dim_order (the boundary clone), "
            f"got {len(clone_nodes)}."
        )
        boundary_clone = clone_nodes[0]
        assert (
            boundary_clone.args[0] is nhwc_placeholder
        ), "Boundary clone must consume the non-contiguous placeholder directly."
        assert list(boundary_clone.kwargs["dim_order"]) == [0, 1, 2, 3], (
            f"Boundary clone dim_order must be [0,1,2,3], "
            f"got {boundary_clone.kwargs['dim_order']}."
        )
        assert _count(gm, _TO_DIM_ORDER_COPY) == 0

        gm.graph.lint()

    def test_cifarnet_meta_val_dim_order_after_pass(self):
        """After the pass:
        - The NHWC input placeholder's meta['val'] must remain non-contiguous
          (it reflects the actual runtime input format).
        - Every other node must have contiguous meta['val'], including conv
          outputs that carried channels-last format before the pass.
        """
        model = CifarNet().eval()
        ep = _export_to_edge(model, (_CIFARNET_INPUT_NHWC,))

        gm, _ = _run_pass(ep)

        assert (
            len(_non_contiguous_placeholders(gm)) == 1
        ), "The channels-last input placeholder must remain non-contiguous."

        non_contiguous_internal = [
            node
            for node in gm.graph.nodes
            if node.op != "placeholder"
            and isinstance(node.meta.get("val"), torch.Tensor)
            and not node.meta["val"].is_contiguous()
        ]
        assert non_contiguous_internal == [], (
            "The following non-placeholder nodes still have non-contiguous "
            "meta['val'] after the pass:\n"
            + "\n".join(
                f"  {n.name!r} ({n.target}): dim_order={n.meta['val'].dim_order()}"
                for n in non_contiguous_internal
            )
        )

    def test_cifarnet_pass_is_idempotent(self):
        """A second run must report modified=False and leave the boundary
        clone count unchanged."""
        model = CifarNet().eval()
        ep = _export_to_edge(model, (_CIFARNET_INPUT_NHWC,))

        gm, modified_first = _run_pass(ep)
        assert modified_first

        clone_count = _count(gm, _CLONE_DIM_ORDER)
        result_second = EnforceContiguousDimOrder()(gm)

        assert not result_second.modified
        assert _count(result_second.graph_module, _CLONE_DIM_ORDER) == clone_count

    def test_cifarnet_numerical_correctness(self):
        """The pass must not alter computed output values, and a model exported
        with NHWC input (pass applied) must agree with the same model exported
        with NCHW input for identical data."""
        torch.manual_seed(42)
        model = CifarNet().eval()

        x_nchw = _CIFARNET_INPUT_NCHW.clone()
        x_nhwc = x_nchw.to(memory_format=torch.channels_last)

        reference = model(x_nchw).flatten()

        # NHWC export with pass: must match the reference.
        ep_nhwc = _export_to_edge(model, (x_nhwc,))
        _run_pass(ep_nhwc)
        out_nhwc = ep_nhwc.module()(x_nhwc)[0].flatten()

        assert out_nhwc.shape == reference.shape
        assert torch.allclose(out_nhwc, reference, atol=1e-5), (
            f"NHWC export (pass applied) vs reference — "
            f"max absolute error: {(out_nhwc - reference).abs().max().item():.2e}"
        )

        # NCHW export (no pass needed): must also match — confirms that the
        # NHWC export is not just self-consistent but genuinely equivalent.
        ep_nchw = _export_to_edge(model, (x_nchw,))
        out_nchw = ep_nchw.module()(x_nchw)[0]

        assert torch.allclose(out_nchw, out_nhwc, atol=1e-5), (
            f"NCHW export vs NHWC export (pass applied) — "
            f"max absolute error: {(out_nchw - out_nhwc).abs().max().item():.2e}"
        )
