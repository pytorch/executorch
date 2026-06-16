# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Optional

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as edge_ops
from executorch.exir.passes.reinplace import DEFAULT_INPLACEABLE_OPS, reinplace_pass
from executorch.extension.pybindings.portable_lib import (  # @manual=//executorch/extension/pybindings:portable_lib
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.export.exported_program import ExportedProgram


def _find_nodes(
    ep: ExportedProgram,
    contains: str,
    excludes: Optional[str] = None,
) -> List[torch.fx.Node]:
    """Return all ``call_function`` nodes whose target string contains
    ``contains``. If ``excludes`` is given, also drop any node whose
    target string contains that substring (used to distinguish the
    functional form ``index_put`` from the in-place ``index_put_``).
    """
    return [
        n
        for n in ep.graph.nodes
        if n.op == "call_function"
        and contains in str(n.target)
        and (excludes is None or excludes not in str(n.target))
    ]


class TestReinplacePass(unittest.TestCase):
    def test_index_put_reinplace(self) -> None:
        """Test that index_put on a mutable buffer can be reinplaced."""

        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                # index_put on buffer (non-user input) should be safe
                self.state.index_put_((indices,), values)
                return self.state

        model = IndexPutModel()
        indices = torch.tensor([0])
        values = torch.tensor([1.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge._edge_programs["forward"]

        self.assertEqual(
            len(_find_nodes(edge_program, "index_put")),
            1,
            "Should find an index_put node",
        )

        et = edge.to_executorch(ExecutorchBackendConfig(run_reinplace_pass=True))
        et_program = et.exported_program()
        self.assertEqual(
            len(_find_nodes(et_program, "index_put_")),
            1,
            "Should find an index_put_ node",
        )
        self.assertEqual(
            len(_find_nodes(et_program, "copy_")),
            0,
            "Shouldn't find a copy_ node",
        )

        e = _load_for_executorch_from_buffer(et.buffer)
        self.assertTrue(
            torch.allclose(
                e.forward((indices, values))[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
            )
        )

    def test_cant_reinplace(self) -> None:
        """Test that index_put on a mutable buffer that is viewed later is not safe."""

        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                # index_put on buffer (non-user input) should be safe
                x = self.state.index_put((indices,), values)
                self.state.add_(1)
                return x

        model = IndexPutModel()
        indices = torch.tensor([0])
        values = torch.tensor([1.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        self.assertEqual(
            len(_find_nodes(edge_program, "index_put")),
            1,
            "Should find an index_put node",
        )

        ep = reinplace_pass(edge_program)
        self.assertEqual(
            len(_find_nodes(ep, "index_put", excludes="index_put_")),
            1,
            "Should still find a functional index_put node",
        )

        # Lower to ExecuTorch and verify runtime correctness against eager.
        et = edge.to_executorch()
        loaded = _load_for_executorch_from_buffer(et.buffer)
        et_out = loaded.forward((indices, values))
        eager_out = IndexPutModel()(indices, values)
        self.assertTrue(torch.allclose(et_out[0], eager_out))

    def test_unsafe_downstream_blocks_upstream_reinplace(self) -> None:
        """When an upstream index_put's mutated arg is also an input to an
        unsafe downstream index_put, the upstream one must not be reinplaced
        either. Otherwise the in-place mutation by the upstream op would
        change the value the (still-functional) downstream op reads.
        """

        class TwoIndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # Intermediate tensor consumed by both index_puts.
                t = values + 1.0
                # Upstream index_put — mutates t (an intermediate, so
                # by itself it would look safe to reinplace).
                a = t.index_put((indices,), torch.tensor([5.0]))
                # Downstream index_put — reads t. This one is itself
                # unsafe because state is read again by the add_ below.
                b = self.state.index_put((indices,), t)
                self.state.add_(1)
                return a, b

        model = TwoIndexPutModel()
        # `indices` and `values` must have matching shape so the
        # downstream `state.index_put((indices,), t)` is well-formed
        # (state[indices[i]] = t[i]). The original test used values
        # shape [5] which is fine for graph-shape checks but not
        # runnable in ET.
        indices = torch.tensor([0])
        values = torch.tensor([2.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        # Sanity check: both functional index_puts present pre-pass.
        self.assertEqual(
            len(_find_nodes(edge_program, "index_put", excludes="index_put_")),
            2,
            "Should have two functional index_put nodes",
        )

        ep = reinplace_pass(edge_program)

        # Neither index_put should be reinplaced. The downstream one is
        # unsafe directly (state used later); the upstream one is unsafe
        # because its mutated arg `t` is read by the unsafe downstream op.
        self.assertEqual(
            len(_find_nodes(ep, "index_put", excludes="index_put_")),
            2,
            "Neither index_put should be reinplaced",
        )
        self.assertEqual(
            len(_find_nodes(ep, "index_put_")),
            0,
            "No index_put_ should have been introduced",
        )

        # Lower to ExecuTorch and verify runtime correctness against eager.
        et = edge.to_executorch()
        loaded = _load_for_executorch_from_buffer(et.buffer)
        et_out = loaded.forward((indices, values))
        eager_a, eager_b = TwoIndexPutModel()(indices, values)
        self.assertTrue(torch.allclose(et_out[0], eager_a))
        self.assertTrue(torch.allclose(et_out[1], eager_b))

    def test_kwargs_are_forwarded(self) -> None:
        """When the matched node carries a value in ``node.kwargs`` (e.g.
        ``accumulate=True`` for ``index_put``), the rewrite must forward
        those kwargs to the in-place form. Otherwise the in-place op
        falls back to the schema default and silently changes semantics.

        ``export`` normalizes most arguments into positional form, so we
        explicitly move ``accumulate`` into ``node.kwargs`` after export
        to exercise the kwarg-forwarding path.
        """

        class IndexPutAccumulateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                self.state.index_put_((indices,), values, accumulate=True)
                return self.state

        model = IndexPutAccumulateModel()
        indices = torch.tensor([0, 0])
        values = torch.tensor([1.0, 2.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        # Find the functional index_put node and force `accumulate` onto
        # its kwargs (export normalized it to positional). This is what
        # exercises the kwarg-forwarding path in reinplace_pass.
        functionals = _find_nodes(edge_program, "index_put", excludes="index_put_")
        self.assertEqual(len(functionals), 1, "Should find a functional index_put")
        functional = functionals[0]

        # index_put schema: (self, indices, values, accumulate=False).
        # Move arg[3] -> kwargs["accumulate"] if present.
        if len(functional.args) >= 4:
            new_args = functional.args[:3]
            new_kwargs = dict(functional.kwargs)
            new_kwargs["accumulate"] = functional.args[3]
            functional.args = new_args
            functional.kwargs = new_kwargs
        self.assertEqual(
            functional.kwargs.get("accumulate"),
            True,
            "Test setup: accumulate should now be a kwarg",
        )

        ep = reinplace_pass(edge_program)

        # Find the rewritten in-place index_put_ node.
        inplace_nodes = _find_nodes(ep, "index_put_")
        self.assertEqual(len(inplace_nodes), 1, "Should find an index_put_ node")
        index_put_inplace = inplace_nodes[0]

        # `accumulate` must survive the rewrite, in either args or kwargs.
        accumulate = index_put_inplace.kwargs.get("accumulate")
        if accumulate is None and len(index_put_inplace.args) >= 4:
            accumulate = index_put_inplace.args[3]
        self.assertEqual(
            accumulate,
            True,
            "accumulate=True must be preserved through the rewrite",
        )

    def test_ops_to_inplace_extends_with_add(self) -> None:
        """A custom ``ops_to_inplace`` set can extend the pass to ops
        outside the default set. Add the edge-dialect ``add.Tensor`` to
        the set; the in-place form (``add_.Tensor``) is auto-derived by
        name + schema match. Verify a safe-to-reinplace add gets
        rewritten while an unsafe one (mutating an immutable input)
        does not.
        """

        class TwoAddModel(torch.nn.Module):
            def forward(
                self,
                x: torch.Tensor,
                y: torch.Tensor,
                z: torch.Tensor,
            ) -> torch.Tensor:
                # First add: mutated arg `x` is an immutable user input
                # → not safe to reinplace.
                t = x + y
                # Second add: mutated arg `t` is an intermediate with no
                # later use → safe to reinplace.
                return t + z

        model = TwoAddModel()
        args = (torch.zeros(5), torch.ones(5), torch.full((5,), 2.0))
        exported_program = export(model, args, strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        # Sanity: no in-place adds before the pass.
        self.assertEqual(len(_find_nodes(edge_program, "add_")), 0)

        custom_set = {edge_ops.edge.aten.add.Tensor}
        ep = reinplace_pass(edge_program, ops_to_inplace=custom_set)

        # Exactly one of the two adds should be reinplaced.
        self.assertEqual(
            len(_find_nodes(ep, "add_")),
            1,
            "Exactly one (intermediate-mutating) add should be reinplaced",
        )
        self.assertEqual(
            len(_find_nodes(ep, "aten.add.Tensor")),
            1,
            "The add mutating an immutable input must remain functional",
        )

        # Lower to ExecuTorch. The portable runtime does not register a
        # kernel for `add_.Tensor`, so we only verify lowering succeeds
        # (the in-place rewrite must serialize cleanly into the ET
        # program). Runtime execution is covered by ops with portable
        # kernels in the other tests in this file.
        edge.to_executorch()

    def test_ops_to_inplace_empty_disables_all_rewrites(self) -> None:
        """Passing an empty ``ops_to_inplace`` set should disable every
        rewrite, even ops that are in ``DEFAULT_INPLACEABLE_OPS``.
        """

        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                self.state.index_put_((indices,), values)
                return self.state

        model = IndexPutModel()
        indices = torch.tensor([0])
        values = torch.tensor([1.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        # Sanity: index_put is in the default set, so without our
        # explicit override it would be reinplaced.
        self.assertIn(
            edge_ops.edge.aten.index_put.default,
            DEFAULT_INPLACEABLE_OPS,
            "Sanity check: edge index_put should be in the default set",
        )

        ep = reinplace_pass(edge_program, ops_to_inplace={})

        self.assertEqual(
            len(_find_nodes(ep, "index_put_")),
            0,
            "Empty set must disable all rewrites, including the default ones",
        )
        self.assertEqual(
            len(_find_nodes(ep, "index_put", excludes="index_put_")),
            1,
            "The functional index_put must remain",
        )

        # Lower to ExecuTorch and verify runtime correctness against eager.
        et = edge.to_executorch()
        loaded = _load_for_executorch_from_buffer(et.buffer)
        et_out = loaded.forward((indices, values))
        eager_out = IndexPutModel()(indices, values)
        self.assertTrue(torch.allclose(et_out[0], eager_out))

    def test_ops_to_inplace_custom_does_not_inherit_default(self) -> None:
        """A custom ``ops_to_inplace`` set replaces — not augments —
        ``DEFAULT_INPLACEABLE_OPS``. Passing a set that doesn't include
        ``index_put`` leaves it functional, even though it would be
        reinplaced under the default. Callers who want to extend rather
        than replace should union with ``DEFAULT_INPLACEABLE_OPS``
        explicitly (per the docstring guidance on ``reinplace_pass``).
        """

        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                self.state.index_put_((indices,), values)
                return self.state

        model = IndexPutModel()
        indices = torch.tensor([0])
        values = torch.tensor([1.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        # Custom set containing only `add` — no `index_put`.
        custom_set = {edge_ops.edge.aten.add.Tensor}
        ep = reinplace_pass(edge_program, ops_to_inplace=custom_set)

        self.assertEqual(
            len(_find_nodes(ep, "index_put_")),
            0,
            "index_put must remain functional when not in the custom set",
        )

        # Lower to ExecuTorch and verify runtime correctness against eager.
        et = edge.to_executorch()
        loaded = _load_for_executorch_from_buffer(et.buffer)
        et_out = loaded.forward((indices, values))
        eager_out = IndexPutModel()(indices, values)
        self.assertTrue(torch.allclose(et_out[0], eager_out))

    def test_kv_cache_style_reinplace(self) -> None:
        """HF-style static KV cache update via ``index_copy_`` along the
        ``cache_len`` dim. Mirrors transformers' ``StaticCache.update``::

            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)

        Cache shape mirrors HF::
            (max_batch_size, num_heads, max_cache_len, head_dim)

        ``index_copy`` is **not** in ``DEFAULT_INPLACEABLE_OPS``, so this
        also exercises the new ``ops_to_inplace`` extension API. We use
        ``EdgeCompileConfig(preserve_ops=[index_copy])`` to keep the op
        from being decomposed by ``to_edge``, then add ``index_copy`` to
        a custom set (the in-place form is auto-derived by name + schema
        match) and verify both buffer updates are rewritten in-place.
        """

        max_batch_size, num_heads, max_cache_len, head_dim = 1, 2, 4, 8

        class HFStyleStaticCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "keys",
                    torch.zeros((max_batch_size, num_heads, max_cache_len, head_dim)),
                )
                self.register_buffer(
                    "values",
                    torch.zeros((max_batch_size, num_heads, max_cache_len, head_dim)),
                )

            def forward(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                cache_position: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # HF static-cache style: in-place index_copy along dim=2.
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
                return self.keys, self.values

        model = HFStyleStaticCache()
        key_states = torch.full((max_batch_size, num_heads, 1, head_dim), 1.0)
        value_states = torch.full((max_batch_size, num_heads, 1, head_dim), 2.0)
        cache_position = torch.tensor([1])

        exported_program = export(
            model, (key_states, value_states, cache_position), strict=True
        )

        # Preserve `index_copy` through the edge lowering so the pass
        # actually sees it (default to_edge decomposes it into index_put).
        edge = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                preserve_ops=[torch.ops.aten.index_copy.default],
            ),
        )
        edge_program = edge.exported_program()

        # Sanity: pre-pass, both updates are functional `index_copy`.
        self.assertEqual(
            len(_find_nodes(edge_program, "index_copy", excludes="index_copy_")),
            2,
            "Pre-pass: should have two functional index_copy nodes (keys, values)",
        )

        # Add `index_copy` to the set; the in-place form is
        # auto-derived by name + schema match
        # (ops.edge.aten.index_copy_.default).
        custom_set = DEFAULT_INPLACEABLE_OPS | {
            edge_ops.edge.aten.index_copy.default,
        }
        ep = reinplace_pass(edge_program, ops_to_inplace=custom_set)

        # Both updates should now be in-place.
        self.assertEqual(
            len(_find_nodes(ep, "index_copy_")),
            2,
            "Both keys and values index_copy ops should be reinplaced",
        )
        self.assertEqual(
            len(_find_nodes(ep, "index_copy", excludes="index_copy_")),
            0,
            "No functional index_copy nodes should remain",
        )

        # Lower to ExecuTorch. The portable runtime does not register a
        # kernel for `index_copy_`, so we only verify lowering succeeds
        # (the in-place rewrite must serialize cleanly into the ET
        # program).
        edge.to_executorch()

    def test_chain_of_inplaceable_ops(self) -> None:
        """A chain of safe-to-reinplace ops gets fully rewritten in
        topological order. Exercises:
          * Multiple distinct ops (`add` and `relu`) registered together
            via a single set, with in-place forms auto-derived.
          * Reverse-walk safety propagation: each intermediate is
            consumed exactly once by the next op, so every step except
            the first sees its mutated arg as not-yet-used.
          * The first ``add`` mutates an immutable user input
            (``x``) and must remain functional.
        """

        class ChainModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                t = x + y  # add #1: mutates `x` (immutable input) -> unsafe.
                t = torch.relu(t)  # relu #1: mutates intermediate -> safe.
                t = t + y  # add #2: mutates intermediate -> safe.
                t = torch.relu(t)  # relu #2: mutates intermediate -> safe.
                return t

        model = ChainModel()
        x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
        y = torch.tensor([1.0, 1.0, 1.0, 1.0])

        exported_program = export(model, (x, y), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge.exported_program()

        custom_set = {
            edge_ops.edge.aten.add.Tensor,
            edge_ops.edge.aten.relu.default,
        }
        ep = reinplace_pass(edge_program, ops_to_inplace=custom_set)

        self.assertEqual(
            len(_find_nodes(ep, "aten.add.Tensor")),
            1,
            "First add mutates an immutable input; must stay functional",
        )
        self.assertEqual(
            len(_find_nodes(ep, "add_")),
            1,
            "Second add mutates an intermediate; should be reinplaced",
        )
        self.assertEqual(
            len(_find_nodes(ep, "aten.relu.default")),
            0,
            "Both relus mutate intermediates; neither should remain functional",
        )
        self.assertEqual(
            len(_find_nodes(ep, "relu_")),
            2,
            "Both relus should be reinplaced",
        )

        # Lower to ExecuTorch. The portable runtime does not register
        # kernels for `add_.Tensor` or `relu_.default`, so we only
        # verify lowering succeeds (the in-place rewrites must
        # serialize cleanly into the ET program).
        edge.to_executorch()
