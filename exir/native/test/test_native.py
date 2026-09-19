# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import operator
import os
import tempfile
import unittest
import zipfile
from collections import Counter
from dataclasses import asdict

import torch
import torch.nn as nn

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.serialization.graph_serialize import deserialize_program
from executorch.backends.native.serialization.schema import (
    Graph,
    Method,
    MutableBufferSpec,
    Program,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize._ptn import ALIASES_ENTRY, PTG_ENTRY, SAFETENSORS_ENTRY
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_submodules,
)
from executorch.exir.native import to_native
from executorch.exir.native.native import (
    _check_delegate_contract,
    _validate_merged_program,
)


def _canonical_graph(graph) -> str:
    # Structural, order-stable form of a schema Graph for multiset comparison.
    return json.dumps(asdict(graph), sort_keys=True, default=str)


def _safetensors_keys(blob: bytes) -> set[str]:
    header_len = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8 : 8 + header_len])
    return {k for k in header if k != "__metadata__"}


class ToNativeEquivalenceTest(unittest.TestCase):
    def _test_equivalence(self, programs):
        """Assert to_native's .ptn matches the ET partition/lower flow.

        The per-method native graph (ptg) and the constant key set must agree:
        to_native merges the per-method graphs into one multi-method Program
        and ships constants in a deduped safetensors + alias map, while the ET
        flow keeps one single-method delegate blob per method and would ship the
        same constants in a .ptd. The graphs are compared modulo that wrapping,
        and the full key sets (safetensors owners + aliases) must match the
        delegates' named-data keys.
        """
        method_programs = (
            programs if isinstance(programs, dict) else {"forward": programs}
        )

        # to_native path: .ptn -> ptg + safetensors + alias map.
        native = to_native(programs)
        self.assertEqual(native.methods, set(method_programs))
        with tempfile.TemporaryDirectory() as td:
            ptn_path = os.path.join(td, "m.ptn")
            native.save(ptn_path)
            with zipfile.ZipFile(ptn_path) as pkg:
                entries = set(pkg.namelist())
                ptg_blob = pkg.read(PTG_ENTRY)
                st_keys = (
                    _safetensors_keys(pkg.read(SAFETENSORS_ENTRY))
                    if SAFETENSORS_ENTRY in entries
                    else set()
                )
                aliases = (
                    json.loads(pkg.read(ALIASES_ENTRY))
                    if ALIASES_ENTRY in entries
                    else {}
                )

        native_program = deserialize_program(ptg_blob)
        native_graphs = {m.name: m.graph for m in native_program.methods}
        native_keys = st_keys | set(aliases.keys())

        # ET path: partition and lower, then read each delegate directly -- absent
        # the PTN compile spec, processed_bytes is the native graph flatbuffer --
        # plus the constant keys from its named data store (a .ptd's key set).
        edge = to_edge_transform_and_lower(
            method_programs,
            transform_passes=get_default_passes(),
            partitioner=[NativePartitioner()],
            compile_config=get_default_compile_config(),
        )
        et_keys: set[str] = set()
        et_graphs = []
        for name in method_programs:
            lowered = get_lowered_submodules(edge.exported_program(name).graph_module)
            self.assertEqual(len(lowered), 1)
            module = lowered[0][1]
            et_graphs.append(
                deserialize_program(module.processed_bytes).methods[0].graph
            )
            nds = module.named_data_store_output
            if nds is not None:
                # Constants may be embedded in the .pte (pte_data) or shipped to a
                # .ptd (external_data) depending on external_constants_tag; collect
                # both so the key set matches to_native's .ptn/.safetensors.
                et_keys |= set(nds.pte_data.keys())
                for key_to_entry in nds.external_data.values():
                    et_keys |= set(key_to_entry.keys())

        # Emission is outside the comparison, but keep proving the ExecuTorch path
        # still completes end to end through the unchanged NamedDataStore branch.
        self.assertTrue(edge.to_executorch().buffer)

        # 1. Same method set, and every ptg graph matches modulo multi-method
        #    wrapping (compared as a multiset, ignoring method names).
        self.assertEqual(set(native_graphs), set(method_programs))
        self.assertEqual(len(et_graphs), len(method_programs))
        self.assertEqual(
            Counter(_canonical_graph(g) for g in native_graphs.values()),
            Counter(_canonical_graph(g) for g in et_graphs),
        )

        # 2. Constant keys match; safetensors holds owners, the alias map redirects
        #    duplicates to a real owner key, and owners are never aliased.
        self.assertEqual(native_keys, et_keys)
        self.assertTrue(set(aliases.values()) <= st_keys)
        self.assertFalse(st_keys & set(aliases.keys()))
        return native_keys, st_keys, aliases

    def test_linear(self):
        ep = torch.export.export(nn.Linear(8, 4), (torch.randn(2, 8),))
        _, st_keys, aliases = self._test_equivalence(ep)
        self.assertTrue(st_keys)  # weight + bias shipped
        self.assertFalse(aliases)  # distinct constants, nothing to dedup

    def test_mlp(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        ep = torch.export.export(model, (torch.randn(2, 8),))
        _, st_keys, _ = self._test_equivalence(ep)
        self.assertEqual(len(st_keys), 4)  # two linears, weight + bias each

    def test_duplicate_constants_dedup(self):
        class DupConst(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(4))
                self.register_buffer("b", torch.ones(4))

            def forward(self, x):
                return x + self.a + self.b

        ep = torch.export.export(DupConst(), (torch.randn(4),))
        native_keys, st_keys, aliases = self._test_equivalence(ep)
        # a and b are byte-identical: one owner entry, one alias to it.
        self.assertEqual(len(native_keys), 2)
        self.assertEqual(len(st_keys), 1)
        self.assertEqual(len(aliases), 1)

    def test_multi_method(self):
        model = nn.Linear(8, 4)
        programs = {
            "prefill": torch.export.export(model, (torch.randn(2, 8),)),
            "decode": torch.export.export(model, (torch.randn(1, 8),)),
        }
        # Shared weights resolve to the same fqns across methods, so the key sets
        # still line up between the two paths.
        _, st_keys, _ = self._test_equivalence(programs)
        self.assertEqual(len(st_keys), 2)  # one weight + bias, shared by fqn

    def test_conv_relu(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        ep = torch.export.export(model, (torch.randn(1, 3, 16, 16),))
        _, st_keys, _ = self._test_equivalence(ep)
        self.assertEqual(len(st_keys), 2)  # conv weight + bias

    def test_tied_weights(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, 8, bias=False)
                self.decoder = nn.Linear(8, 8, bias=False)
                self.decoder.weight = self.encoder.weight

            def forward(self, x):
                return self.decoder(self.encoder(x))

        ep = torch.export.export(Tied(), (torch.randn(2, 8),))
        native_keys, st_keys, aliases = self._test_equivalence(ep)
        # The tied weight is byte-identical wherever it is referenced, so it owns
        # a single safetensors entry and any extra fqn aliases to it.
        self.assertTrue(native_keys)
        self.assertEqual(len(st_keys), len(native_keys) - len(aliases))

    def test_matmul_with_parameter(self):
        class MatmulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8, 4))

            def forward(self, x):
                # matmul is a non-core op the partitioner opts into and preserves.
                return torch.matmul(x, self.w)

        ep = torch.export.export(MatmulModel(), (torch.randn(2, 8),))
        _, st_keys, _ = self._test_equivalence(ep)
        self.assertTrue(st_keys)

    def test_view_ops(self):
        class ViewModel(nn.Module):
            def forward(self, x, y):
                # transpose + reshape lower to view_copy ops that
                # ReplaceCopyWithAliasPass rewrites to aliasing views in-delegate.
                return x.transpose(1, 2).reshape(x.shape[0], -1) + y

        ep = torch.export.export(
            ViewModel(), (torch.randn(2, 3, 4), torch.randn(2, 12))
        )
        self._test_equivalence(ep)

    def test_lifted_constant(self):
        class ConstAdd(nn.Module):
            def forward(self, x):
                # The scalar 1.0 is materialized as a lifted tensor constant
                # during to_edge; it must still be sourced and shipped.
                return x + 1.0

        ep = torch.export.export(ConstAdd(), (torch.randn(2, 4),))
        _, st_keys, _ = self._test_equivalence(ep)
        self.assertTrue(any("lifted" in k for k in st_keys))


class ToNativeInputValidationTest(unittest.TestCase):
    def test_empty_method_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one method"):
            to_native({})

    def test_wrong_top_level_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be an ExportedProgram"):
            to_native(object())

    def test_non_string_method_name_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "method name must be str"):
            to_native({1: object()})

    def test_empty_method_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            to_native({"": object()})

    def test_wrong_method_value_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must map to an ExportedProgram"):
            to_native({"forward": object()})


class MergedProgramValidationTest(unittest.TestCase):
    def test_missing_mutable_buffer_metadata_is_rejected(self):
        program = Program(
            methods=[
                Method(
                    name="forward",
                    graph=Graph(nodes=[]),
                    mutable_buffers=[MutableBufferSpec(name="state", fqn="state")],
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing tensor metadata"):
            _validate_merged_program(program, {})

    def test_unresolved_graph_reference_is_rejected(self):
        program = Program(
            methods=[
                Method(
                    name="forward",
                    graph=Graph(nodes=[], outputs=["undefined"]),
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "unresolved value reference"):
            _validate_merged_program(program, {})


class DelegateContractTest(unittest.TestCase):
    """One delegate is not the same as an identity wrapper around one delegate.

    Real partial delegations are awkward to construct -- they need an op the
    partitioner rejects sitting beside one it claims -- and the outer-graph shapes
    that lose semantics need no ops at all, so drive the check directly.
    """

    def _graph_module(
        self,
        outputs: int = 1,
        residual: bool = False,
        passthrough: bool = False,
        order: list[int] | None = None,
        drop_input: bool = False,
        stray_getitem: bool = False,
        reuse_output: bool = False,
        literal_output: bool = False,
    ) -> torch.fx.GraphModule:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        lowered = graph.get_attr("lowered_module_0")
        args = (lowered, x) if drop_input else (lowered, x, y)
        call = graph.call_function(executorch_call_delegate, args)

        indices = order if order is not None else list(range(outputs))
        leaves = [graph.call_function(operator.getitem, (call, i)) for i in indices]
        if residual:
            leaves[0] = graph.call_function(torch.ops.aten.relu.default, (leaves[0],))
        if stray_getitem:
            # Rooted in another node rather than the delegate call.
            leaves[0] = graph.call_function(operator.getitem, (leaves[0], 0))
        if reuse_output:
            # Reusing one Node is distinct from creating two getitem Nodes with
            # the same index: Node.all_input_nodes collapses this duplicate.
            leaves.append(leaves[0])
        if literal_output:
            # Literals do not appear in Node.all_input_nodes at all.
            leaves.append(7)
        if passthrough:
            leaves.insert(0, x)

        graph.output(tuple(leaves))
        # GraphModule copies attrs its get_attr nodes name out of the root and
        # rejects ones the root lacks, so supply it via a dict root.
        return torch.fx.GraphModule({"lowered_module_0": nn.Module()}, graph)

    def _check(self, graph_module):
        _check_delegate_contract(graph_module, ["x", "y"], "forward")

    def test_identity_wrapper_is_accepted(self):
        self._check(self._graph_module())

    def test_multi_output_identity_projection_is_accepted(self):
        self._check(self._graph_module(outputs=3))

    def test_residual_op_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not fully delegate") as ctx:
            self._check(self._graph_module(residual=True))
        # The message has to name the op, or the failure is unactionable.
        self.assertIn("relu", str(ctx.exception))

    def test_passthrough_output_is_rejected(self):
        # Needs no op, so a residual-op scan alone would let it through and the
        # packaged program would quietly lose that output.
        with self.assertRaisesRegex(
            ValueError, "does not come from the native delegate"
        ):
            self._check(self._graph_module(passthrough=True))

    def test_swapped_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "in order"):
            self._check(self._graph_module(order=[1, 0]))

    def test_duplicated_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "in order"):
            self._check(self._graph_module(order=[0, 0]))

    def test_reused_output_node_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "in order"):
            self._check(self._graph_module(reuse_output=True))

    def test_literal_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "returns literal 7"):
            self._check(self._graph_module(literal_output=True))

    def test_output_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "in order"):
            self._check(self._graph_module(order=[0, 2]))

    def test_getitem_not_rooted_in_delegate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "indexes something other than"):
            self._check(self._graph_module(stray_getitem=True))

    def test_dropped_public_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not pass its inputs"):
            self._check(self._graph_module(drop_input=True))


class MutableBufferDedupTest(unittest.TestCase):
    def test_identical_mutable_buffers_stay_separate(self):
        """Equal bytes must not make two independently mutable buffers one.

        Both start as zeros, so content dedup would alias them and a write to one
        would show up in the other.
        """

        class TwoCounters(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(4))
                self.register_buffer("b", torch.zeros(4))

            def forward(self, x):
                self.a.add_(x)
                self.b.add_(x * 2)
                return self.a + self.b

        ep = torch.export.export(TwoCounters(), (torch.ones(4),))
        native = to_native(ep)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.ptn")
            native.save(path)
            with zipfile.ZipFile(path) as pkg:
                entries = set(pkg.namelist())
                owners = _safetensors_keys(pkg.read(SAFETENSORS_ENTRY))
                aliases = (
                    json.loads(pkg.read(ALIASES_ENTRY))
                    if ALIASES_ENTRY in entries
                    else {}
                )

        mutable = {k for k in owners | set(aliases) if k.endswith(("a", "b"))}
        self.assertEqual(len(mutable), 2, f"expected both buffers, got {mutable}")
        self.assertFalse(
            set(aliases) & mutable, f"a mutable buffer was aliased: {aliases}"
        )


if __name__ == "__main__":
    unittest.main()
