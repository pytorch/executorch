# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple

import torch
import torch.nn as nn

from executorch.backends.native.serialization import (
    deserialize_graph,
    deserialize_program,
    serialize_graph,
    serialize_program,
    validate_graph,
    validate_program,
)
from executorch.backends.native.serialization.graph_serialize import (
    _named_arguments,
    _output_alias_of,
    _tensor_meta,
    _to_arg_value,
    serialize_operator,
)
from executorch.backends.native.serialization.schema import (
    GraphArg,
    InputKind,
    IntArg,
    OpKind,
    OutputKind,
    ScalarType,
    StringArg,
    SymIntListArg,
    TensorArg,
    TensorListArg,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as edge_ops


class _Add(nn.Module):
    def forward(self, x, y):
        return x + y


class _Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))

    def forward(self, x):
        self.count.add_(1)
        return x + self.count


class _KVCache(nn.Module):
    # A non-persistent (not saved in state_dict) mutable buffer, like a KV cache.
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(4), persistent=False)

    def forward(self, x):
        self.cache.add_(x)
        return self.cache + 1.0


_ADD_INPUTS = (torch.randn(2, 3), torch.randn(2, 3))

# Sentinel marking a positional slot that should be a fresh tensor placeholder.
_T = object()

_Round = namedtuple("_Round", ["edge_ep", "graph", "data", "constants"])


def _roundtrip(model, example_inputs, dynamic_shapes=None) -> _Round:
    ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    edge_ep = to_edge(ep).exported_program()
    data, constants = serialize_graph(
        edge_ep.graph_module,
        edge_ep.graph_signature,
        edge_ep.state_dict,
        edge_ep.constants,
    )
    return _Round(edge_ep, deserialize_graph(data), data, constants)


def _edge_method(model, example_inputs):
    """(graph_module, signature, state_dict, constants) tuple for serialize_program."""
    edge_ep = to_edge(torch.export.export(model, example_inputs)).exported_program()
    return (
        edge_ep.graph_module,
        edge_ep.graph_signature,
        edge_ep.state_dict,
        edge_ep.constants,
    )


def _call_targets(graph):
    return [n.target for n in graph.nodes if n.op_kind == OpKind.CALL_FUNCTION]


class SerializeRoundTripTest(unittest.TestCase):
    def test_add_target_roundtrips(self):
        graph = _roundtrip(_Add(), _ADD_INPUTS).graph
        self.assertIn("torch.ops.aten.add.Tensor", _call_targets(graph))

    def test_topological_order_preserved(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1.0

        r = _roundtrip(Model(), (torch.randn(2, 2),))
        expected = [n.name for n in r.edge_ep.graph_module.graph.nodes]
        self.assertEqual(expected, [n.name for n in r.graph.nodes])

    def test_file_identifier(self):
        self.assertEqual(_roundtrip(_Add(), _ADD_INPUTS).data[4:8], b"NPTG")

    def test_placeholder_and_output_nodes(self):
        graph = _roundtrip(_Add(), _ADD_INPUTS).graph
        kinds = {n.op_kind for n in graph.nodes}
        self.assertIn(OpKind.PLACEHOLDER, kinds)
        self.assertIn(OpKind.OUTPUT, kinds)
        self.assertIn(OpKind.CALL_FUNCTION, kinds)
        self.assertTrue(graph.inputs)
        self.assertTrue(graph.outputs)

    def test_input_tensor_metadata_recorded(self):
        graph = _roundtrip(_Add(), _ADD_INPUTS).graph
        by_name = {tv.name: tv for tv in graph.tensor_values or []}
        meta = by_name[graph.inputs[0]].meta
        self.assertEqual(meta.dtype, ScalarType.FLOAT)
        self.assertEqual([s.as_int for s in meta.sizes], [2, 3])

    def test_scalar_and_tensor_list_args(self):
        class CatModel(nn.Module):
            def forward(self, x, y):
                return torch.cat([x, y], dim=1)

        graph = _roundtrip(CatModel(), _ADD_INPUTS).graph
        arg_types = {
            type(na.arg.value) for n in graph.nodes for na in (n.inputs or [])
        }
        self.assertIn(TensorListArg, arg_types)
        self.assertIn(IntArg, arg_types)

    def test_constants_referenced_by_fqn(self):
        r = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),))
        fqns = {c.fqn for c in r.graph.constants}
        self.assertTrue(any("weight" in f for f in fqns))
        self.assertTrue(any("bias" in f for f in fqns))
        # Raw data is returned separately, keyed by the same fqns.
        self.assertEqual(set(r.constants.keys()), fqns)

    def test_tensor_args_reference_by_name(self):
        graph = _roundtrip(_Add(), _ADD_INPUTS).graph
        valid = {n.name for n in graph.nodes}
        for node in graph.nodes:
            for na in node.inputs or []:
                if isinstance(na.arg.value, TensorArg):
                    self.assertIn(na.arg.value.name, valid)


class InputClassificationTest(unittest.TestCase):
    def test_parameter_classified_and_not_mutated(self):
        graph = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),)).graph
        weight = next(c for c in graph.constants if "weight" in c.fqn)
        self.assertEqual(weight.kind, InputKind.PARAMETER)
        self.assertFalse(weight.mutated)

    def test_mutated_buffer_flagged(self):
        graph = _roundtrip(_Counter(), (torch.randn(1),)).graph
        count = next(c for c in graph.constants if "count" in c.fqn)
        self.assertEqual(count.kind, InputKind.BUFFER)
        self.assertTrue(count.mutated)


class SerializeOperatorTest(unittest.TestCase):
    def test_plain_op_overload(self):
        # A plain aten OpOverload (e.g. an aliasing op) must serialize to its real
        # name, not the bare "torch._ops.aten." from over-unwrapping its `_op`.
        self.assertEqual(
            serialize_operator(torch.ops.aten.unsqueeze.default),
            "torch.ops.aten.unsqueeze.default",
        )

    def test_sym_size_overload(self):
        self.assertEqual(
            serialize_operator(torch.ops.aten.sym_size.int),
            "torch.ops.aten.sym_size.int",
        )

    def test_edge_op_unwraps_to_aten(self):
        self.assertEqual(
            serialize_operator(edge_ops.edge.aten.mul.Tensor),
            "torch.ops.aten.mul.Tensor",
        )


class OutputSpecTest(unittest.TestCase):
    def test_user_output_classified(self):
        graph = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),)).graph
        self.assertTrue(graph.output_specs)
        self.assertTrue(
            all(s.kind == OutputKind.USER_OUTPUT for s in graph.output_specs)
        )

    def test_buffer_mutation_classified(self):
        graph = _roundtrip(_Counter(), (torch.randn(1),)).graph
        muts = [s for s in graph.output_specs if s.kind == OutputKind.BUFFER_MUTATION]
        self.assertTrue(muts, "expected a BUFFER_MUTATION output spec")
        self.assertTrue(any(s.target and "count" in s.target for s in muts))
        users = [s for s in graph.output_specs if s.kind == OutputKind.USER_OUTPUT]
        self.assertEqual(len(users), 1)


class ValidateGraphTest(unittest.TestCase):
    def test_self_contained_passes(self):
        r = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),))
        # validate_graph returns None and raises on any inconsistency; a
        # self-contained graph must validate cleanly.
        self.assertIsNone(validate_graph(r.graph, set(r.constants.keys())))

    def test_missing_constant_data_raises(self):
        graph = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),)).graph
        with self.assertRaises(ValueError):
            validate_graph(graph, available_data_keys=set())


class DynamicShapeTest(unittest.TestCase):
    def _dynamic_view(self) -> _Round:
        class M(nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1) + 1.0

        return _roundtrip(
            M(),
            (torch.randn(4, 8),),
            dynamic_shapes={"x": {0: torch.export.Dim("b")}},
        )

    def test_symbolic_size_list_roundtrips(self):
        # A dynamic view size like [s0, -1] must serialize as a SymIntListArg that
        # preserves the symbol, not collapse to an empty IntListArg.
        graph = self._dynamic_view().graph
        sym_lists = [
            na.arg.value
            for n in graph.nodes
            for na in (n.inputs or [])
            if isinstance(na.arg.value, SymIntListArg)
        ]
        self.assertTrue(sym_lists, "expected a SymIntListArg for the dynamic view size")
        self.assertTrue(
            any(s.as_symbol for sl in sym_lists for s in sl.values),
            f"expected a symbolic dim, got {sym_lists}",
        )
        for sl in sym_lists:
            self.assertTrue(sl.values)

    def test_dynamic_dim_not_frozen_in_tensor_meta(self):
        # int(sym) would specialize to the hint and freeze the dim; TensorMeta must
        # keep it symbolic.
        graph = self._dynamic_view().graph
        symbolic_dims = [
            s
            for tv in (graph.tensor_values or [])
            for s in tv.meta.sizes
            if s.as_symbol
        ]
        self.assertTrue(symbolic_dims, "dynamic dim was frozen in TensorMeta")


class ArgSerializationTest(unittest.TestCase):
    def test_unrepresentable_scalar_arg_raises(self):
        class Weird:
            pass

        with self.assertRaises(ValueError):
            _to_arg_value(Weird())

    def test_device_and_layout_serialize_as_string(self):
        self.assertIsInstance(_to_arg_value(torch.device("cpu")), StringArg)
        self.assertIsInstance(_to_arg_value(torch.strided), StringArg)
        self.assertIsInstance(_to_arg_value(torch.contiguous_format), StringArg)

    def test_default_args_are_materialized(self):
        # aten.add.Tensor has a kwarg-only `alpha=1` default that x+y never passes;
        # the serialized node must still carry it so the graph is self-describing.
        graph = _roundtrip(_Add(), (torch.randn(3), torch.randn(3))).graph
        add = next(
            n for n in graph.nodes if n.target and n.target.endswith("add.Tensor")
        )
        self.assertIn("alpha", {na.name for na in (add.inputs or [])})

    def test_noncontiguous_constant_meta_matches_shipped_data(self):
        # A non-contiguous constant is shipped contiguous, so its serialized strides
        # must describe the contiguous layout, not the original view.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.randn(8, 4).transpose(0, 1))

            def forward(self, x):
                return x + self.w

        graph = _roundtrip(M(), (torch.randn(4, 8),)).graph
        w = next(c for c in graph.constants if "w" in c.fqn)
        self.assertEqual([s.as_int for s in w.meta.strides], [8, 1])


class MutationAndAliasTest(unittest.TestCase):
    """Mutation and view aliasing are sourced from the op schema (Tensor(a!) /
    Tensor(a)), not the op-name convention. These exercise the extraction helpers
    directly on hand-built fx nodes (export/edge functionalize these ops away)."""

    @staticmethod
    def _call(target, args):
        g = torch.fx.Graph()
        real = [g.placeholder(f"in{i}") if a is _T else a for i, a in enumerate(args)]
        return g.call_function(target, tuple(real))

    def test_inplace_op_marks_only_written_arg(self):
        # add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1): only self is
        # mutated even though there are two tensor inputs.
        node = self._call(torch.ops.aten.add_.Tensor, [_T, _T])
        mutated = {na.name: na.mutated for na in _named_arguments(node)}
        self.assertTrue(mutated["self"])
        self.assertFalse(mutated["other"])

    def test_inplace_op_output_aliases_written_input(self):
        node = self._call(torch.ops.aten.add_.Tensor, [_T, _T])
        self.assertEqual(_output_alias_of(node), node.args[0].name)

    def test_view_op_output_aliases_input_without_mutation(self):
        # view(Tensor(a) self, SymInt[] size) -> Tensor(a): read-only storage share.
        node = self._call(torch.ops.aten.view.default, [_T, [6]])
        self.assertEqual(_output_alias_of(node), node.args[0].name)
        self.assertFalse(any(na.mutated for na in _named_arguments(node)))

    def test_functional_op_has_no_alias_or_mutation(self):
        node = self._call(torch.ops.aten.add.Tensor, [_T, _T])
        self.assertIsNone(_output_alias_of(node))
        self.assertFalse(any(na.mutated for na in _named_arguments(node)))

    def test_view_alias_survives_roundtrip(self):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        x.meta["val"] = torch.zeros(2, 3)
        v = g.call_function(torch.ops.aten.view.default, (x, [6]))
        v.meta["val"] = torch.zeros(6)
        g.output((v,))
        gm = torch.fx.GraphModule(nn.Module(), g)

        data, _ = serialize_graph(gm, object(), {}, None)
        graph = deserialize_graph(data)
        view = next(n for n in graph.nodes if n.target and n.target.endswith("view.default"))
        self.assertEqual(view.outputs[0].alias_of, "x")


class SubgraphHOPTest(unittest.TestCase):
    def _cond_program(self):
        class CondModel(nn.Module):
            def forward(self, pred, x):
                def true_fn(x):
                    return x + 1.0

                def false_fn(x):
                    return x - 1.0

                return torch.cond(pred, true_fn, false_fn, (x,))

        ep = torch.export.export(
            CondModel(), (torch.tensor(True), torch.randn(3))
        )
        data, _ = serialize_graph(
            ep.graph_module, ep.graph_signature, ep.state_dict, ep.constants
        )
        return deserialize_graph(data)

    def test_cond_branches_serialized_as_inlined_subgraphs(self):
        graph = self._cond_program()
        graphargs = [
            na.arg.value
            for n in graph.nodes
            for na in (n.inputs or [])
            if isinstance(na.arg.value, GraphArg)
        ]
        # torch.cond has a true and a false branch, each an inlined subgraph.
        self.assertEqual(len(graphargs), 2)
        self.assertTrue(all(ga.graph.nodes for ga in graphargs))

    def test_cond_node_present_and_get_attr_dropped(self):
        graph = self._cond_program()
        self.assertTrue(
            any(n.target and n.target.endswith("cond") for n in graph.nodes)
        )
        # The get_attr nodes that named the branch submodules are inlined away.
        # The format has no GET_ATTR op, so any leaked get_attr would have raised.
        valid_kinds = {OpKind.CALL_FUNCTION, OpKind.PLACEHOLDER, OpKind.OUTPUT}
        self.assertTrue(all(n.op_kind in valid_kinds for n in graph.nodes))

    def test_unlifted_tensor_get_attr_raises(self):
        # An unlifted module keeps params/buffers as get_attr tensor nodes. The
        # format has no GET_ATTR op, so such a graph must fail loud (no fqn/data to
        # resolve). A lifted ExportedProgram never hits this.
        class _Unlifted(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(3))

            def forward(self, x):
                return x + self.w

        gm = torch.fx.symbolic_trace(_Unlifted())
        self.assertTrue(any(n.op == "get_attr" for n in gm.graph.nodes))
        with self.assertRaisesRegex(ValueError, "not lifted"):
            serialize_graph(gm, object(), {}, None)


class DebugHandleTest(unittest.TestCase):
    def test_debug_handle_round_trips(self):
        # node.meta["debug_handle"] provenance must survive serialization.
        ep = torch.export.export(_Add(), (torch.randn(3), torch.randn(3)))
        expected = {}
        handle = 1
        for node in ep.graph_module.graph.nodes:
            if node.op == "call_function":
                node.meta["debug_handle"] = handle
                expected[node.name] = handle
                handle += 1
        self.assertTrue(expected)
        data, _ = serialize_graph(
            ep.graph_module, ep.graph_signature, ep.state_dict, ep.constants
        )
        graph = deserialize_graph(data)
        got = {n.name: n.debug_handle for n in graph.nodes if n.debug_handle}
        self.assertEqual(got, expected)

    def test_missing_debug_handle_round_trips_as_unset(self):
        # Graphs without the debug-handle pass carry no handles; the field is
        # optional and must round-trip cleanly (unset -> falsy).
        ep = torch.export.export(_Add(), (torch.randn(3), torch.randn(3)))
        data, _ = serialize_graph(
            ep.graph_module, ep.graph_signature, ep.state_dict, ep.constants
        )
        graph = deserialize_graph(data)
        self.assertTrue(all(not n.debug_handle for n in graph.nodes))


class TensorMetaTest(unittest.TestCase):
    def test_requires_grad_not_in_format(self):
        # requires_grad is a training-only concept, irrelevant at inference. It is
        # not part of the format: a requires_grad tensor (e.g. a parameter)
        # serializes fine and TensorMeta carries no requires_grad field.
        meta = _tensor_meta(torch.randn(3, requires_grad=True))
        self.assertFalse(hasattr(meta, "requires_grad"))

    def test_tensor_meta_records_dtype_and_shape(self):
        meta = _tensor_meta(torch.randn(2, 3))
        self.assertEqual(meta.dtype, ScalarType.FLOAT)
        self.assertEqual([s.as_int for s in meta.sizes], [2, 3])


class MultiMethodTest(unittest.TestCase):
    def test_program_bundles_named_methods(self):
        methods = {
            "add": _edge_method(_Add(), _ADD_INPUTS),
            "linear": _edge_method(nn.Linear(4, 4), (torch.randn(1, 4),)),
        }
        data, constants = serialize_program(methods)
        program = deserialize_program(data)
        self.assertEqual({m.name for m in program.methods}, {"add", "linear"})
        validate_program(program, set(constants.keys()))

    def test_shared_constant_fqn_deduped_across_methods(self):
        shared = nn.Linear(4, 4)
        data, constants = serialize_program(
            {
                "a": _edge_method(shared, (torch.randn(1, 4),)),
                "b": _edge_method(shared, (torch.randn(2, 4),)),
            }
        )
        program = deserialize_program(data)
        # Both methods bind the weight/bias fqns; data is merged (deduped) by fqn.
        fqns_a = {c.fqn for c in program.methods[0].graph.constants}
        fqns_b = {c.fqn for c in program.methods[1].graph.constants}
        self.assertEqual(fqns_a, fqns_b)
        self.assertEqual(set(constants.keys()), fqns_a)

    def test_serialize_graph_is_single_forward_method(self):
        data, _ = serialize_graph(*_edge_method(_Add(), _ADD_INPUTS))
        program = deserialize_program(data)
        self.assertEqual([m.name for m in program.methods], ["forward"])

    def test_conflicting_constant_fqn_across_methods_raises(self):
        # Two independent Linear(4, 4) instances share fqns ("weight"/"bias") but
        # hold different random data -- serialize_program must reject rather than
        # silently clobber one method's data.
        with self.assertRaises(ValueError):
            serialize_program(
                {
                    "a": _edge_method(nn.Linear(4, 4), (torch.randn(1, 4),)),
                    "b": _edge_method(nn.Linear(4, 4), (torch.randn(1, 4),)),
                }
            )


class MutableBufferTest(unittest.TestCase):
    def test_non_persistent_buffer_recorded_without_data(self):
        r = _roundtrip(_KVCache(), (torch.randn(4),))
        graph = r.graph
        self.assertTrue(graph.mutable_buffers, "expected a non-persistent buffer")
        mb = graph.mutable_buffers[0]
        self.assertIn("cache", mb.fqn)
        # It is NOT a data-backed constant and ships no bytes.
        self.assertNotIn("cache", {c.fqn for c in (graph.constants or [])})
        self.assertNotIn(mb.fqn, r.constants)
        # Shape/dtype are still available via the tensor_values side table.
        self.assertIn(mb.name, {tv.name for tv in (graph.tensor_values or [])})

    def test_mutable_buffer_graph_validates(self):
        r = _roundtrip(_KVCache(), (torch.randn(4),))
        # Mutable buffers are exempt from the constant-data-keys check.
        self.assertIsNone(validate_graph(r.graph, set(r.constants.keys())))
