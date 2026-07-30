# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple
from types import SimpleNamespace

import torch
import torch.nn as nn

from executorch.backends.native.serialization import (
    deserialize_graph,
    deserialize_program,
    serialize_graph,
    serialize_program,
    validate_graph,
    validate_method,
    validate_program,
)
from executorch.backends.native.serialization.graph_serialize import (
    _build_output_specs,
    _compile_to_bytes,
    _dim_order,
    _named_arguments,
    _node_outputs,
    _output_alias_of,
    _to_arg_value,
    serialize_operator,
)
from executorch.backends.native.serialization.schema import (
    AffineGroup,
    Argument,
    BoolArg,
    BoolListArg,
    Dim,
    FloatArg,
    FloatListArg,
    Graph,
    GraphArg,
    InputKind,
    IntArg,
    IntListArg,
    Method,
    MutableBufferSpec,
    NamedArgument,
    Node,
    NoneArg,
    OpKind,
    OptionalTensorListArg,
    Output,
    OutputKind,
    OutputValueKind,
    PackedQuant,
    Program,
    QuantSpec,
    ScalarType,
    ScalarTypeArg,
    StringArg,
    TensorArg,
    TensorListArg,
    TensorMeta,
    TensorValue,
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

_Round = namedtuple("_Round", ["edge_ep", "method", "graph", "data", "constants"])


def _roundtrip(model, example_inputs, dynamic_shapes=None) -> _Round:
    ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    edge_ep = to_edge(ep).exported_program()
    data, constants = serialize_graph(
        edge_ep.graph_module,
        edge_ep.graph_signature,
        edge_ep.state_dict,
        edge_ep.constants,
    )
    method = deserialize_program(data).methods[0]
    # Baseline invariants asserted for every roundtrip so each test starts from a
    # validated method rather than re-checking these individually.
    assert data[4:8] == b"NPTG", f"unexpected file identifier {data[4:8]!r}"
    assert method.graph.nodes, "deserialized graph has no nodes"
    validate_method(method, set(constants.keys()))
    return _Round(edge_ep, method, method.graph, data, constants)


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
        self.assertEqual([d.max for d in meta.sizes], [2, 3])

    def test_intermediate_metadata_emitted(self):
        # Every tensor-valued node (not just graph I/O) records shape + dtype, so a
        # consumer never re-infers intermediate metadata.
        class Model(nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1.0

        graph = _roundtrip(Model(), (torch.randn(2, 2),)).graph
        meta_names = {tv.name for tv in (graph.tensor_values or [])}
        call_outputs = {
            n.name for n in graph.nodes if n.op_kind == OpKind.CALL_FUNCTION
        }
        self.assertTrue(call_outputs)
        # includes the intermediate relu output, not only the final add output
        self.assertTrue(call_outputs <= meta_names)

    def test_scalar_and_tensor_list_args(self):
        class CatModel(nn.Module):
            def forward(self, x, y):
                return torch.cat([x, y], dim=1)

        graph = _roundtrip(CatModel(), _ADD_INPUTS).graph
        arg_types = {type(na.arg.value) for n in graph.nodes for na in (n.inputs or [])}
        self.assertIn(TensorListArg, arg_types)
        self.assertIn(IntArg, arg_types)

    def test_constants_referenced_by_fqn(self):
        r = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),))
        fqns = {c.data_key for c in r.method.constants}
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
        method = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),)).method
        weight = next(c for c in method.constants if "weight" in c.data_key)
        self.assertEqual(weight.kind, InputKind.PARAMETER)
        self.assertFalse(weight.mutated)

    def test_mutated_buffer_flagged(self):
        method = _roundtrip(_Counter(), (torch.randn(1),)).method
        count = next(c for c in method.constants if "count" in c.data_key)
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
        method = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),)).method
        self.assertTrue(method.output_specs)
        self.assertTrue(
            all(s.kind == OutputKind.USER_OUTPUT for s in method.output_specs)
        )

    def test_buffer_mutation_classified(self):
        method = _roundtrip(_Counter(), (torch.randn(1),)).method
        muts = [s for s in method.output_specs if s.kind == OutputKind.BUFFER_MUTATION]
        self.assertTrue(muts, "expected a BUFFER_MUTATION output spec")
        self.assertTrue(any(s.target and "count" in s.target for s in muts))
        users = [s for s in method.output_specs if s.kind == OutputKind.USER_OUTPUT]
        self.assertEqual(len(users), 1)

    def test_parameter_mutation_rejected(self):
        sig = SimpleNamespace(
            buffers_to_mutate={},
            user_inputs_to_mutate={},
            parameters_to_mutate={"w": "w"},
        )
        with self.assertRaisesRegex(ValueError, "parameter mutation is not supported"):
            _build_output_specs([], sig)


class ValidateGraphTest(unittest.TestCase):
    def test_self_contained_passes(self):
        r = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),))
        # validate_method returns None and raises on any inconsistency; a
        # self-contained method must validate cleanly.
        self.assertIsNone(validate_method(r.method, set(r.constants.keys())))

    def test_missing_constant_data_raises(self):
        r = _roundtrip(nn.Linear(4, 4), (torch.randn(1, 4),))
        with self.assertRaises(ValueError):
            validate_method(r.method, available_data_keys=set())

    @staticmethod
    def _meta() -> TensorMeta:
        return TensorMeta(
            dtype=ScalarType.FLOAT, sizes=[Dim(min=1, max=1)], dim_order=[0]
        )

    @staticmethod
    def _placeholder(name: str) -> Node:
        return Node(name=name, op_kind=OpKind.PLACEHOLDER, outputs=[Output(name=name)])

    @classmethod
    def _call(cls, name: str, inputs=None) -> Node:
        return Node(
            name=name,
            op_kind=OpKind.CALL_FUNCTION,
            target="aten::relu",
            inputs=inputs,
            outputs=[Output(name=name)],
        )

    def test_placeholder_restating_input_passes(self):
        g = Graph(
            nodes=[
                self._placeholder("x"),
                self._call(
                    "y",
                    [NamedArgument(name="self", arg=Argument(value=TensorArg("x")))],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            tensor_values=[
                TensorValue(name="x", meta=self._meta()),
                TensorValue(name="y", meta=self._meta()),
            ],
        )
        self.assertIsNone(validate_graph(g))

    def test_duplicate_input_raises(self):
        g = Graph(nodes=[], inputs=["x", "x"])
        with self.assertRaisesRegex(ValueError, "duplicate value definition 'x'"):
            validate_graph(g)

    def test_node_redefining_input_raises(self):
        g = Graph(
            nodes=[self._call("x")],
            inputs=["x"],
            tensor_values=[TensorValue(name="x", meta=self._meta())],
        )
        with self.assertRaisesRegex(ValueError, "duplicate value definition 'x'"):
            validate_graph(g)

    def test_duplicate_node_raises(self):
        g = Graph(nodes=[self._call("y"), self._call("y")])
        with self.assertRaisesRegex(ValueError, "duplicate node name"):
            validate_graph(g)

    def test_undeclared_placeholder_raises(self):
        g = Graph(nodes=[self._placeholder("z")])
        with self.assertRaisesRegex(
            ValueError, "not a declared input or external binding"
        ):
            validate_graph(g)

    def test_duplicate_placeholder_raises(self):
        g = Graph(
            nodes=[self._placeholder("a"), self._placeholder("a")],
            inputs=["a"],
            tensor_values=[TensorValue(name="a", meta=self._meta())],
        )
        with self.assertRaisesRegex(ValueError, "duplicate node name"):
            validate_graph(g)

    def test_optional_tensor_list_length_mismatch_raises(self):
        g = Graph(
            nodes=[
                self._placeholder("a"),
                self._call(
                    "y",
                    [
                        NamedArgument(
                            name="indices",
                            arg=Argument(
                                value=OptionalTensorListArg(
                                    names=["a", "b"], has_value=[True]
                                )
                            ),
                        )
                    ],
                ),
            ],
            inputs=["a"],
            tensor_values=[TensorValue(name="a", meta=self._meta())],
        )
        with self.assertRaisesRegex(ValueError, "OptionalTensorListArg names length"):
            validate_graph(g)

    def test_unresolved_reference_raises(self):
        g = Graph(
            nodes=[
                self._call(
                    "y",
                    [NamedArgument(arg=Argument(value=TensorArg("missing")))],
                )
            ],
            tensor_values=[TensorValue(name="y", meta=self._meta())],
        )
        with self.assertRaisesRegex(ValueError, "unresolved value reference 'missing'"):
            validate_graph(g)

    def test_int_list_refs_length_mismatch_raises(self):
        g = Graph(
            nodes=[
                self._placeholder("a"),
                self._call(
                    "y",
                    [
                        NamedArgument(
                            arg=Argument(value=IntListArg(values=[1, 2], refs=["a"]))
                        )
                    ],
                ),
            ],
            inputs=["a"],
            tensor_values=[TensorValue(name="a", meta=self._meta())],
        )
        with self.assertRaisesRegex(ValueError, "IntListArg refs length"):
            validate_graph(g)

    def test_input_missing_metadata_raises(self):
        g = Graph(nodes=[self._placeholder("x")], inputs=["x"])
        with self.assertRaisesRegex(ValueError, "input 'x' missing tensor metadata"):
            validate_graph(g)

    def test_output_missing_metadata_raises(self):
        g = Graph(
            nodes=[
                self._placeholder("x"),
                self._call(
                    "y",
                    [NamedArgument(arg=Argument(value=TensorArg("x")))],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            tensor_values=[TensorValue(name="x", meta=self._meta())],
        )
        with self.assertRaisesRegex(ValueError, "output 'y' missing tensor metadata"):
            validate_graph(g)

    def test_mutable_buffer_missing_metadata_raises(self):
        method = Method(
            name="forward",
            graph=Graph(
                nodes=[self._placeholder("x")],
                inputs=["x"],
                tensor_values=[TensorValue(name="x", meta=self._meta())],
            ),
            mutable_buffers=[MutableBufferSpec(name="cache", fqn="m.cache")],
        )
        with self.assertRaisesRegex(
            ValueError, "mutable buffer 'cache' .* missing tensor metadata"
        ):
            validate_method(method)


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
        # A dynamic view size like [s0, -1] serializes as an IntListArg whose `refs`
        # marks the symbolic element as a reference to an in-graph int value (the
        # sym_size node), keeping the -1 as a literal, not collapsed or dropped.
        graph = self._dynamic_view().graph
        int_lists = [
            na.arg.value
            for n in graph.nodes
            for na in (n.inputs or [])
            if isinstance(na.arg.value, IntListArg)
        ]
        ref_lists = [il for il in int_lists if il.refs and any(il.refs)]
        self.assertTrue(
            ref_lists, "expected an IntListArg with a ref for the dynamic view size"
        )
        il = ref_lists[0]
        # refs is parallel to values; the dynamic element has a non-empty ref, the
        # -1 literal has an empty ref.
        self.assertEqual(len(il.refs), len(il.values))
        self.assertTrue(any(r for r in il.refs))
        self.assertTrue(any(not r for r in il.refs))

    def test_dynamic_dim_not_frozen_in_tensor_meta(self):
        # int(sym) would specialize to the hint and freeze the dim; TensorMeta must
        # keep it as a range (min != max), not a single concrete value.
        graph = self._dynamic_view().graph
        dynamic_dims = [
            d
            for tv in (graph.tensor_values or [])
            for d in tv.meta.sizes
            if d.max != d.min
        ]
        self.assertTrue(dynamic_dims, "dynamic dim was frozen (min==max) in TensorMeta")


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
        # A non-contiguous constant is shipped contiguous, so its serialized layout
        # (dim_order) must describe the contiguous layout, not the original view.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.randn(8, 4).transpose(0, 1))

            def forward(self, x):
                return x + self.w

        method = _roundtrip(M(), (torch.randn(4, 8),)).method
        w = next(c for c in method.constants if "w" in c.data_key)
        self.assertEqual(w.meta.dim_order, [0, 1])


class DimOrderTest(unittest.TestCase):
    def test_contiguous(self):
        self.assertEqual(_dim_order(torch.randn(2, 3, 4)), [0, 1, 2])

    def test_permuted_contiguous(self):
        t = torch.randn(2, 3, 4).permute(0, 2, 1)
        self.assertEqual(_dim_order(t), [0, 2, 1])

    def test_channels_last_with_size_one_channel(self):
        # channels-last leaves the size-1 channel with an arbitrary stride, which
        # must not be treated as a non-expressible layout.
        t = torch.randn(2, 1, 3, 4).to(memory_format=torch.channels_last)
        self.assertEqual(_dim_order(t), [0, 2, 1, 3])

    def test_sliced_layout_raises(self):
        t = torch.randn(4, 8)[:, :4]
        with self.assertRaisesRegex(ValueError, "not dim-order expressible"):
            _dim_order(t)


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
        view = next(
            n for n in graph.nodes if n.target and n.target.endswith("view.default")
        )
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

        ep = torch.export.export(CondModel(), (torch.tensor(True), torch.randn(3)))
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
        # The get_attr nodes that named the branch submodules are inlined away: their
        # names appear as GraphArg labels but never as nodes.
        node_names = {n.name for n in graph.nodes}
        ga_names = {
            na.arg.value.name
            for n in graph.nodes
            for na in (n.inputs or [])
            if isinstance(na.arg.value, GraphArg)
        }
        self.assertTrue(ga_names)
        self.assertTrue(ga_names.isdisjoint(node_names))

    def test_cond_graph_validates(self):
        # Inlined subgraphs must themselves be self-contained (references resolve,
        # I/O has tensor metadata); validate recurses into every GraphArg.
        self.assertIsNone(validate_graph(self._cond_program()))

    def test_cond_symbool_predicate_is_bool_ref(self):
        class M(nn.Module):
            def forward(self, x):
                def tf(t):
                    return t + 1

                def ff(t):
                    return t - 1

                return torch.cond(x.shape[0] > 4, tf, ff, (x,))

        ep = torch.export.export(
            M(),
            (torch.randn(6),),
            dynamic_shapes={"x": {0: torch.export.Dim("b", min=2, max=16)}},
        )
        data, _ = serialize_graph(
            ep.graph_module, ep.graph_signature, ep.state_dict, ep.constants
        )
        graph = deserialize_graph(data)
        cond_node = next(
            n for n in graph.nodes if n.target and n.target.endswith("cond")
        )
        pred = cond_node.inputs[0].arg.value
        self.assertIsInstance(pred, BoolArg)
        self.assertTrue(pred.ref)
        bool_out_names = {
            o.name
            for n in graph.nodes
            for o in (n.outputs or [])
            if o.kind == OutputValueKind.BOOL
        }
        self.assertIn(pred.ref, bool_out_names)
        validate_graph(graph)

    def test_subgraph_in_list_arg_fails_loud(self):
        # A HOP subgraph can only be inlined as a direct GraphArg. If a subgraph
        # get_attr node shows up inside a list arg it must raise, not emit a dangling
        # TensorListArg reference (the get_attr node is dropped from the graph).
        g = torch.fx.Graph()
        attr = g.get_attr("sub")
        sg = torch.fx.Graph()
        p = sg.placeholder("x")
        sg.output((p,))
        sub_gm = torch.fx.GraphModule(nn.Module(), sg)
        with self.assertRaises(ValueError):
            _to_arg_value([attr], subgraph_map={attr.name: sub_gm})


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
        fqns_a = {c.data_key for c in program.methods[0].constants}
        fqns_b = {c.data_key for c in program.methods[1].constants}
        self.assertEqual(fqns_a, fqns_b)
        self.assertEqual(set(constants.keys()), fqns_a)

    def test_serialize_graph_is_single_forward_method(self):
        data, _ = serialize_graph(*_edge_method(_Add(), _ADD_INPUTS))
        program = deserialize_program(data)
        self.assertEqual([m.name for m in program.methods], ["forward"])

    def test_conflicting_constant_fqn_across_methods_raises(self):
        # Two independent Linear(4, 4) instances share fqns ("weight"/"bias") but
        # hold different random data, so serialize_program must reject rather than
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
        method = r.method
        self.assertTrue(method.mutable_buffers, "expected a non-persistent buffer")
        mb = method.mutable_buffers[0]
        self.assertIn("cache", mb.fqn)
        # It is NOT a data-backed constant and ships no bytes.
        self.assertNotIn("cache", {c.data_key for c in (method.constants or [])})
        self.assertNotIn(mb.fqn, r.constants)
        # Shape/dtype are still available via the tensor_values side table.
        self.assertIn(mb.name, {tv.name for tv in (method.graph.tensor_values or [])})

    def test_mutable_buffer_graph_validates(self):
        r = _roundtrip(_KVCache(), (torch.randn(4),))
        # Mutable buffers are exempt from the constant-data-keys check.
        self.assertIsNone(validate_method(r.method, set(r.constants.keys())))


class QuantSpecRoundTripTest(unittest.TestCase):
    """The quant spec rides on TensorMeta. No producer sets it yet (the QDQ-fold
    pass is future work), so these build the dataclasses directly and round-trip
    them through flatc to lock the schema + optional-union wrapper."""

    def _roundtrip_meta(self, quant) -> TensorMeta:
        meta = TensorMeta(
            dtype=ScalarType.CHAR,
            sizes=[Dim(min=4, max=4)],
            dim_order=[0],
            quant=quant,
        )
        graph = Graph(
            nodes=[Node(name="w", op_kind=OpKind.PLACEHOLDER)],
            tensor_values=[TensorValue(name="w", meta=meta)],
        )
        program = Program(version="1", methods=[Method(name="forward", graph=graph)])
        out = deserialize_program(_compile_to_bytes(program))
        return out.methods[0].graph.tensor_values[0].meta

    def test_absent_quant_is_none(self):
        self.assertIsNone(self._roundtrip_meta(None).quant)

    def test_affine_group_quant_roundtrips(self):
        expected = AffineGroup(
            scale_data_key="w.scale",
            scale_dtype=ScalarType.HALF,
            quant_min=-8,
            quant_max=7,
            group_size=32,
            zero_point_data_key="w.zp",
            zero_point_dtype=ScalarType.INT,
        )
        scheme = self._roundtrip_meta(QuantSpec(scheme=expected)).quant.scheme
        self.assertEqual(scheme, expected)

    def test_packed_quant_roundtrips(self):
        # Weight-only formats (GGUF, MXFP4, NVFP4) carry only an opaque codec name.
        scheme = self._roundtrip_meta(
            QuantSpec(scheme=PackedQuant(codec="gguf:q4k"))
        ).quant.scheme
        self.assertIsInstance(scheme, PackedQuant)
        self.assertEqual(scheme.codec, "gguf:q4k")


class NonTensorInputTest(unittest.TestCase):
    def test_non_tensor_user_input_fails_loud(self):
        # A scalar (non-tensor) user input cannot be represented, so serialization
        # must raise rather than emit a graph with an untyped/valueless input.
        class M(nn.Module):
            def forward(self, x, n):
                return x + n

        edge = to_edge(torch.export.export(M(), (torch.randn(3), 2))).exported_program()
        with self.assertRaises(ValueError):
            serialize_graph(
                edge.graph_module,
                edge.graph_signature,
                edge.state_dict,
                edge.constants,
            )


class MultiOutputTest(unittest.TestCase):
    def test_topk_outputs_and_metadata(self):
        class M(nn.Module):
            def forward(self, x):
                vals, idx = torch.topk(x, 2)
                return vals + 1, idx

        r = _roundtrip(M(), (torch.randn(4),))
        graph = r.graph
        producer = next(
            n for n in graph.nodes if n.target and n.target.endswith("topk.default")
        )
        # topk returns (values, indices): two distinctly named outputs.
        self.assertEqual(len(producer.outputs), 2)
        self.assertEqual(len({o.name for o in producer.outputs}), 2)
        # Each result carries its own tensor metadata.
        meta_names = {tv.name for tv in (graph.tensor_values or [])}
        for out in producer.outputs:
            self.assertIn(out.name, meta_names)
        # getitem users are folded into the producer's outputs, not emitted as nodes.
        self.assertFalse(
            any(n.target and n.target.endswith("getitem") for n in graph.nodes)
        )
        validate_method(r.method, set(r.constants.keys()))

    def test_split_tensor_list_output(self):
        class M(nn.Module):
            def forward(self, x):
                a, b = torch.split(x, 2)
                return a + 1, b + 1

        r = _roundtrip(M(), (torch.randn(4),))
        graph = r.graph
        producer = next(n for n in graph.nodes if n.target and "split" in n.target)
        # A Tensor[] return is one TENSOR_LIST output holding the element names.
        self.assertEqual(len(producer.outputs), 1)
        out = producer.outputs[0]
        self.assertEqual(out.kind, OutputValueKind.TENSOR_LIST)
        self.assertEqual(len(out.names or []), 2)
        meta_names = {tv.name for tv in (graph.tensor_values or [])}
        for nm in out.names:
            self.assertIn(nm, meta_names)
        self.assertFalse(
            any(n.target and n.target.endswith("getitem") for n in graph.nodes)
        )
        validate_method(r.method, set(r.constants.keys()))

    def test_scalar_node_output_kind(self):
        class M(nn.Module):
            def forward(self, x):
                n = x.shape[0]
                return x + n

        r = _roundtrip(
            M(),
            (torch.randn(4, 3),),
            dynamic_shapes={"x": {0: torch.export.Dim("b")}},
        )
        scalar_outs = [
            o
            for n in r.graph.nodes
            for o in (n.outputs or [])
            if o.kind == OutputValueKind.INT
        ]
        self.assertGreaterEqual(
            len(scalar_outs), 1, "expected an int-kind scalar output (sym_size)"
        )

    def test_zero_return_op_has_no_outputs(self):
        g = torch.fx.Graph()
        n = g.call_function(torch.ops.aten._assert_scalar.default, (True, "msg"))
        n.meta["val"] = None
        self.assertEqual(_node_outputs(n, {}, {}), [])


class DynamicScalarNodeArgTest(unittest.TestCase):
    # A node producing a dynamic bool/float (e.g. a `x.shape[0] > 4` cond predicate)
    # is referenced by SSA name, like a dynamic int, not serialized as a tensor.
    def test_float_node_arg_uses_ref(self):
        g = torch.fx.Graph()
        n = g.placeholder("s")
        n.meta["val"] = 1.5
        arg = _to_arg_value(n)
        self.assertIsInstance(arg, FloatArg)
        self.assertEqual(arg.ref, "s")

    def test_bool_node_arg_uses_ref(self):
        g = torch.fx.Graph()
        n = g.placeholder("b")
        n.meta["val"] = True
        arg = _to_arg_value(n)
        self.assertIsInstance(arg, BoolArg)
        self.assertEqual(arg.ref, "b")


class LiteralOutputTest(unittest.TestCase):
    def test_literal_output_preserved(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1, 7

        r = _roundtrip(M(), (torch.randn(3),))
        out_node = next(n for n in r.graph.nodes if n.op_kind == OpKind.OUTPUT)
        arg_vals = [na.arg.value for na in (out_node.inputs or [])]
        # The output node's arguments are the full ordered result list.
        self.assertEqual(len(arg_vals), 2)
        self.assertIsInstance(arg_vals[0], TensorArg)
        self.assertIsInstance(arg_vals[1], IntArg)
        self.assertEqual(arg_vals[1].value, 7)
        # The literal is not a tensor value, so it is absent from Graph.outputs.
        self.assertEqual(len(r.graph.outputs or []), 1)
        validate_method(r.method, set(r.constants.keys()))


class ArgValueDispatchTest(unittest.TestCase):
    """Each ArgumentValue variant, exercised in isolation through _to_arg_value."""

    @staticmethod
    def _node(name: str, val: object) -> torch.fx.Node:
        n = torch.fx.Graph().placeholder(name)
        n.meta["val"] = val
        return n

    def test_int_literal(self):
        v = _to_arg_value(3)
        self.assertIsInstance(v, IntArg)
        self.assertEqual(v.value, 3)
        self.assertIsNone(v.ref)

    def test_float_literal(self):
        v = _to_arg_value(1.5)
        self.assertIsInstance(v, FloatArg)
        self.assertEqual(v.value, 1.5)
        self.assertIsNone(v.ref)

    def test_bool_literal(self):
        v = _to_arg_value(True)
        self.assertIsInstance(v, BoolArg)
        self.assertTrue(v.value)
        self.assertIsNone(v.ref)

    def test_none(self):
        self.assertIsInstance(_to_arg_value(None), NoneArg)

    def test_string(self):
        v = _to_arg_value("nchw")
        self.assertIsInstance(v, StringArg)
        self.assertEqual(v.value, "nchw")

    def test_scalar_type(self):
        v = _to_arg_value(torch.float16)
        self.assertIsInstance(v, ScalarTypeArg)
        self.assertEqual(v.value, ScalarType.HALF)

    def test_int_list_literal(self):
        v = _to_arg_value([1, 2, 3])
        self.assertIsInstance(v, IntListArg)
        self.assertEqual(v.values, [1, 2, 3])

    def test_float_list_literal(self):
        v = _to_arg_value([1.0, 2.0])
        self.assertIsInstance(v, FloatListArg)
        self.assertEqual(v.values, [1.0, 2.0])

    def test_bool_list_literal(self):
        v = _to_arg_value([True, False])
        self.assertIsInstance(v, BoolListArg)
        self.assertEqual(v.values, [True, False])

    def test_int_node_uses_ref(self):
        v = _to_arg_value(self._node("s", 5))
        self.assertIsInstance(v, IntArg)
        self.assertEqual(v.ref, "s")

    def test_tensor_node_is_tensor_arg(self):
        v = _to_arg_value(self._node("t", torch.zeros(2)))
        self.assertIsInstance(v, TensorArg)
        self.assertEqual(v.name, "t")

    def test_tensor_list(self):
        v = _to_arg_value(
            [self._node("a", torch.zeros(2)), self._node("b", torch.zeros(2))]
        )
        self.assertIsInstance(v, TensorListArg)
        self.assertEqual(v.names, ["a", "b"])

    def test_optional_tensor_list(self):
        v = _to_arg_value([self._node("a", torch.zeros(2)), None])
        self.assertIsInstance(v, OptionalTensorListArg)
        self.assertEqual(v.names, ["a", ""])
        self.assertEqual(v.has_value, [True, False])
