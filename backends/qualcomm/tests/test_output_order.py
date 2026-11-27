# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections import defaultdict
from types import SimpleNamespace

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import pytest
import torch
from executorch.backends.qualcomm.builders.node_visitor import NodeVisitor
from executorch.backends.qualcomm.utils.constants import QCOM_TENSOR_NAME


def _visitor(graph, *, buffers=None, mutations=None, validation=False, dump=False):
    program = SimpleNamespace(
        graph_signature=SimpleNamespace(
            inputs_to_buffers=buffers or {},
            buffers_to_mutate=mutations or {},
            inputs_to_parameters={},
            inputs_to_lifted_tensor_constants={},
        )
    )
    return NodeVisitor(
        {node: index for index, node in enumerate(graph.nodes)},
        program,
        enable_tensor_dump=dump,
        is_qnn_partitioner=not validation,
    )


def test_reordered_same_shaped_outputs_keep_program_positions():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    add = graph.call_function(torch.add, (x, 7))
    neg = graph.call_function(torch.neg, (x,))
    graph.output((neg, add))
    module = torch.fx.GraphModule({}, graph)
    sample = torch.tensor([1.0, 2.0])
    expected = module(sample)
    visitor = _visitor(graph, dump=True)
    wrappers = defaultdict(dict)
    built = [
        visitor.define_tensor(
            node,
            node,
            value,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            wrappers,
        )
        for node, value in ((add, sample + 7), (neg, -sample))
    ]
    built = [PyQnnManager.PyQnnTensorWrapper(wrapper) for wrapper in built]
    assert [wrapper.GetName() for wrapper in built] == [
        f"output_1_{add.name}@0",
        f"output_0_{neg.name}@0",
    ]
    assert add.meta[QCOM_TENSOR_NAME][0] == built[0].GetName()
    assert neg.meta[QCOM_TENSOR_NAME][0] == built[1].GetName()
    assert expected[0].shape == expected[1].shape
    assert not torch.equal(expected[0], expected[1])


@pytest.mark.parametrize("escaping", [(2, 0), (1,), (0, 2, 1)])
def test_multi_output_getitem_is_numbered_by_escape_order(escaping):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    multi = graph.call_function(
        torch.ops.aten.native_layer_norm.default, (x, [2], None, None, 1e-5)
    )
    items = [graph.call_function(operator.getitem, (multi, i)) for i in range(3)]
    graph.output(tuple(items[i] for i in escaping))
    visitor = _visitor(graph)
    for i in range(3):
        name = visitor.get_tensor_name(multi, i)
        if i in escaping:
            assert name == f"output_{escaping.index(i)}_{multi.name}@{i}"
        else:
            assert name == f"{multi.name}@{i}"


def test_direct_and_repeated_outputs_use_first_program_position():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.neg, (x,))
    graph.output((y, y))
    assert _visitor(graph).get_tensor_name(y) == f"output_0_{y.name}@0"


def test_single_output_graph_is_numbered():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.neg, (x,))
    graph.output(y)
    assert _visitor(graph).get_tensor_name(y) == f"output_0_{y.name}@0"


def test_mutable_buffer_names_remain_unchanged():
    graph = torch.fx.Graph()
    cache = graph.placeholder("b_cache")
    x = graph.placeholder("x")
    updated = graph.call_function(torch.add, (cache, x))
    ordinary = graph.call_function(torch.neg, (x,))
    graph.output((updated, ordinary))
    visitor = _visitor(
        graph, buffers={cache.name: "cache"}, mutations={updated.name: "cache"}
    )
    assert visitor.get_tensor_name(cache) == f"input_0_mutbuf_0_{cache.name}@0"
    assert visitor.get_tensor_name(updated) == f"output_mutbuf_0_{updated.name}@0"
    assert visitor.get_tensor_name(ordinary) == f"output_1_{ordinary.name}@0"


def test_op_validation_does_not_publish_output_names():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.neg, (x,))
    graph.output((y,))
    assert _visitor(graph, validation=True).get_tensor_name(y) == f"{y.name}@0"
