# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.transpose_options import (
    Transpose,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.ir.edge_passes.test_remove_io_quant_ops_pass import (
    MultiInputOutputModule,
)


def test_topological_sorting(mocker):
    # This model requires the insertion of a `Transpose` operator in place of a `dequantize` node, which breaks the
    #  topological order.
    model = MultiInputOutputModule()

    # Capture the operators before and after topological sorting.
    original_sort_fn = ModelBuilder.sort_operators_topologically
    captured_ops = {}

    def mock_sort_fn(self: ModelBuilder):
        captured_ops["pre_sort"] = deepcopy(self.get_operators().vector)
        original_sort_fn(self)  # type: ignore
        captured_ops["post_sort"] = deepcopy(self.get_operators().vector)

    mocker.patch.object(
        ModelBuilder,
        "sort_operators_topologically",
        autospec=True,
        side_effect=mock_sort_fn,
    )

    input_shapes = [(1, 4, 32, 32), (1, 1, 1, 31)]
    to_quantized_edge_program(model, input_shapes)

    ops_pre_sort = captured_ops["pre_sort"]
    ops_post_sort = captured_ops["post_sort"]
    assert len(ops_pre_sort) == len(ops_post_sort)

    # Before sorting, the operator on index `2` should be an identity `Transpose`, which uses the output of the
    #  operator on index `3` (breaking the topological order).
    assert isinstance(ops_pre_sort[2].builtin_options, Transpose)
    assert all(ops_pre_sort[2].tmp_inputs[1].tmp_buffer.data == [0, 1, 2, 3])
    assert ops_pre_sort[2].tmp_inputs[0] == ops_pre_sort[3].tmp_outputs[0]

    # After the sort, the operators on indices `2` and `3` are swapped.
    assert (
        ops_post_sort[2].builtin_options is None
    )  # A Relu, which doesn't have a `BuiltinOptions` in Neutron IR.
    assert isinstance(ops_post_sort[3].builtin_options, Transpose)
    assert all(ops_post_sort[3].tmp_inputs[1].tmp_buffer.data == [0, 1, 2, 3])
    assert ops_post_sort[2].tmp_outputs[0] == ops_post_sort[3].tmp_inputs[0]

    # Make sure all nodes follow topological order.
    tensor_to_producer_op = {
        output_tensor: op for op in ops_post_sort for output_tensor in op.tmp_outputs
    }
    for op_idx, op in enumerate(ops_post_sort):
        for input_tensor in op.tmp_inputs:
            if input_tensor in tensor_to_producer_op:
                assert ops_post_sort.index(tensor_to_producer_op[input_tensor]) < op_idx
