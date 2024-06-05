# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.dim_order_utils import (
    get_dim_order,
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)

from executorch.exir.pass_base import ExportPass, ProxyValue

from torch.export import export
from torch.testing import FileCheck
from torch.utils._pytree import tree_flatten


@dataclass
class MemoryFormatTestSet:
    module: torch.nn.Module
    sample_input: Tuple[Any, ...]
    target_memory_format: torch.memory_format
    is_aten_mode: bool


class SimpleToCopyContiguousModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.double, memory_format=torch.contiguous_format)


class SimpleToCopyChannelsLastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.double, memory_format=torch.channels_last)


class PropagateToCopyChannalsLastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = x.to(dtype=torch.double, memory_format=torch.channels_last)
        t2 = t1 + t1
        return t1 * t2


class TestMemoryFormatOpsPass(unittest.TestCase):
    def memory_format_test_runner(self, test_set: MemoryFormatTestSet):
        aten_op_str = "torch.ops.aten._to_copy.default"
        edge_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        before = export(test_set.module, test_set.sample_input)

        # check op strings before
        FileCheck().check_count(aten_op_str, 1, exactly=True).check_not(
            edge_op_str
        ).run(before.graph_module.code)

        epm = to_edge(before)

        # check op strings
        FileCheck().check_not(aten_op_str).check_count(
            edge_op_str, 1, exactly=True
        ).run(epm.exported_program().graph_module.code)

        # check EdgeOp and the new BackendOp should behave the same
        expected = before.module()(*test_set.sample_input)
        actual = epm.exported_program().module()(*test_set.sample_input)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(
            is_channel_last_dim_order(actual),
            is_channel_last_dim_order(expected),
        )
        if test_set.target_memory_format == torch.channels_last:
            self.assertTrue(is_channel_last_dim_order(actual))
        elif test_set.target_memory_format == torch.contiguous_format:
            self.assertTrue(is_contiguous_dim_order(actual))
        else:
            raise RuntimeError("Unknown memory format")

        # check EdgeOp and the new BackendOp should behave the same in the runtime
        executorch_prog = epm.to_executorch()

        if test_set.is_aten_mode:
            from executorch.extension.pybindings.aten_lib import (  # @manual
                _load_for_executorch_from_buffer,
            )
        else:
            from executorch.extension.pybindings.portable_lib import (  # @manual
                _load_for_executorch_from_buffer,
            )

        executorch_module = _load_for_executorch_from_buffer(executorch_prog.buffer)
        inputs_flattened = tree_flatten(test_set.sample_input)[0]
        runtime_output = executorch_module.run_method(
            "forward", tuple(inputs_flattened)
        )[0]
        self.assertTrue(torch.allclose(runtime_output, expected))
        self.assertEqual(
            is_channel_last_dim_order(runtime_output),
            is_channel_last_dim_order(expected),
        )

    def test_op_to_copy_replacement_2d(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=False,
            )
        )

    def test_op_to_copy_replacement_2d_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=True,
            )
        )

    def test_op_to_copy_replacement_4d(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=False,
            )
        )

    def test_op_to_copy_replacement_4d_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=True,
            )
        )

    def test_op_dim_order_update(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyChannelsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=False,
            ),
        )

    def test_op_dim_order_update_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyChannelsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=True,
            ),
        )

    def test_op_dim_order_propagation(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=PropagateToCopyChannalsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=False,
            )
        )

    def test_op_dim_order_propagation_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=PropagateToCopyChannalsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=True,
            )
        )

    def test_dim_order_replacement(self) -> None:
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                t1 = x + y
                t2 = t1 * x
                t3 = t2 + y
                return t3

        class MulIOToChannelsLastPass(ExportPass):
            """
            This pass updates the dim order of the input tensors of mul op from contiguous to channels_last, and change the output dim order back to contiguous.
            """

            def call_operator(self, op, args, kwargs, meta):
                if not (
                    isinstance(op, EdgeOpOverload) and op.__name__ == "aten.mul.Tensor"
                ):
                    return super().call_operator(
                        op,
                        args,
                        kwargs,
                        meta,
                    )

                # new kwargs with dim_order, and no memory_format for the new op
                nkwargs = dict(copy.deepcopy(kwargs))  # orig kwargs are immutable

                _to_dim_order_copy_op = (
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default
                )

                # update the dim order of the input tensors of mul op from contiguous to channels_last
                new_args = []
                for arg in args:
                    # get the number of dimmention of mul op's input, as well as its dtype
                    ndim = None
                    dtype = None
                    tensor_input = None
                    # can always get the shape, assuming rank is specialized
                    if isinstance(arg, ProxyValue) and arg.is_tensor():
                        ndim = arg.to_tensor().dim()
                        dtype = arg.to_tensor().dtype
                        tensor_input = arg.to_tensor()

                    elif isinstance(arg, torch.Tensor):
                        ndim = arg.dim()
                        dtype = arg.dtype
                        tensor_input = arg
                    else:
                        assert (
                            0
                        ), f"Expecting a Tensor or a ProxyValue buy got {type(arg)}"

                    ndim = tensor_input.dim()
                    dtype = tensor_input.dtype

                    assert tensor_input.is_contiguous(
                        memory_format=torch.contiguous_format
                    ), "mul input should be in contiguous memory format"

                    channels_last_dim_order = get_dim_order(torch.channels_last, ndim)
                    contiguous_dim_order = get_dim_order(torch.contiguous_format, ndim)

                    # convert the input tensors to channels_last
                    arg_channels_last = super().call_operator(
                        _to_dim_order_copy_op,
                        (arg,),
                        {"dtype": dtype, "dim_order": channels_last_dim_order},
                        meta,
                    )

                    new_args.append(arg_channels_last)

                new_args = tuple(new_args)

                # call the mul op with the self tensor in channels_last
                # mul op is using the same kernel to handle contiguous and channels_last inputs.
                mul_out = super().call_operator(
                    op,
                    new_args,
                    nkwargs,
                    meta,
                )

                # convert the mul op output to contiguous
                mul_out_contiguous = super().call_operator(
                    _to_dim_order_copy_op,
                    (mul_out,),
                    {"dtype": dtype, "dim_order": contiguous_dim_order},
                    meta,
                )

                return mul_out_contiguous

        toy_model = ToyModel()
        sample_input = (
            torch.randn([2, 2, 2, 2], dtype=torch.float32),
            torch.randn([2, 2, 2, 2], dtype=torch.float32),
        )

        _to_dim_order_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        before_epm = to_edge(export(toy_model, sample_input))

        # should not contain _to_dim_order_copy op
        FileCheck().check_not(_to_dim_order_op_str).run(
            before_epm.exported_program().graph_module.code
        )

        # add the pass to update the input and output of mul op from contiguous to channels_last
        updated_epm = before_epm.transform([MulIOToChannelsLastPass()])

        # should contain exactly three _to_dim_order_copy op to update the dim order of input and output of mul op
        FileCheck().check_count(_to_dim_order_op_str, 3, exactly=True).run(
            updated_epm.exported_program().graph_module.code
        )

        # check original graph and update graph should have same result
        expected = before_epm.exported_program().module()(*sample_input)
        actual = updated_epm.exported_program().module()(*sample_input)
        self.assertTrue(torch.allclose(actual, expected))

        self.assertTrue(is_contiguous_dim_order(actual))
        self.assertTrue(is_contiguous_dim_order(expected))
