# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from typing import Union

import torch

import torchvision
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.dim_order_utils import (
    get_dim_order,
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)
from executorch.exir.pass_base import ExportPass, ProxyValue

from executorch.exir.tests.test_memory_format_ops_pass_utils import (
    MemoryFormatOpsPassTestUtils,
    MemoryFormatTestSet,
    PropagateToCopyChannalsLastModule,
    SimpleToCopyChannelsLastModule,
    SimpleToCopyContiguousModule,
)

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)

from torch.export import export
from torch.testing import FileCheck


class TestMemoryFormatOpsPass(unittest.TestCase):
    def test_op_to_copy_replacement_2d(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_to_copy_replacement_4d(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_dim_order_update(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
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
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_dim_order_propagation(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
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
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    # Only test dim order replacement result in lean mode test.
    # This test is irrelevant with operator mode.
    def test_dim_order_replacement(self) -> None:
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                t1 = x + y
                t2 = t1 * x
                t3 = t2 + y
                return t3

        def grab_tensor(arg: Union[ProxyValue, torch.Tensor]):
            t = (
                arg.to_tensor()
                if isinstance(arg, ProxyValue) and arg.is_tensor()
                else arg
            )
            assert isinstance(
                t, torch.Tensor
            ), f"Expecting a Tensor or a ProxyValue but got {type(t)}"
            return t

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
                    # can always get the shape, assuming rank is specialized
                    tensor_input = grab_tensor(arg)
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

        class MulIOCheckChannelsLastPass(ExportPass):
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

                # check if the dim order of the input tensors of mul op is channels_last
                for arg in args:
                    # can always get the shape, assuming rank is specialized
                    tensor_input = grab_tensor(arg)

                    assert is_channel_last_dim_order(tensor_input)

                # call the mul op with the self tensor in channels_last
                # mul op is using the same kernel to handle contiguous and channels_last inputs.
                mul_out = super().call_operator(
                    op,
                    args,
                    nkwargs,
                    meta,
                )

                # check if the dim order of the mul op output is channels_last
                output_tensor = grab_tensor(mul_out)
                assert is_channel_last_dim_order(output_tensor)

                return mul_out

        toy_model = ToyModel()
        sample_input = (
            torch.randn([2, 2, 2, 2], dtype=torch.float32),
            torch.randn([2, 2, 2, 2], dtype=torch.float32),
        )

        _to_dim_order_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        before_epm = to_edge(
            export(toy_model, sample_input),
            compile_config=EdgeCompileConfig(_skip_dim_order=False),
        )

        # should not contain _to_dim_order_copy op
        FileCheck().check_not(_to_dim_order_op_str).run(
            before_epm.exported_program().graph_module.code
        )

        # add the pass to update the input and output of mul op from contiguous to channels_last
        updated_epm = before_epm.transform([MulIOToChannelsLastPass()])

        # should contain exactly three _to_dim_order_copy op to update the dim order of both inputs and output of mul op
        FileCheck().check_count(_to_dim_order_op_str, 3, exactly=True).run(
            updated_epm.exported_program().graph_module.code
        )

        # add the pass to check the dim order of the input and output of mul op are channels_last
        updated_epm = updated_epm.transform([MulIOCheckChannelsLastPass()])

        # check original graph and update graph should have same result
        expected = before_epm.exported_program().module()(*sample_input)
        actual = updated_epm.exported_program().module()(*sample_input)
        self.assertTrue(torch.allclose(actual, expected))

        self.assertTrue(is_contiguous_dim_order(actual))
        self.assertTrue(is_contiguous_dim_order(expected))

    def test_resnet18(self) -> None:
        model = torchvision.models.resnet18()
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )

    def test_resnet18_xnnpack(self) -> None:
        model = torchvision.models.resnet18()
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                use_xnnpack=True,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )

    def test_mobilenet_v3(self) -> None:
        model = torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )

    def test_mobilenet_v3_xnnpack(self) -> None:
        model = torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                use_xnnpack=True,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )
