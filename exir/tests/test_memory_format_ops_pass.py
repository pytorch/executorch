# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from typing import Union
from unittest.mock import patch

import torch

import torchvision
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.dim_order_utils import (
    get_dim_order,
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)
from executorch.exir.pass_base import ExportPass, ProxyValue

from executorch.exir.tests.test_memory_format_ops_pass_utils import (
    AmbiguousDimOrderError,
    MemoryFormatOpsPassTestUtils,
    MemoryFormatTestSet,
    PropagateToCloneChannelsLastModule,
    PropagateToCopyChannalsLastModule,
    SimpleCloneChannelsLastModule,
    SimpleCloneContiguousModule,
    SimpleEmptyChannelLastModule,
    SimpleEmptyContiguoustModule,
    SimpleToCopyChannelsLastModule,
    SimpleToCopyContiguousModule,
)

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)

from torch._export.verifier import SpecViolationError
from torch.export import Dim, export, ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import create_op_support
from torch.testing import FileCheck


class NoOpPartitioner(Partitioner):
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(exported_program, {})


class TestMemoryFormatOpsPass(unittest.TestCase):
    def test_op_to_copy_replacement_2d(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                op=torch.ops.aten._to_copy.default,
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
                op=torch.ops.aten._to_copy.default,
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_empty_replacement_channels_last(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleEmptyChannelLastModule().eval(),
                op=torch.ops.aten.empty.memory_format,
                sample_input=(torch.randn((1, 10, 24, 24), dtype=torch.float32),),
                target_memory_format=torch.channels_last,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_empty_replacement_contiguous(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleEmptyContiguoustModule().eval(),
                op=torch.ops.aten.empty.memory_format,
                sample_input=(torch.randn((1, 10, 24, 24), dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_empty_dynamic_size(self) -> None:
        class EmptyCopy(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.empty(x.shape[0], 3)
                out.copy_(x)
                return out

        ep = export(
            EmptyCopy(),
            (torch.randn(2, 3),),
            dynamic_shapes=({0: Dim("batch", max=8)},),
        )
        program = to_edge(ep).to_executorch()
        module = _load_for_executorch_from_buffer(program.buffer)
        for batch in (2, 5):
            x = torch.randn(batch, 3)
            self.assertTrue(torch.equal(module.run_method("forward", (x,))[0], x))

    def test_op_empty_data_dependent_size(self) -> None:
        class EmptyCopy(torch.nn.Module):
            def forward(self, n: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                size = n.item()
                torch._check(size >= 2)
                torch._check(size <= 8)
                return torch.empty(size, 3).copy_(y)

        y = torch.randn(1, 3)
        program = to_edge(export(EmptyCopy(), (torch.tensor(5), y))).to_executorch()
        module = _load_for_executorch_from_buffer(program.buffer)
        for size in (2, 5):
            output = module.run_method("forward", (torch.tensor(size), y))[0]
            self.assertTrue(torch.equal(output, y.expand(size, 3)))

    def test_op_empty_like_follows_layout_of_input(self) -> None:
        class EmptyLikeCopy(torch.nn.Module):
            def __init__(self, transform=lambda x: x, **kwargs):
                super().__init__()
                self.transform = transform
                self.kwargs = kwargs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.transform(x)
                out = torch.empty_like(y, **self.kwargs)
                out.copy_(y)
                return out

        channels_last = torch.arange(144, dtype=torch.float32).reshape(2, 3, 4, 6)
        channels_last = channels_last.contiguous(memory_format=torch.channels_last)
        permuted = torch.arange(144, dtype=torch.float32).reshape(2, 4, 6, 3)
        cases = {
            "channels_last_input": (EmptyLikeCopy(), [channels_last], None),
            # With one channel the strides alone do not say channels last.
            "single_channel": (
                EmptyLikeCopy(),
                [torch.randn(2, 1, 4, 6).to(memory_format=torch.channels_last)],
                None,
            ),
            # A no-op slice is removed, so the buffer follows the input.
            "noop_slice": (EmptyLikeCopy(lambda x: x[:]), [channels_last], None),
            # Channels last strides in eager, a contiguous copy in Edge.
            "permuted_view": (
                EmptyLikeCopy(lambda x: x.permute(0, 3, 1, 2)),
                [permuted],
                None,
            ),
            "explicit_preserve_format": (
                EmptyLikeCopy(
                    lambda x: x.permute(0, 3, 1, 2), memory_format=torch.preserve_format
                ),
                [permuted],
                None,
            ),
            # Channels last in Edge, made contiguous later by to_executorch.
            "avg_pool2d": (
                EmptyLikeCopy(lambda x: torch.nn.functional.avg_pool2d(x, 2)),
                [channels_last],
                None,
            ),
            "dynamic_batch": (
                EmptyLikeCopy(),
                [torch.randn(2, 3), torch.randn(5, 3)],
                ({0: Dim("batch", max=8)},),
            ),
            "integer_input": (EmptyLikeCopy(), [torch.arange(6).reshape(2, 3)], None),
            "dtype_argument": (
                EmptyLikeCopy(dtype=torch.int32),
                [torch.arange(6.0).reshape(2, 3)],
                None,
            ),
        }
        for name, (model, inputs, dynamic_shapes) in cases.items():
            with self.subTest(name):
                ep = export(model, (inputs[0],), dynamic_shapes=dynamic_shapes)
                edge = to_edge(ep)
                (copy_node,) = [
                    node
                    for node in edge.exported_program().graph_module.graph.nodes
                    if node.target == exir_ops.edge.aten.copy.default
                ]
                buffer, source = copy_node.args[:2]
                self.assertEqual(
                    buffer.meta["val"].stride(), source.meta["val"].stride()
                )
                program = edge.to_executorch()
                module = _load_for_executorch_from_buffer(program.buffer)
                for x in inputs:
                    output = module.run_method("forward", (x,))[0]
                    torch.testing.assert_close(output, model(x), rtol=0, atol=0)

    def test_op_empty_like_does_not_depend_on_input(self) -> None:
        class EmptyLikeCopy(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x + x
                out = torch.empty_like(y)
                out.copy_(y)
                return out * y

        edge = to_edge(export(EmptyLikeCopy(), (torch.randn(2, 3),)))
        graph_module = edge.exported_program().graph_module
        empty = exir_ops.edge.dim_order_ops._empty_dim_order.default
        support = create_op_support(
            lambda _, node: node.op == "call_function" and node.target != empty
        )
        partitions = CapabilityBasedPartitioner(
            graph_module, support
        ).propose_partitions()
        # One partition holds the add, the copy and the mul around it.
        self.assertEqual([len(p.nodes) for p in partitions], [3])

    def test_op_empty_like_with_explicit_memory_format(self) -> None:
        class EmptyLike(torch.nn.Module):
            def __init__(self, memory_format):
                super().__init__()
                self.memory_format = memory_format

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(x, memory_format=self.memory_format)

        lowerings = {
            "to_edge": to_edge,
            # Keeps empty_like until ReplaceEmptyLikePass.
            "to_edge_transform_and_lower": lambda ep: to_edge_transform_and_lower(
                ep, partitioner=[NoOpPartitioner()]
            ),
            # With partitioners the config's preserve_ops is not applied, as on main.
            "config_preserve_ops": lambda ep: to_edge_transform_and_lower(
                ep,
                partitioner=[NoOpPartitioner()],
                compile_config=EdgeCompileConfig(
                    preserve_ops=[torch.ops.aten.empty_like.default]
                ),
            ),
        }
        x = torch.randn(2, 3, 4, 5)
        for memory_format in (torch.contiguous_format, torch.channels_last):
            for source in (x, x.to(memory_format=torch.channels_last)):
                for name, lower in lowerings.items():
                    with self.subTest(
                        name, format=memory_format, source=source.stride()
                    ):
                        edge = lower(export(EmptyLike(memory_format), (source,)))
                        (node,) = [
                            n
                            for n in edge.exported_program().graph.nodes
                            if n.target
                            == exir_ops.edge.dim_order_ops._empty_dim_order.default
                        ]
                        self.assertEqual(
                            node.kwargs["dim_order"], get_dim_order(memory_format, 4)
                        )

    def test_op_empty_like_does_not_force_decomposition(self) -> None:
        class EmptyLike(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(x)

        ep = export(EmptyLike(), (torch.randn(2, 3),))
        with patch.object(
            ExportedProgram,
            "run_decompositions",
            autospec=True,
            side_effect=ExportedProgram.run_decompositions,
        ) as run_decompositions:
            to_edge_transform_and_lower(
                ep, partitioner=[NoOpPartitioner(), NoOpPartitioner()]
            )
        # Only the first call, which functionalizes the graph.
        self.assertEqual(run_decompositions.call_count, 1)

    def test_op_empty_like_layout_without_memory_format(self) -> None:
        class EmptyLike(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.kwargs = kwargs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(x, **self.kwargs)

        cases = {
            "transposed": (torch.randn(4, 3).t(), [1, 0]),
            "channels_last_3d": (
                torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),
                [0, 2, 3, 4, 1],
            ),
        }
        config = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
        for name, (x, layout) in cases.items():
            for kwargs in ({}, {"memory_format": torch.preserve_format}):
                with self.subTest(name, **kwargs):
                    model = EmptyLike(**kwargs)
                    edge = to_edge(export(model, (x,)), compile_config=config)
                    (node,) = [
                        n
                        for n in edge.exported_program().graph.nodes
                        if n.target == exir_ops.edge.aten.empty_permuted.default
                    ]
                    self.assertEqual(node.args[1], layout)

                    with self.assertRaises(SpecViolationError):
                        to_edge(export(model, (x,)))

    def test_op_empty_like_kept_when_preserved(self) -> None:
        class EmptyLike(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(x)

        config = EdgeCompileConfig(preserve_ops=[torch.ops.aten.empty_like.default])
        lowerings = {
            "to_edge": to_edge,
            "to_edge_transform_and_lower": to_edge_transform_and_lower,
        }
        for name, lower in lowerings.items():
            with self.subTest(name):
                edge = lower(
                    export(EmptyLike(), (torch.randn(2, 3),)), compile_config=config
                )
                targets = [n.target for n in edge.exported_program().graph.nodes]
                self.assertIn(exir_ops.edge.aten.empty_like.default, targets)

    def test_op_empty_permuted_after_pytorch_decompositions(self) -> None:
        class EmptyLikeCopy(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                out.copy_(x)
                return out

        model = EmptyLikeCopy()
        x = torch.randn(2, 3)
        program = to_edge(export(model, (x,)).run_decompositions()).to_executorch()
        output = _load_for_executorch_from_buffer(program.buffer).run_method(
            "forward", (x,)
        )[0]
        self.assertTrue(torch.equal(output, model(x)))

        # Its eager channels last layout may not match the layout in Edge.
        x = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)
        with self.assertRaises(SpecViolationError):
            to_edge(export(model, (x,)).run_decompositions())

    def test_op_clone_replacement_contiguous(self) -> None:
        model = SimpleCloneContiguousModule()
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                op=torch.ops.aten.clone.default,
                sample_input=(
                    torch.randn((3, 4, 5, 6)).to(memory_format=torch.channels_last),
                ),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_clone_replacement_channels_last(self) -> None:
        model = SimpleCloneChannelsLastModule()
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                op=torch.ops.aten.clone.default,
                sample_input=(
                    torch.randn((3, 4, 5, 6)).to(memory_format=torch.contiguous_format),
                ),
                target_memory_format=torch.channels_last,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_dim_order_update(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleToCopyChannelsLastModule().eval(),
                op=torch.ops.aten._to_copy.default,
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
                op=torch.ops.aten._to_copy.default,
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
            check_unambiguous_dim_order=True,
        )

    def test_op_clone_dim_order_propagation(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=PropagateToCloneChannelsLastModule().eval(),
                op=torch.ops.aten.clone.default,
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
            check_unambiguous_dim_order=True,
        )

    def test_op_dim_order_propagation_ambiguous(self) -> None:
        try:
            MemoryFormatOpsPassTestUtils.memory_format_test_runner(
                self,
                MemoryFormatTestSet(
                    module=PropagateToCopyChannalsLastModule().eval(),
                    op=torch.ops.aten._to_copy.default,
                    sample_input=(
                        torch.rand_like(
                            torch.zeros(
                                [2, 1, 2, 2]
                            ),  # Ambiguous shape should trigger AmbiguousDimOrderError!
                            dtype=torch.float32,
                            memory_format=torch.contiguous_format,
                        ),
                    ),
                    target_memory_format=torch.channels_last,
                    _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                ),
                check_unambiguous_dim_order=True,
            )
            AssertionError("Should have raised AmbiguousDimOrderError")
        except AmbiguousDimOrderError:
            pass  # Expected error

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
            export(toy_model, sample_input, strict=True),
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
                op=torch.ops.aten._to_copy.default,
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
                op=torch.ops.aten._to_copy.default,
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
                op=torch.ops.aten._to_copy.default,
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
                op=torch.ops.aten._to_copy.default,
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                use_xnnpack=True,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )
