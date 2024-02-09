# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401
import functools
import inspect
import os
import random
import unittest
from typing import Callable, Dict, Optional, Tuple, Type
from unittest import skip, skipUnless

import executorch.exir as exir

import executorch.exir.control_flow as control_flow

# @manual=//executorch/extension/pytree:pybindings
import executorch.extension.pytree as pytree
import torch

from executorch.exir import (
    CaptureConfig,
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    memory,
)
from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode
from executorch.exir.emit import emit_program
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import (
    DebugPass,
    MemoryPlanningPass,
    to_scratch_op_pass,
    ToOutVarPass,
)
from executorch.exir.print_program import pretty_print, print_program
from executorch.exir.tensor import make_tensor_value, TensorSpec
from executorch.exir.tests.control_flow_models import (
    FTCondBasic,
    FTCondDynShape,
    FTMapBasic,
    FTMapDynShape,
)
from executorch.exir.tests.dynamic_shape_models import BatchNormModel

from executorch.exir.tests.transformer import Transformer
from functorch.experimental.control_flow import cond

kernel_mode = None  # either aten mode or lean mode
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_bundled_program_from_buffer,
        _load_for_executorch_from_buffer,
        _load_for_executorch_from_bundled_program,
    )

    kernel_mode = "lean"
except ImportError as e:
    print(e)
    pass

try:
    from executorch.extension.pybindings.aten_lib import (
        _load_bundled_program_from_buffer,
        _load_for_executorch_from_buffer,
        _load_for_executorch_from_bundled_program,
    )

    assert kernel_mode is None
    kernel_mode = "aten"
except ImportError as e:
    print(e)
    pass

assert kernel_mode is not None

is_aten_mode = kernel_mode == "aten"
is_lean_mode = kernel_mode == "lean"

from torch import nn
from torch.utils import _pytree as torch_pytree

from .exported_module import ExportedModule


RUN_SKIPPED = int(os.environ.get("RUN_SKIPPED", "0"))


class ModuleBasic(nn.Module):
    def __init__(self):
        super(ModuleBasic, self).__init__()

    def forward(self, x):
        return torch.sin(x).max()

    def get_random_inputs(self):
        return (torch.randn(100),)


class ModuleOpsReturnMulti(nn.Module):
    def __init__(self):
        super(ModuleOpsReturnMulti, self).__init__()

    def forward(self, a, b):
        x, y = torch.topk(a, 3)
        return x * 2 + b

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(3))


class ModuleAdd(nn.Module):
    def __init__(self):
        super(ModuleAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2))


class ModuleFloatAddWithAlpha(nn.Module):
    def __init__(self):
        super(ModuleFloatAddWithAlpha, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: float):
        return torch.add(x, y, alpha=c)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2), random.random())


class ModuleIntAddWithAlpha(nn.Module):
    def __init__(self):
        super(ModuleIntAddWithAlpha, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: int):
        return torch.add(x, y, alpha=c)

    def get_random_inputs(self):
        return (
            torch.randint(0, 10, (2, 2)),
            torch.randint(0, 10, (2, 2)),
            random.randint(0, 10),
        )


class ModuleContainers(nn.Module):
    def __init__(self):
        super(ModuleContainers, self).__init__()

    def forward(self, d):
        a = d["a"]
        b = d["b"]
        return {"inputs": (a, b), "c": torch.add(a, b)}

    def get_random_inputs(self):
        return ({"a": torch.randn(2, 2), "b": torch.randn(2, 2)},)


class ToyModelForMemPlanning(nn.Module):
    def __init__(self):
        super(ToyModelForMemPlanning, self).__init__()

    def forward(self, a, b):
        o = a
        for i in range(3):
            o = o * a
            o = o + b
        return o

    def get_random_inputs(self):
        return (
            torch.randn(10),
            torch.randn(10),
        )


class MemPlanningWithScratchTensor(nn.Module):
    def __init__(self):
        super(MemPlanningWithScratchTensor, self).__init__()
        self.linear1 = nn.Linear(4, 2)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, a, b):
        o1 = self.linear1(a)
        o2 = self.linear2(b)
        return o1 + o2

    def get_random_inputs(self):
        return (
            torch.randn(10, 4),
            torch.randn(10, 4),
        )


class ModuleOpsReturnTensorList(nn.Module):
    def __init__(self):
        super(ModuleOpsReturnTensorList, self).__init__()

    def forward(self, x):
        split = torch.ops.aten.tensor_split.sections(x, 3)
        return split[0]

    def get_random_inputs(self):
        return (torch.randn(100),)


class ModuleReturnInput(nn.Module):
    def __init__(self):
        super(ModuleReturnInput, self).__init__()

    def forward(self, x):
        return (x, x, {"x": x, "y": x}, [x, x, x])

    def get_random_inputs(self):
        return (torch.randn(1),)


class ModuleIfElse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, c, x):
        x = x * x

        def addloop(x, n):
            out = x
            for _ in range(n - 1):
                out = out + x
            return out

        def true_branch(c, x):
            return addloop(x, 3)

        def false_branch(c, x):
            return addloop(x, 4)

        y = cond(c, true_branch, false_branch, (c, x))
        return y * y

    def get_random_inputs(self):
        return (torch.randint(2, [1]) == 0, torch.randn(10))


class ModuleIfElseWithBoolInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, c: bool, x: torch.Tensor):
        x = x * x

        def addloop(x, n):
            out = x
            for _ in range(n - 1):
                out = out + x
            return out

        def true_branch(c, x):
            return addloop(x, 3)

        def false_branch(c, x):
            return addloop(x, 4)

        y = cond(c, true_branch, false_branch, (c, x))

        return y * y

    def get_random_inputs(self):
        return (random.randint(0, 1) == 0, torch.randn(10))


class ModuleWhileIf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, accum, cnt):
        @control_flow.tracing_context(
            inputs=(torch.zeros([1]).to(dtype=torch.long), torch.randint(10, 100, [1]))
        )
        def loop_cond(accum, cnt):
            return cnt != torch.zeros([1]).to(dtype=torch.long)

        @control_flow.tracing_context(
            inputs=(torch.zeros([1]).to(dtype=torch.long), torch.randint(10, 100, [1]))
        )
        def loop_body(accum, cnt):
            # return accum + cnt, cnt - torch.ones([1]).to(dtype=torch.long)
            @control_flow.tracing_context(
                inputs=(torch.zeros([1]).to(dtype=torch.long),)
            )
            def true_branch(cnt):
                return cnt

            @control_flow.tracing_context(
                inputs=(torch.zeros([1]).to(dtype=torch.long),)
            )
            def false_branch(cnt):
                return torch.zeros([1], dtype=torch.long)

            accum = accum + cond(
                torch.BoolTensor([True]), true_branch, false_branch, (cnt,)
            )
            # 'cnt - 1' does not work yet since the runtime does not expect
            # tensor to be mixed with scalar for sub op.
            return accum, cnt - torch.ones([1]).to(dtype=torch.long)

        y, _ = control_flow.while_loop(
            loop_cond,
            loop_body,
            (accum, cnt),
        )
        return y

    def get_random_inputs(self):
        return (torch.zeros([1]).to(dtype=torch.long), torch.randint(10, 100, [1]))


class ModuleIfWhile(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, accum, cnt):
        @control_flow.tracing_context(
            inputs=(torch.zeros([1]).to(dtype=torch.long), torch.randint(10, 100, [1]))
        )
        def true_branch(accum, cnt):
            @control_flow.tracing_context(
                inputs=(
                    torch.zeros([1]).to(dtype=torch.long),
                    torch.randint(10, 100, [1]),
                )
            )
            def loop_cond(accum, cnt):
                return cnt != torch.zeros([1]).to(dtype=torch.long)

            @control_flow.tracing_context(
                inputs=(
                    torch.zeros([1]).to(dtype=torch.long),
                    torch.randint(10, 100, [1]),
                )
            )
            def loop_body(accum, cnt):
                return accum + cnt, cnt - torch.ones([1]).to(dtype=torch.long)

            return control_flow.while_loop(loop_cond, loop_body, (accum, cnt))

        @control_flow.tracing_context(
            inputs=(torch.zeros([1]).to(dtype=torch.long), torch.randint(10, 100, [1]))
        )
        def false_branch(accum, cnt):
            return accum, cnt

        return cond(torch.BoolTensor([True]), true_branch, false_branch, (accum, cnt))[
            0
        ]

    def get_random_inputs(self):
        return (torch.zeros([1]).to(dtype=torch.long), torch.randint(10, 100, [1]))


class ModuleContiguousTensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 32)

    def forward(self, arg):
        return self.linear(arg)

    def get_random_inputs(self):
        return (torch.randn(3, 8),)


class ModuleInputDynamicShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for i in range(4):
            x = x + x
            x = x * x
        return x

    def get_upper_bound_inputs(self):
        return (torch.randn(10),)

    def get_random_inputs(self):
        n = random.randint(1, 10)
        return (torch.randn(n),)


class ModuleIntermediateDynamicShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * x

        # We should use x[torch.nonzero(x)] ideally, but index op is not supported
        # in the runtime so far.
        x = torch.nonzero(x)
        return x + x

    def get_random_inputs(self):
        return (torch.randint(0, 2, (10,), dtype=torch.float),)


def allclose(lhs, rhs, rtol=1e-5, atol=1e-8):
    r"""
    Unlike torch.allocse which only handles Tensor arguments, allclose handles
    list, tuple, dict and nesting of these as well.
    """
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return torch.allclose(lhs, rhs, rtol, atol)
    if isinstance(lhs, (tuple, list)) and isinstance(rhs, (tuple, list)):
        return len(lhs) == len(rhs) and all(
            allclose(a, b, rtol, atol) for a, b in zip(lhs, rhs)
        )
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        lhs_keys = set(lhs.keys())
        rhs_keys = set(rhs.keys())
        if lhs_keys != rhs_keys:
            return False
        return all(allclose(lhs[k], rhs[k], rtol, atol) for k in lhs)
    else:
        raise RuntimeError(
            f"Unexpected types: lhs type {type(lhs)}, rhs type {type(rhs)}"
        )


def validate_contiguous_tensors(program):
    def _is_contiguous_tensor(tensor: exir.schema.Tensor):
        """
        Ensure the tensor is pytorch contigous (torch.memory_format=torch.contiguous)
        since the runtime can not handle non-contiguous tensors so far.
        """
        sizes = tensor.sizes
        dim_order = tensor.dim_order
        assert len(sizes) == len(dim_order)
        for i, val in enumerate(dim_order):
            if i != val:
                return False
        return True

    for execution_plan in program.execution_plan:
        for value in execution_plan.values:
            if isinstance(value.val, exir.schema.Tensor):
                assert _is_contiguous_tensor(
                    value.val
                ), f"Non-contiguous tensor found: size {value.val.sizes} stride {value.val.strides}. constant_buffer_idx {value.val.constant_buffer_idx}. allocation_info {value.val.allocation_info}."


class BoundMethod(object):
    def __init__(self, instance, callable):
        self._instance = instance
        self._callable = callable

    def __call__(self, *args, **kwargs):
        return self._callable(self.instance, *args, **kwargs)


def maketest(
    module_cls: Type[nn.Module],
    niter: int = 10,
    run_executor: bool = True,
    do_tree_flatten: bool = False,
    run_graph_module: bool = True,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    ignore_to_out_var_failure: bool = False,
    allow_non_contiguous_tensor: bool = False,
    method: str = "forward",
    dynamic_memory_planning_mode: DynamicMemoryPlanningMode = DynamicMemoryPlanningMode.UPPER_BOUND,
    capture_config=None,
    verify_graph: Optional[Callable] = None,
) -> Callable[[unittest.TestCase], None]:
    r"""Returns a TestCase method to test the provided module class and method.

    Args:
        module_cls: The subclass of nn.Module to export.
        niter: The number of random input data sets to test with.
        run_executor: Whether to run the model on the executor. We may want to
            skip running a model thru executor since some kernels are not
            implemented.
        do_tree_flatten: Whether to flatten input and unflatten output.
        run_graph_module: Whether to run the traced and transformed GraphModule.
            One may want to skip this if some custom ops do not have
            implementation in torch.ops but is implemented in the executor.
        atol: Absolute tolerance used in allclose and torch.allclose
        rtol: Relative tolerance used in allclose and torch.allclose
        ignore_to_out_var_failure: Whether to ignore the failue when a
            functional op does not have an out variant.
        allow_non_contiguous_tensor: If false, will validate that the emitted
            program only contains contiguous tensors.
        method: The name of the module_cls method to trace.
        dynamic_memory_planning_mode: The dynamic memory planning mode to use.

    Returns:
        A TestCase method that tests the provided module class and method.
    """

    def wrapper(self: unittest.TestCase) -> None:
        """A TestCase method that traces/exports/tests an nn.Module and method."""
        module = ExportedModule.export(
            module_class=module_cls,
            # testend2end only supports modules with single methods defined
            methods=(method,),
            ignore_to_out_var_failure=ignore_to_out_var_failure,
            dynamic_memory_planning_mode=dynamic_memory_planning_mode,
            capture_config=capture_config,
        )
        if verify_graph:
            verify_graph(self, module.exported_program.graph_module)
        print(f"inputs for tracing: {module.trace_inputs}")

        # compare the result between the eager module and graph module
        inputs_list = [module.get_random_inputs() for _ in range(niter)]

        if run_graph_module:
            for inputs in inputs_list:
                with torch.no_grad():
                    # only one method is supported so just grab that single method
                    expected = getattr(module.eager_module, module.methods[0])(*inputs)
                with torch.no_grad():
                    result = module.exported_program.module()(*inputs)
                self.assertTrue(allclose(expected, result, rtol, atol))

        program = module.executorch_program.executorch_program
        pretty_print(program)
        print_program(program, show_meminfo=True, mark_dynamic_shape_tensor=True)
        print(f"mem buffer sizes: {program.execution_plan[0].non_const_buffer_sizes}")
        if not allow_non_contiguous_tensor:
            validate_contiguous_tensors(program)
        self.assertTrue(len(program.execution_plan[0].non_const_buffer_sizes) >= 2)
        # We should not enable the following assertion since for some models
        # that simply returning graph input, no mutable memory should be allocated
        # self.assertTrue(all(s > 0 for s in program.program.execution_plan[0].non_const_buffer_sizes[1:]))

        program.version = 0
        buff = module.executorch_program.buffer
        # Check that the magic version number is in the expected place, and
        # follows the expected pattern.
        self.assertRegex(buff[4:8].decode(errors="replace"), r"^ET[0-9][0-9]$")

        if run_executor:
            print("Running on the runtime")
            executorch_module = _load_for_executorch_from_buffer(buff)
            # compare the result between eager module and executor
            for idx, inputs in enumerate(inputs_list):
                with torch.no_grad():
                    expected = getattr(module.eager_module, method)(*inputs)

                if do_tree_flatten:
                    # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
                    flatten_inputs, inputs_spec = pytree.tree_flatten(*inputs)
                    executorch_result = executorch_module.forward([*flatten_inputs])
                    # pyre-fixme[16]: Module `pytree` has no attribute `TreeSpec`.
                    executorch_result_unflatten = pytree.TreeSpec.from_str(
                        program.execution_plan[0].container_meta_type.encoded_out_str
                    ).tree_unflatten(executorch_result)
                    actual = executorch_result_unflatten
                else:
                    actual = executorch_module.forward(inputs)[0]
                is_close = allclose(expected, actual, rtol, atol)
                if not is_close:
                    print(f"Fail for {idx}th inputs: {inputs}")
                    print(f"expected result: {expected}")
                    print(f"actual result: {actual}")
                self.assertTrue(is_close)

    return wrapper


class E2ETest(unittest.TestCase):
    r"""
    When adding a new unittest, call maketest(ModuleName) if possible since
    maketest handles all the boilterplate part. Ideally, we only need define
    a new nn.Module and add one line to call maketest for new end2end test cases.
    """

    # don't run the model thru executor because aten::sin.out is not defined
    # in the executor currently.
    #
    # aten::max.default does not have an out variant. Thus we need set
    # ignore_to_out_var_failure to be True.
    def test_basic(self):
        maketest(ModuleBasic, run_executor=False, ignore_to_out_var_failure=True)(self)

    # Make sure we can handle ops that return mutliple values. E.g. topk
    # At one time we can not properly setup TensorSpec for an Fx node
    # returning multiple tensors
    #
    # don't run the model thru executor because aten::topk.values is not defined
    # in the executor currently
    def test_ops_return_multi(self):
        maketest(ModuleOpsReturnMulti, run_executor=False)(self)

    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) Fix this in both fbcode and oss")
    def test_mem_planning_toy_model(self):
        maketest(
            ToyModelForMemPlanning,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
            ),
        )(self)

    # TODO: add ops implementations and turn on 'run_executor'
    def test_mem_planning_scratch_tensor(self):
        maketest(
            MemPlanningWithScratchTensor,
            run_graph_module=False,
            run_executor=False,
            atol=1e-5,
        )(self)

    def test_executorch_forward(self):
        maketest(ModuleAdd)(self)

    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) Fix this in both fbcode and oss")
    def test_containers(self):
        maketest(
            ModuleContainers,
            do_tree_flatten=True,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
            ),
        )(self)

    # can not run the graph module since the out variance with tensor list out
    # argument returns None rather than tensor list.
    #
    # Can not run in the executor since kernel for tensor splitting is not implemented..
    def test_ops_return_tensorlist(self):
        maketest(ModuleOpsReturnTensorList, run_graph_module=False, run_executor=False)(
            self
        )

    # Failed to produce a graph during tracing w/ dynamo because there are no torch ops
    # test_return_input = maketest(ModuleReturnInput, do_tree_flatten=True)

    # can not run this on the executor because missing the following ops:
    #   aten::select_copy.int_out, aten::eq.Scalar_out
    # TODO(zhxchen17) re-enable these tests.
    # test_control_flow_cond = maketest(ControlFlowCond, run_executor=False)
    # fail to trace with functionalization enabled
    # test_ifelse = maketest(ModuleIfElse)

    # fail to trace with functionalization enabled
    # Fail with error: Missing out variants: {'aten::select', 'aten::_shape_as_tensor', 'aten::tensor_split'}
    # TODO(zhxchen17) re-enable these tests.
    # test_while_0 = maketest(
    #     ControlFlowWhile,
    #     ignore_to_out_var_failure=True,
    #     run_executor=False,
    # )

    # test_while = maketest(ModuleWhile)

    # test_while_if = maketest(ModuleWhileIf)
    # test_if_while = maketest(ModuleIfWhile)
    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) This fails on OSS macos job")
    def test_contiguous_tensor(self):
        maketest(ModuleContiguousTensor, run_executor=False)(self)


class DynamicModelE2ETest(unittest.TestCase):
    """
    End2end tests for dynamic models. For dynamic models we mean models with
    control flow or dynamic shape.
    """

    @skip("Revisit when unbacked symint is ready")
    def test_intermediate_dynamic_shape(self):
        maketest(
            ModuleIntermediateDynamicShape,
            run_graph_module=False,
            allow_non_contiguous_tensor=True,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
            ),
        )(self)

    # TODO(shunting): some non constant tensors for transformer are non-contiguous.
    # Ignore for now. Will debug more.
    # NOTE: can not run on runtime since missing these ops: P535190636
    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) This fails on OSS macos job")
    def test_transformer_encode(self):
        maketest(
            Transformer,
            method="encode",
            allow_non_contiguous_tensor=True,
            run_executor=False,
        )(self)

    # basic test for functorch torch.ops.higher_order.cond
    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) Fix this in both fbcode and oss")
    def test_ft_cond_basic(self):
        maketest(
            FTCondBasic,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
                enable_functionalization=False,  # TODO enable functionalization
            ),
        )(self)

    @skipUnless(RUN_SKIPPED, "Emitter is not ready yet")
    def test_ft_map_basic(self):
        maketest(
            FTMapBasic,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
                enable_functionalization=False,  # TODO enable functionalization
            ),
        )(self)

    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) Fix this in both fbcode and oss")
    def test_ft_cond_dynshape(self):
        maketest(
            FTCondDynShape,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
                enable_functionalization=False,  # TODO enable functionalization
            ),
        )(self)

    @skipUnless(RUN_SKIPPED, "Emitter is not ready yet")
    def test_ft_map_dynshape(self):
        maketest(
            FTMapDynShape,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
                enable_functionalization=False,  # TODO enable functionalization
            ),
        )(self)

    @skipUnless(RUN_SKIPPED, "TODO(larryliu0820) Fix this in both fbcode and oss")
    def test_batch_norm(self):
        maketest(
            BatchNormModel,
            capture_config=exir.CaptureConfig(
                enable_dynamic_shape=True,
            ),
            verify_graph=BatchNormModel.verify_graph,
            # TODO: lean mode does not have native_batch_norm.out implemented
            # run this on aten mode.
            run_executor=is_aten_mode,
        )(self)
