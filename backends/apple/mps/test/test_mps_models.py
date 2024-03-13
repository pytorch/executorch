#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import random

import executorch.exir.control_flow as control_flow
import torch
from functorch.experimental.control_flow import cond
from torch import nn


class LayerNormModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm([5, 10, 10])

    def forward(self, arg):
        return self.norm(arg)

    @staticmethod
    def get_example_inputs():
        return (torch.randn(20, 5, 10, 10),)


class Conv2DModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1)

    def forward(self, arg):
        return self.conv(arg)

    @staticmethod
    def get_example_inputs():
        return (torch.randn(1, 1, 3, 3),)


class ModuleBasic(torch.nn.Module):
    def __init__(self):
        super(ModuleBasic, self).__init__()

    def forward(self, x):
        return torch.sin(x).max()

    def get_random_inputs(self):
        return (torch.randn(100),)


class ModuleOpsReturnMulti(torch.nn.Module):
    def __init__(self):
        super(ModuleOpsReturnMulti, self).__init__()

    def forward(self, a, b):
        x, y = torch.topk(a, 3)
        return x * 2 + b

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(3))


class ModuleAdd(torch.nn.Module):
    def __init__(self):
        super(ModuleAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2))


class ModuleFloatAddWithAlpha(torch.nn.Module):
    def __init__(self):
        super(ModuleFloatAddWithAlpha, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: float):
        return torch.add(x, y, alpha=c)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2), random.random())


class ModuleIntAddWithAlpha(torch.nn.Module):
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


class ModuleContainers(torch.nn.Module):
    def __init__(self):
        super(ModuleContainers, self).__init__()

    def forward(self, d):
        a = d["a"]
        b = d["b"]
        return {"inputs": (a, b), "c": torch.add(a, b)}

    def get_random_inputs(self):
        return ({"a": torch.randn(2, 2), "b": torch.randn(2, 2)},)


class ToyModelForMemPlanning(torch.nn.Module):
    def __init__(self):
        super(ToyModelForMemPlanning, self).__init__()

    def forward(self, a, b):
        o = a
        for _ in range(3):
            o = o * a
            o = o + b
        return o

    def get_random_inputs(self):
        return (
            torch.randn(10),
            torch.randn(10),
        )


class MemPlanningWithScratchTensor(torch.nn.Module):
    def __init__(self):
        super(MemPlanningWithScratchTensor, self).__init__()
        self.linear1 = torch.nn.Linear(4, 2)
        self.linear2 = torch.nn.Linear(4, 2)

    def forward(self, a, b):
        o1 = self.linear1(a)
        o2 = self.linear2(b)
        return o1 + o2

    def get_random_inputs(self):
        return (
            torch.randn(10, 4),
            torch.randn(10, 4),
        )


class ModuleOpsReturnTensorList(torch.nn.Module):
    def __init__(self):
        super(ModuleOpsReturnTensorList, self).__init__()

    def forward(self, x):
        split = torch.ops.aten.tensor_split.sections(x, 3)
        return split[0]

    def get_random_inputs(self):
        return (torch.randn(100),)


class ModuleReturnInput(torch.nn.Module):
    def __init__(self):
        super(ModuleReturnInput, self).__init__()

    def forward(self, x):
        return (x, x, {"x": x, "y": x}, [x, x, x])

    def get_random_inputs(self):
        return (torch.randn(1),)


class ModuleIfElse(torch.nn.Module):
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


class ModuleIfElseWithBoolInput(torch.nn.Module):
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


class ModuleWhileIf(torch.nn.Module):
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


class ModuleIfWhile(torch.nn.Module):
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


class ModuleContiguousTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 32)

    def forward(self, arg):
        return self.linear(arg)

    def get_random_inputs(self):
        return (torch.randn(3, 8),)


class ModuleInputDynamicShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for _ in range(4):
            x = x + x
            x = x * x
        return x

    def get_upper_bound_inputs(self):
        return (torch.randn(10),)

    def get_random_inputs(self):
        n = random.randint(1, 10)
        return (torch.randn(n),)


class ModuleIntermediateDynamicShape(torch.nn.Module):
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


MPS_MODEL_NAME_TO_MODEL = {
    "conv2D": lambda: (Conv2DModule(), Conv2DModule.get_example_inputs()),
    "norm": lambda: (LayerNormModule(), LayerNormModule.get_example_inputs()),
    "module_basic": lambda: (ModuleBasic(), ModuleBasic().get_random_inputs()),
    "module_ops_return_multi": lambda: (
        ModuleOpsReturnMulti(),
        ModuleOpsReturnMulti().get_random_inputs(),
    ),
    "module_add": lambda: (ModuleAdd(), ModuleAdd().get_random_inputs()),
    "module_float_add_with_alpha": lambda: (
        ModuleFloatAddWithAlpha(),
        ModuleFloatAddWithAlpha().get_random_inputs(),
    ),
    "module_int_add_with_alpha": lambda: (
        ModuleIntAddWithAlpha(),
        ModuleIntAddWithAlpha().get_random_inputs(),
    ),
    "module_containers": lambda: (
        ModuleContainers(),
        ModuleContainers().get_random_inputs(),
    ),
    "toy_model_for_mem_planning": lambda: (
        ToyModelForMemPlanning(),
        ToyModelForMemPlanning().get_random_inputs(),
    ),
    "mem_planning_with_scratch_tensor": lambda: (
        MemPlanningWithScratchTensor(),
        MemPlanningWithScratchTensor().get_random_inputs(),
    ),
    "module_ops_return_tensor_list": lambda: (
        ModuleOpsReturnTensorList(),
        ModuleOpsReturnTensorList().get_random_inputs(),
    ),
    "module_return_input": lambda: (
        ModuleReturnInput(),
        ModuleReturnInput().get_random_inputs(),
    ),
    "module_if_else": lambda: (ModuleIfElse(), ModuleIfElse().get_random_inputs()),
    "module_if_else_with_bool_input": lambda: (
        ModuleIfElseWithBoolInput(),
        ModuleIfElseWithBoolInput().get_random_inputs(),
    ),
    "module_while_if": lambda: (ModuleWhileIf(), ModuleWhileIf().get_random_inputs()),
    "module_if_while": lambda: (ModuleIfWhile(), ModuleIfWhile().get_random_inputs()),
    "module_contiguous_tensor": lambda: (
        ModuleContiguousTensor(),
        ModuleContiguousTensor().get_random_inputs(),
    ),
    "module_input_dynamic_shape": lambda: (
        ModuleInputDynamicShape(),
        ModuleInputDynamicShape().get_random_inputs(),
    ),
    "module_intermediate_dynamic_shape": lambda: (
        ModuleIntermediateDynamicShape(),
        ModuleIntermediateDynamicShape().get_random_inputs(),
    ),
}
