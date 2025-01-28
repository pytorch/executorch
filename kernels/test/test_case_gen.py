# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys

from abc import ABC, abstractmethod

from enum import Enum

import torch

"""
A helper library to generate test cases for ET kernels.

It simplifies the steps to generate a new c++ test case. User just need
to specify the inputs and we use pytorch kernel to calculate the result.
"""


# Seed the RNG in all the common libraries for test reproducibility
torch.manual_seed(0)


def make_out_static_shape(tensor: torch.Tensor):
    sizes = list(tensor.size())
    sizes = [str(s) for s in sizes]
    sizes_str = "{" + ", ".join(sizes) + "}"
    return sizes_str


def make_out_dynamic_shape_bound_shape_same(tensor: torch.Tensor):
    sizes = list(tensor.size())
    sizes = [str(s) for s in sizes]
    sizes_str = "{" + ", ".join(sizes) + "}"
    return sizes_str + ", torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"


def make_out_dynamic_shape_bound_shape_larger(tensor: torch.Tensor):
    sizes = list(tensor.size())
    extra_sizes = [x * 2 for x in sizes]
    extra_sizes = [str(s) for s in extra_sizes]
    sizes_str = "{" + ", ".join(extra_sizes) + "}"
    return sizes_str + ", torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"


def make_out_dynamic_shape_unbound_shape(tensor: torch.Tensor):
    sizes = list(tensor.size())
    smaller_sizes = [1 for x in sizes]
    smaller_sizes = [str(s) for s in smaller_sizes]
    sizes_str = "{" + ", ".join(smaller_sizes) + "}"
    return sizes_str + ", torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"


class ShapeDynamism(Enum):
    # Static shape; shape is determined from pytorch output
    STATIC = 1
    # Dynamic bound with same size; shape is determined from pytorch output using the same size as static
    DYNAMIC_BOUND_SAME_SHAPE = 2
    # Dynamic bound with a larger size to test functionality; shape is determined from pytorch output
    DYNAMIC_BOUND_LARGER_SHAPE = 3
    # Dynamic unbound with a smaller size to test functionality
    DYNAMIC_UNBOUND = 4


out_dynamic_shape_fn_map = {
    ShapeDynamism.STATIC: make_out_static_shape,
    ShapeDynamism.DYNAMIC_BOUND_SAME_SHAPE: make_out_dynamic_shape_bound_shape_same,
    ShapeDynamism.DYNAMIC_BOUND_LARGER_SHAPE: make_out_dynamic_shape_bound_shape_larger,
    ShapeDynamism.DYNAMIC_UNBOUND: make_out_dynamic_shape_unbound_shape,
}


def make_test_cases_dynamic_shape(*args):
    """
    A helper to make a list of tuples (test cases). Each tuple contains
    the name,
    inputs and output (expanded from *args),
    dynamic shape type
    """
    return [
        (
            "DynamicShapeUpperBoundSameAsExpected",
            *args,
            ShapeDynamism.DYNAMIC_BOUND_SAME_SHAPE,
        ),
        (
            "DynamicShapeUpperBoundLargerThanExpected",
            *args,
            ShapeDynamism.DYNAMIC_BOUND_LARGER_SHAPE,
        ),
        (
            "DynamicShapeUnbound",
            *args,
            ShapeDynamism.DYNAMIC_UNBOUND,
        ),
    ]


def make_test_cases_broadcast_two_input_tensor(x, y, cpp_args, torch_args, torch_fn):
    """
    A helper to make a list of tuples (test cases). Each tuple contains
    the name,
    inputs and output (expanded from *args),
    dynamic shape type (use static here)

    Used when we have two input tensors (like add, mul, div).
    Generate test cases where
    we drop a dimension from the first/second tensor
    we set a dimension to one from the first/second tensor
    """
    x_remove_dim = x[0]
    x_first_dim_1 = x_remove_dim.squeeze(0)
    y_remove_dim = y[0]
    y_first_dim_1 = y_remove_dim.squeeze(0)

    return [
        (
            "BroadcastDimSizeIsOneAB",
            x_first_dim_1,
            y,
            *cpp_args,
            torch_fn(x_first_dim_1, y, *torch_args),
            ShapeDynamism.STATIC,
        ),
        (
            "BroadcastDimSizeMissingAB",
            x_remove_dim,
            y,
            *cpp_args,
            torch_fn(x_remove_dim, y, *torch_args),
            ShapeDynamism.STATIC,
        ),
        (
            "BroadcastDimSizeIsOneBA",
            x,
            y_first_dim_1,
            *cpp_args,
            torch_fn(x, y_first_dim_1, *torch_args),
            ShapeDynamism.STATIC,
        ),
        (
            "BroadcastDimSizeMissingBA",
            x,
            y_remove_dim,
            *cpp_args,
            torch_fn(x, y_remove_dim, *torch_args),
            ShapeDynamism.STATIC,
        ),
    ]


class ArgType(ABC):
    """
    Represents an argument for generated C++ code and for pytorch call
    """

    @abstractmethod
    def to_pytorch(self):
        return None

    @abstractmethod
    def to_cpp(self) -> str:
        return ""


class Scalar(ArgType):
    def __init__(self, val):
        self.val = val

    def to_pytorch(self):
        return self.val

    def to_cpp(self):
        return f"Scalar({self.val})"


class OptScalar(ArgType):
    def __init__(self, val):
        self.val = val

    def to_pytorch(self):
        return self.val

    def to_cpp(self):
        return f"OptScalar({self.val})"


class ArrayRef(ArgType):
    def __init__(self, dtype, data: list):
        self.dtype = dtype
        self.data = data

    def to_pytorch(self):
        return self.data

    def to_cpp(self):
        array_str = "{" + ",".join(str(data) for data in self.data) + "}"
        return f"ArrayRef<{self.dtype}>({array_str})"


class EnumArg(ArgType):
    def __init__(self, text):
        self.text = text

    def to_pytorch(self):
        # Most likely it cannot be directly used
        return ""

    def to_cpp(self):
        return self.text


class StringArg(ArgType):
    def __init__(self, text):
        self.text = text

    def to_pytorch(self):
        return self.text

    def to_cpp(self):
        return f'"{self.text}"'


def tensor_to_cpp_code(tensor: torch.Tensor) -> str:
    sizes = list(tensor.size())
    sizes = [str(s) for s in sizes]
    sizes_str = "{" + ", ".join(sizes) + "}"
    data = torch.flatten(tensor).tolist()
    data = [str(d) for d in data]
    data_str = "{" + ", ".join(data) + "}"
    if tensor.dtype == torch.bool:
        return f"""tf_bool.make({sizes_str}, {data_str})""".replace(
            "True", "true"
        ).replace("False", "false")
    return f"""tf.make({sizes_str}, {data_str})"""


def argument_to_cpp_code(arg):
    if isinstance(arg, str):
        return arg
    elif isinstance(arg, bool):
        return "true" if arg else "false"
    elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
        # Note: We explicitly exclude bool because bool is a subset of int
        return str(arg)
    elif isinstance(arg, bool):
        return "true" if arg else "false"
    elif isinstance(arg, torch.Tensor):
        return tensor_to_cpp_code(arg)
    elif isinstance(arg, ArgType):
        return arg.to_cpp()
    return "?"


def argument_to_pytorch(arg):
    if isinstance(arg, (str, int, float, torch.Tensor)):
        return arg
    elif isinstance(arg, ArgType):
        return arg.to_pytorch()
    return "?"


class ArgForPyTorch:
    """Sometimes an arg for cpp cannot directly be used in torch because it is not used, or used only in torch, or it is a kwarg"""

    def __init__(self, cpp_arg, torch_kwarg_key, torch_kwarg_val):
        self.cpp_arg = cpp_arg
        self.kwarg_pair = torch_kwarg_key, torch_kwarg_val

    def used_in_cpp(self):
        return self.cpp_arg is not None

    def used_in_torch(self):
        return self.kwarg_pair != (None, None)


def make_simple_generated_case(*args, torch_fn):
    cpp_args = tuple(
        arg.cpp_arg if isinstance(arg, ArgForPyTorch) else arg
        for arg in args
        if not isinstance(arg, ArgForPyTorch) or arg.used_in_cpp()
    )
    torch_args = tuple(
        argument_to_pytorch(arg) for arg in args if not isinstance(arg, ArgForPyTorch)
    )
    kwargs_for_torch_fn = dict(
        arg.kwarg_pair
        for arg in args
        if isinstance(arg, ArgForPyTorch) and arg.used_in_torch()
    )
    return [
        (
            "SimpleGeneratedCase",
            *cpp_args,
            torch_fn(*torch_args, **kwargs_for_torch_fn),
            ShapeDynamism.STATIC,
        )
    ]


def gen_test_cases(suite_name: str, op_name: str, test_cases, test_f=False):
    """
    Used when some inputs are not Tensor or scalar. Treat them as code text and generate.
    Each test case should be a tuple of
    (test_case_name, inputs, expected_result, shape_dynamism)
    out_size is the pre-allocatd size for out tensor
    Set test_f to True if we want TEST_F (gtest fixture)

    For example, in https://www.internalfb.com/code/fbsource/[7280e42e309e85294a77fbb51ccc6de1948f2497]/fbcode/executorch/kernels/test/op_add_test.cpp?lines=19-23, we have an additional alpha parameter
    """

    variable_names = "xyzabcdefghijk"
    newline = "\n"

    generated_cases = []

    for test_name, *inputs, expected_result, shape_dynamism in test_cases:
        out_dynamic_shape_fn = out_dynamic_shape_fn_map[shape_dynamism]
        input_code = [argument_to_cpp_code(i) for i in inputs]
        input_lines = [
            f"auto {variable_names[i]} = {input_code[i]};" for i in range(len(inputs))
        ]

        need_tf_bool = any(
            isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in inputs
        )

        ret_value = f"""{op_name}({", ".join(variable_names[:len(inputs)])}, out)"""

        generated_cases.append(
            f"""
{"TEST_F" if test_f else "TEST"}({suite_name}, {test_name}) {{
  TensorFactory<ScalarType::Float> tf;
  {"TensorFactory<ScalarType::Bool> tf_bool;" if need_tf_bool else ""}

  {newline.join(input_lines)}
  Tensor expected_result = {tensor_to_cpp_code(expected_result)};

  Tensor out = tf.zeros({out_dynamic_shape_fn(expected_result)});
  Tensor ret = {ret_value};
  EXPECT_TENSOR_CLOSE(out, expected_result);
}}
"""
        )
    return generated_cases


def gen_test_case_op_arange():
    return gen_test_cases(
        "OpArangeOutTest",
        "arange_out",
        make_test_cases_dynamic_shape(Scalar(5), torch.arange(5)),
        test_f=True,
    )


def gen_test_case_op_as_strided_copy():
    # TODO: Implement
    return


def gen_test_case_op_bitwise_not():
    # TODO: Implement
    return


def gen_test_case_op_cat():
    # TODO: Implement
    return


def gen_test_case_op_clamp():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpClampOutTest",
        "clamp_out",
        make_simple_generated_case(
            torch.ones(10, 10), OptScalar(-0.5), OptScalar(0.5), torch_fn=torch.clamp
        )
        + make_test_cases_dynamic_shape(
            x, OptScalar(-0.5), OptScalar(0.5), torch.clamp(x, -0.5, 0.5)
        ),
    )


def gen_test_case_op_clone():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpCloneTest",
        "clone_out",
        make_simple_generated_case(
            torch.ones(10, 10),
            ArgForPyTorch(
                EnumArg("exec_aten::MemoryFormat::Contiguous"),
                "memory_format",
                torch.contiguous_format,
            ),
            torch_fn=torch.clone,
        )
        + make_test_cases_dynamic_shape(
            x,
            EnumArg("exec_aten::MemoryFormat::Contiguous"),
            torch.clone(x, memory_format=torch.contiguous_format),
        ),
    )


def gen_test_case_op_cumsum():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpCumSumOutTest",
        "cumsum_out",
        make_simple_generated_case(
            torch.ones(10, 10),
            ArgForPyTorch(1, "dim", 1),
            ArgForPyTorch(EnumArg("ScalarType::Float"), "dtype", torch.float),
            torch_fn=torch.cumsum,
        )
        + make_test_cases_dynamic_shape(
            x,
            1,
            EnumArg("ScalarType::Float"),
            torch.cumsum(x, dim=1, dtype=torch.float),
        ),
    )


def gen_test_case_op_detach_copy():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpDetachCopyOutKernelTest",
        "_detach_copy_out",
        make_simple_generated_case(torch.ones(10, 10), torch_fn=torch.detach)
        + make_test_cases_dynamic_shape(x, torch.Tensor.detach(x)),
    )


def gen_test_case_op_exp():
    # TODO: Implement
    return


def gen_test_case_op_expand():
    # TODO: Implement
    return


def gen_test_case_op_full_like():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpFullLikeTest",
        "full_like_out",
        make_simple_generated_case(
            torch.ones(10, 10),
            Scalar(3.0),
            ArgForPyTorch(
                EnumArg("MemoryFormat::Contiguous"),
                "memory_format",
                torch.contiguous_format,
            ),
            torch_fn=torch.full_like,
        )
        + make_test_cases_dynamic_shape(
            x,
            Scalar(3.0),
            EnumArg("MemoryFormat::Contiguous"),
            torch.full_like(x, 3.0, memory_format=torch.contiguous_format),
        ),
    )


def gen_test_case_op_gelu():
    x = torch.rand(3, 2)

    m = torch.nn.GELU(approximate="tanh")

    return gen_test_cases(
        "OpGeluKernelTest",
        "gelu_out",
        make_simple_generated_case(
            torch.ones(10, 10), ArgForPyTorch(StringArg("tanh"), None, None), torch_fn=m
        )
        + make_test_cases_dynamic_shape(x, StringArg("tanh"), m(x)),
    )


def gen_test_case_op_glu():
    x = torch.rand(4, 2)

    m = torch.nn.GLU(0)

    return gen_test_cases(
        "OpGluOutKernelTest",
        "glu_out",
        make_test_cases_dynamic_shape(x, 0, m(x)),
    )


def gen_test_case_op_log():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpLogOutKernelTest",
        "_log_out",
        make_simple_generated_case(torch.ones(10, 10), torch_fn=torch.log)
        + make_test_cases_dynamic_shape(x, torch.log(x)),
    )


def gen_test_case_op_log_softmax():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpLogSoftmaxOutTest",
        "log_softmax_out",
        make_simple_generated_case(
            torch.ones(10, 10),
            1,
            ArgForPyTorch(False, None, None),
            ArgForPyTorch(None, "dtype", torch.float),
            torch_fn=torch.log_softmax,
        )
        + make_test_cases_dynamic_shape(
            x, 1, False, torch.log_softmax(x, 1, torch.float)
        ),
    )


def gen_test_case_op_logit():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpLogitOutKernelTest",
        "logit_out",
        make_simple_generated_case(torch.ones(10, 10), 0.1, torch_fn=torch.logit)
        + make_test_cases_dynamic_shape(x, 0.1, torch.logit(x, 0.1)),
    )


def gen_test_case_op_mean():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpMeanOutTest",
        "mean_dim_out",
        make_simple_generated_case(
            torch.ones(10, 10),
            ArgForPyTorch(ArrayRef("int64_t", [1]), "dim", 1),
            ArgForPyTorch(False, "keepdim", False),
            ArgForPyTorch(EnumArg("ScalarType::Float"), "dtype", torch.float),
            torch_fn=torch.mean,
        )
        + make_test_cases_dynamic_shape(
            x,
            ArrayRef("int64_t", [1]),
            False,
            EnumArg("ScalarType::Float"),
            torch.Tensor.mean(x, dim=1, keepdim=False, dtype=torch.float),
        ),
    )


def gen_test_case_op_nonzero():
    # TODO: Implement
    return


def gen_test_case_op_permute():
    # TODO: Implement
    return


def gen_test_case_op_relu():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpReluOutKernelTest",
        "_relu_out",
        make_simple_generated_case(torch.ones(10, 10), torch_fn=torch.relu)
        + make_test_cases_dynamic_shape(x, torch.relu(x)),
    )


def gen_test_case_op_repeat():
    # TODO: Implement
    return


def gen_test_case_op_round():
    # TODO: Implement
    return


def gen_test_case_op_sigmoid():
    # TODO: Implement
    return


def gen_test_case_op_slice():
    # TODO: Implement
    return


def gen_test_case_op_softmax():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpSoftmaxOutTest",
        "softmax_out",
        make_simple_generated_case(
            torch.ones(10, 10),
            1,
            ArgForPyTorch(False, "dtype", torch.float),
            torch_fn=torch.softmax,
        )
        + make_test_cases_dynamic_shape(x, 1, False, torch.softmax(x, 1, torch.float)),
    )


def gen_test_case_op_squeeze():
    # TODO: Implement
    return


def gen_test_case_op_sum():
    # TODO: Implement
    return


def gen_test_case_op_t():
    # TODO: Implement
    return


def gen_test_case_op_tanh():
    x = torch.rand(3, 2)

    return gen_test_cases(
        "OpTanhOutKernelTest",
        "_tanh_out",
        make_simple_generated_case(torch.ones(10, 10), torch_fn=torch.tanh)
        + make_test_cases_dynamic_shape(x, torch.tanh(x)),
    )


def gen_test_case_op_to():
    # TODO: Implement
    return


def gen_test_case_op_transpose():
    # TODO: Implement
    return


def gen_test_case_op_unsqueeze():
    # TODO: Implement
    return


def gen_test_case_op_view():
    # TODO: Implement
    return


def gen_test_case_op_zeros():
    # TODO: Implement
    return


def gen_test_case_op_add():
    x = torch.rand(3, 2)
    y = torch.rand(3, 2)

    return gen_test_cases(
        "OpAddOutKernelTest",
        "add_out",
        make_simple_generated_case(
            torch.ones(10, 10), torch.ones(10, 10), torch_fn=torch.add
        )
        + make_test_cases_broadcast_two_input_tensor(x, y, (1,), (), torch_fn=torch.add)
        + make_test_cases_dynamic_shape(x, y, 1, torch.add(x, y)),
    )


def gen_test_case_op_bmm():
    x = torch.rand(3, 3, 6)
    y = torch.rand(3, 6, 2)

    return gen_test_cases(
        "OpBmmOutKernelTest",
        "_bmm_out",
        make_test_cases_dynamic_shape(x, y, torch.bmm(x, y)),
    )


def gen_test_case_op_copy():
    # TODO: Implement
    return


def gen_test_case_op_div():
    x = torch.rand(3, 2)
    y = torch.rand(3, 2)

    return gen_test_cases(
        "OpDivOutKernelTest",
        "_div_out",
        make_test_cases_broadcast_two_input_tensor(x, y, (), (), torch_fn=torch.div)
        + make_test_cases_dynamic_shape(x, y, torch.div(x, y)),
    )


def gen_test_case_op_embedding():
    # TODO: Implement
    return


def gen_test_case_op_eq():
    # TODO: Implement
    return


def gen_test_case_op_floor_divide():
    x = torch.rand(3, 2)
    y = torch.rand(3, 2)

    return gen_test_cases(
        "OpFloorDivideKernelTest",
        "_floor_divide_out",
        make_test_cases_broadcast_two_input_tensor(
            x, y, (), (), torch_fn=torch.floor_divide
        )
        + make_test_cases_dynamic_shape(x, y, torch.floor_divide(x, y)),
    )


def gen_test_case_op_le():
    # TODO: Implement
    return


def gen_test_case_op_minimum():
    # TODO: Implement
    return


def gen_test_case_op_mm():
    x = torch.rand(3, 2)
    y = torch.rand(2, 4)

    return gen_test_cases(
        "OpMmOutKernelTest",
        "_mm_out",
        make_test_cases_dynamic_shape(x, y, torch.mm(x, y)),
    )


def gen_test_case_op_mul():
    x = torch.rand(3, 2)
    y = torch.rand(3, 2)

    return gen_test_cases(
        "OpMulOutKernelTest",
        "_mul_out",
        make_test_cases_broadcast_two_input_tensor(x, y, (), (), torch_fn=torch.mul)
        + make_test_cases_dynamic_shape(x, y, torch.mul(x, y)),
    )


def gen_test_case_op_ne():
    # TODO: Implement
    return


def gen_test_case_op_select():
    # TODO: Implement
    return


def gen_test_case_op_select_scatter():
    # TODO: Implement
    return


def gen_test_case_op_sub():
    x = torch.rand(3, 2)
    y = torch.rand(3, 2)

    return gen_test_cases(
        "OpSubOutKernelTest",
        "sub_out",
        make_test_cases_broadcast_two_input_tensor(x, y, (1,), (), torch_fn=torch.sub)
        + make_test_cases_dynamic_shape(x, y, 1, torch.sub(x, y)),
    )


def gen_test_case_op_addmm():
    x = torch.rand(3, 6)
    y = torch.rand(6, 2)

    b = torch.rand(3, 2)
    b_dim_is_1 = torch.rand(1, 2)
    b_miss_dim = torch.squeeze(b_dim_is_1)

    return gen_test_cases(
        "OpAddmmOutKernelTest",
        "addmm_out",
        [
            (
                "BroadcastDimSizeIsOne",
                b_dim_is_1,
                x,
                y,
                Scalar(1),
                Scalar(1),
                torch.addmm(b_dim_is_1, x, y),
                ShapeDynamism.STATIC,
            ),
            (
                "BroadcastDimSizeMissing",
                b_miss_dim,
                x,
                y,
                Scalar(1),
                Scalar(1),
                torch.addmm(b_dim_is_1, x, y),
                ShapeDynamism.STATIC,
            ),
        ]
        + make_test_cases_dynamic_shape(
            b, x, y, Scalar(1), Scalar(1), torch.addmm(b, x, y)
        ),
    )


def gen_test_case_op_convolution():
    # TODO: Implement
    return


def gen_test_case_op_where():
    # TODO: Implement
    return


def gen_test_case_op_masked_fill():
    a = torch.rand(3, 2)

    b = torch.rand(3, 2) > 0.5

    return gen_test_cases(
        "OpMaskedFillTest",
        "masked_fill_scalar_out",
        make_test_cases_broadcast_two_input_tensor(
            a, b, (Scalar(3.0),), (3.0,), torch_fn=torch.masked_fill
        )
        + (
            make_test_cases_dynamic_shape(
                a, b, Scalar(3.0), torch.masked_fill(a, b, 3.0)
            )
        ),
    )


def get_test_case_name(generated_test_case: str):
    m = re.search("TEST(_F)?\\(.*\\)", generated_test_case)
    if m is not None:
        test_case = m.group(0)
        return "".join(test_case.split())


def gen_test_cases_for_file(path_to_tests: str, op_name: str):
    if ("gen_test_case_" + op_name) not in globals():
        print(f"generator function is not defined for {op_name}")
        return
    gen_func = globals()[("gen_test_case_" + op_name)]
    generated_test_cases = gen_func()
    if generated_test_cases is None:
        print(f"generator function is not implemented for {op_name}")
        return
    file_name = op_name + "_test.cpp"
    with open(os.path.join(path_to_tests, file_name), "r+") as f:
        previous = f.read()
        # Remove all white spaces and new lines
        previous = "".join(previous.split())
        for generated_test_case in generated_test_cases:
            if get_test_case_name(generated_test_case) not in previous:
                f.write(generated_test_case)
                print(f"test case {get_test_case_name(generated_test_case)} added")


def main():
    print("Generating test cases...")
    if len(sys.argv) < 2:
        print("Usage: test_case_gen.py <path-to-kernels/test>")
        return
    test_dir = sys.argv[1]
    ops = [
        f[:-9]
        for f in os.listdir(test_dir)
        if f.startswith("op_") and f.endswith("_test.cpp")
    ]
    for op in ops:
        gen_test_cases_for_file(test_dir, op)


if __name__ == "__main__":
    main()
