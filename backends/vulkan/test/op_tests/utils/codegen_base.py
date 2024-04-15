# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, List

from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.model import Argument, NativeFunction

########################
## ATen code patterns ##
########################

AT_INT_ARRAY_REF = "at::IntArrayRef"
AT_SCALAR = "at::Scalar"
AT_TENSOR = "at::Tensor"
BOOL = "bool"
DOUBLE = "double"
INT = "int64_t"
OPT_AT_TENSOR = "::std::optional<at::Tensor>"
OPT_BOOL = "::std::optional<bool>"
OPT_DEVICE = "::std::optional<at::Device>"
OPT_LAYOUT = "::std::optional<at::Layout>"
OPT_SCALARTYPE = "::std::optional<at::ScalarType>"
TWO_TENSOR_TUPLE = "::std::tuple<at::Tensor,at::Tensor>"
THREE_TENSOR_TUPLE = "::std::tuple<at::Tensor,at::Tensor,at::Tensor>"

###########################
## Test Suite definition ##
###########################


@dataclass
class TestSuite:
    input_cases: List[Any]
    prepacked_args = []
    requires_prepack = False

    def supports_prepack(self):
        return len(self.prepacked_args) > 0


##########################
## Test Suite Generation ##
##########################

test_fixture_template = """
class GeneratedOpsTest_{op_name} : public ::testing::Test {{
}};
"""

test_suite_template = """
TEST_P(GeneratedOpsTest_{op_name}, {case_name}) {{
    {create_ref_data}
    {create_and_check_out}
}}
"""


def init_list_str(pylist: Any) -> str:
    if pylist == "[]":
        return "{" + "}"

    if not isinstance(pylist, (list, tuple)):
        pylist = [pylist]

    init_list_str = "{"
    for s in pylist:
        init_list_str += f"{s}, "
    init_list_str = init_list_str[:-2] + "}"
    return init_list_str


def get_or_return_default(arg: Argument, inputs: List[Any], i: int):
    if i < len(inputs):
        return inputs[i]
    else:
        assert arg.default is not None
        return arg.default


class TestSuiteGen:
    backend_key = None

    def __init__(self, f: NativeFunction, test_suite: TestSuite):
        self.f = f
        self.suite_def = test_suite
        self.op_name = f.func.name.unambiguous_name()

        self.f_sig = CppSignatureGroup.from_native_function(
            self.f, method=False, fallback_binding=self.f.manual_cpp_binding
        ).most_faithful_signature()

    def gen_case_name(self, inputs: List[Any], prepack: bool = False) -> str:
        name_str = self.op_name
        if prepack:
            name_str += "_prepack"
        for arg_sizes_or_val in inputs:
            name_str += "_"
            if isinstance(arg_sizes_or_val, tuple):
                for size in arg_sizes_or_val:
                    name_str += str(size) + "x"
                name_str = name_str[:-1]
            elif isinstance(arg_sizes_or_val, list):
                for size in arg_sizes_or_val:
                    name_str += str(size) + "c"
                name_str = name_str[:-1]
            else:
                name_str += str(arg_sizes_or_val).replace(".", "p")
        return name_str

    def create_input_data(self, arg: Argument, data: Any) -> str:
        ctype = cpp.argumenttype_type(arg.type, mutable=arg.is_write, binds=arg.name)
        cpp_type = ctype.cpp_type(strip_ref=True)

        if cpp_type == AT_INT_ARRAY_REF:
            ret_str = f"std::vector<int64_t> {arg.name} = "
        else:
            ret_str = f"{cpp_type} {arg.name} = "

        if cpp_type == AT_TENSOR:
            ret_str += f"make_rand_tensor({init_list_str(data)}, test_dtype);"
        elif cpp_type == OPT_AT_TENSOR:
            if str(data) == "None":
                ret_str += "std::nullopt;"
            else:
                ret_str += f"make_rand_tensor({init_list_str(data)}, test_dtype);"
        elif cpp_type == AT_SCALAR:
            ret_str += f"{data};"
        elif cpp_type == AT_INT_ARRAY_REF:
            ret_str += f"{init_list_str(data)};"
        elif cpp_type == BOOL:
            ret_str += f"{str(data).lower()};"
        elif cpp_type == INT:
            ret_str += f"{str(data).lower()};"
        elif cpp_type == DOUBLE:
            ret_str += f"{str(data).lower()};"
        elif (
            cpp_type == OPT_SCALARTYPE
            or cpp_type == OPT_LAYOUT
            or cpp_type == OPT_DEVICE
            or cpp_type == OPT_BOOL
        ):
            ret_str += "std::nullopt;"
        else:
            raise RuntimeError(f"Unsupported cpp type {cpp_type}")
        return ret_str + "\n"

    def gen_create_ref_data(self, inputs: List[Any]) -> str:
        ref_code = ""

        for i, binding in enumerate(self.f_sig.arguments()):
            arg = binding.argument
            arg_data = get_or_return_default(arg, inputs, i)
            ref_code += self.create_input_data(arg, arg_data)

        return ref_code

    def gen_create_and_check_out(self, prepack=False) -> str:
        test_str = f"check_{self.op_name}("
        if prepack:
            test_str = f"prepacked_check_{self.op_name}("
        for binding in self.f_sig.arguments():
            arg = binding.argument
            test_str += f"{arg.name}, "
        test_str = test_str[:-2] + ");"
        return test_str

    def gen_parameterization(self) -> str:
        return ""

    def generate_fixture_cpp(self) -> str:
        return test_fixture_template.format(op_name=self.f.func.name)

    def generate_case_cpp(self, inputs, prepack=False) -> str:
        return test_suite_template.format(
            op_name=f"{self.op_name}",
            case_name=self.gen_case_name(inputs, prepack),
            create_ref_data=self.gen_create_ref_data(inputs),
            create_and_check_out=self.gen_create_and_check_out(prepack),
        )

    def generate_suite_cpp(self) -> str:
        suite_cpp = self.generate_fixture_cpp() + "\n"
        for inputs in self.suite_def.input_cases:
            if not self.suite_def.requires_prepack:
                suite_cpp += self.generate_case_cpp(inputs)
            if self.suite_def.supports_prepack():
                suite_cpp += self.generate_case_cpp(inputs, prepack=True)

        suite_cpp += self.gen_parameterization()
        return suite_cpp


##########################
## Test File Generation ##
##########################

cpp_test_template = """
#include <gtest/gtest.h>

#include <ATen/ATen.h>

{preamble}

at::Tensor make_rand_tensor(
        std::vector<int64_t> sizes,
        at::ScalarType dtype = at::kFloat,
        float high = 1.0,
        float low = 0.0) {{
    if (high == 1.0 && low == 0.0)
        return at::rand(sizes, at::device(at::kCPU).dtype(dtype));

    return at::rand(sizes, at::device(at::kCPU).dtype(dtype)) * (high - low) + low;
}}

{test_suites_cpp}
"""


class CppTestFileGen:
    def __init__(self, out_path):
        self.out_path = out_path
        self.suites_gens = []

    def generate_cpp(self) -> str:
        return cpp_test_template.format(
            preamble=self.generate_preamble(),
            test_suites_cpp=self.generate_test_suites_cpp(),
        )

    def generate_preamble(self) -> str:
        return ""

    def generate_test_suites_cpp(self) -> str:
        return "\n".join([h.generate_suite_cpp() for h in self.suites_gens])

    def add_suite(self, f: NativeFunction, test_suite: TestSuite) -> None:
        suites_gen = TestSuiteGen(f, test_suite)
        self.suites_gens.append(suites_gen)
