# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
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
AT_TENSOR_LIST = "at::TensorList"
BOOL = "bool"
DOUBLE = "double"
INT = "int64_t"
OPT_AT_DOUBLE_ARRAY_REF = "::std::optional<at::ArrayRef<double>>"
OPT_AT_INT_ARRAY_REF = "at::OptionalIntArrayRef"
OPT_AT_TENSOR = "::std::optional<at::Tensor>"
OPT_BOOL = "::std::optional<bool>"
OPT_INT64 = "::std::optional<int64_t>"
OPT_DEVICE = "::std::optional<at::Device>"
OPT_LAYOUT = "::std::optional<at::Layout>"
OPT_MEMORY_FORMAT = "::std::optional<at::MemoryFormat>"
OPT_SCALAR_TYPE = "::std::optional<at::ScalarType>"
STRING = "c10::string_view"
TWO_TENSOR_TUPLE = "::std::tuple<at::Tensor,at::Tensor>"
THREE_TENSOR_TUPLE = "::std::tuple<at::Tensor,at::Tensor,at::Tensor>"
TENSOR_VECTOR = "::std::vector<at::Tensor>"

###########################
## Test Suite definition ##
###########################


class TestSuite:
    def __init__(self, input_cases: List[Any]):
        self.input_cases: List[Any] = input_cases
        self.prepacked_args: List[str] = []
        self.requires_prepack: bool = False
        self.dtypes: List[str] = ["at::kFloat", "at::kHalf"]

        self.data_gen: str = "make_rand_tensor"
        self.data_range = (0, 1)

        self.arg_dtype = {}
        self.arg_data_range = {}

        self.atol: str = "1e-5"
        self.rtol: str = "1e-5"

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

    list_str = "{"
    for s in pylist:
        if isinstance(s, (list, tuple)):
            list_str += f"{init_list_str(s)}, "
        else:
            list_str += f"{s}, "
    list_str = list_str[:-2] + "}"
    return list_str


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

    def gen_case_name_tuple(self, t) -> str:
        return "x".join(
            [
                (
                    str(e)
                    if not isinstance(e, (list, tuple))
                    else self.gen_case_name_tuple(e)
                )
                for e in t
            ]
        )

    def gen_case_name(self, inputs: List[Any], prepack: bool = False) -> str:
        name_str = self.op_name
        if prepack:
            name_str += "_prepack"
        for arg_sizes_or_val in inputs:
            name_str += "_"
            if isinstance(arg_sizes_or_val, tuple):
                name_str += self.gen_case_name_tuple(arg_sizes_or_val)
            elif isinstance(arg_sizes_or_val, list):
                lst = []
                for size in arg_sizes_or_val:
                    if isinstance(size, (list, tuple)):
                        lst.append(self.gen_case_name_tuple(size))
                    else:
                        lst.append(str(size))
                name_str += "c".join(lst)
            else:
                name_str += str(arg_sizes_or_val).replace(".", "p")

        # minus sign is a invalid char for test case. change to "n".
        name_str = name_str.replace("-", "n")
        return name_str

    def call_data_gen_fn(self, arg: Argument, data: Any, terminate: bool = True) -> str:
        tensor_dtype = (
            "test_dtype"
            if arg.name not in self.suite_def.arg_dtype
            else self.suite_def.arg_dtype[arg.name]
        )

        data_range = (
            self.suite_def.data_range
            if arg.name not in self.suite_def.arg_data_range
            else self.suite_def.arg_data_range[arg.name]
        )

        ret_str = f"{self.suite_def.data_gen}({init_list_str(data)}, {tensor_dtype}, {data_range[0]}, {data_range[1]})"
        if terminate:
            ret_str += ";"

        return ret_str

    def create_input_data(self, arg: Argument, data: Any) -> str:  # noqa: C901
        ctype = cpp.argumenttype_type(arg.type, mutable=arg.is_write, binds=arg.name)
        cpp_type = ctype.cpp_type(strip_ref=True)

        # Short cut exit for TENSORLIST, because it needs multiple lines of
        # construction, deviates from the rest.
        if cpp_type == AT_TENSOR_LIST:
            ret_str = f"std::vector<{AT_TENSOR}> tensor_vec;\n"
            for elem in data:
                ret_str += f"tensor_vec.emplace_back({self.call_data_gen_fn(arg, elem, False)});\n"
            ret_str += f"{cpp_type} {arg.name} = tensor_vec;\n"
            return ret_str + "\n"

        if cpp_type == AT_INT_ARRAY_REF:
            ret_str = f"std::vector<int64_t> {arg.name} = "
        elif cpp_type == OPT_AT_DOUBLE_ARRAY_REF and str(data) != "None":
            ret_str = f"std::vector<double> {arg.name} = "
        elif cpp_type == OPT_AT_INT_ARRAY_REF and str(data) != "None":
            ret_str = f"std::vector<int64_t> {arg.name} = "
        else:
            ret_str = f"{cpp_type} {arg.name} = "

        if cpp_type == AT_TENSOR:
            if arg.name == "index" or arg.name == "indices":
                ret_str += f"make_index_tensor({init_list_str(data)});"
            else:
                ret_str += self.call_data_gen_fn(arg, data)
        elif cpp_type == OPT_AT_TENSOR:
            if str(data) == "None":
                ret_str += "std::nullopt;"
            else:
                ret_str += self.call_data_gen_fn(arg, data)
        elif cpp_type == AT_SCALAR:
            ret_str += f"{data};"
        elif cpp_type == AT_INT_ARRAY_REF:
            ret_str += f"{init_list_str(data)};"
        elif cpp_type == OPT_AT_DOUBLE_ARRAY_REF or cpp_type == OPT_AT_INT_ARRAY_REF:
            if str(data) == "None":
                ret_str += "std::nullopt;"
            else:
                ret_str += f"{init_list_str(data)};"
        elif cpp_type == BOOL:
            ret_str += f"{str(data).lower()};"
        elif cpp_type == INT:
            ret_str += f"{str(data).lower()};"
        elif cpp_type == DOUBLE:
            ret_str += f"{str(data).lower()};"
        elif cpp_type == OPT_INT64:
            if str(data) == "None":
                ret_str += "std::nullopt;"
            else:
                ret_str += f"{str(data)};"
        elif cpp_type == STRING:
            ret_str += f'c10::string_view("{data}");'
        elif (
            cpp_type == OPT_SCALAR_TYPE
            or cpp_type == OPT_LAYOUT
            or cpp_type == OPT_DEVICE
            or cpp_type == OPT_BOOL
            or cpp_type == OPT_MEMORY_FORMAT
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

        ref_code = re.sub(r"^", "  ", ref_code, flags=re.M)
        return ref_code

    def gen_create_and_check_out(self, prepack=False) -> str:
        test_str = f"check_{self.op_name}("
        if prepack:
            test_str = f"prepacked_check_{self.op_name}("
        for binding in self.f_sig.arguments():
            arg = binding.argument
            test_str += f"{arg.name}, "
        test_str = test_str[:-2] + ");"
        test_str = re.sub(r"^", "  ", test_str, flags=re.M)
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
        suite_cpp = self.generate_fixture_cpp()
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
    float low = 0.0,
    float high = 1.0) {{
  if (high == 1.0 && low == 0.0)
    return at::rand(sizes, at::device(at::kCPU).dtype(dtype));
    
  if (dtype == at::kChar)
    return at::randint(high, sizes, at::device(at::kCPU).dtype(dtype));

  return at::rand(sizes, at::device(at::kCPU).dtype(dtype)) * (high - low) + low;
}}

at::Tensor make_seq_tensor(
    std::vector<int64_t> sizes,
    at::ScalarType dtype = at::kFloat,
    float low = 0.0,
    float high = 1.0) {{
  (void)low;
  (void)high;

  int64_t n = 1;
  for (auto size: sizes) {{
    n *= size;
  }}

  std::vector<float> values(n);
  for (int i=0;i<n;i++) {{
    values[i] = (float) i;
  }}

  // Clone as original data will be deallocated upon return.
  return at::from_blob(values.data(), sizes, at::kFloat).toType(dtype).detach().clone();
}}

at::Tensor make_index_tensor(std::vector<int64_t> indices) {{
  at::ScalarType dtype = at::kInt;
  std::vector<int64_t> sizes = {{static_cast<int64_t>(indices.size())}};

  // Clone as original data will be deallocated upon return.
  return at::from_blob(indices.data(), sizes, dtype).detach().clone();
}}

at::Tensor make_index_tensor(std::vector<std::vector<int64_t>> indices) {{
  at::ScalarType dtype = at::kInt;
  std::vector<int64_t> sizes = {{
    static_cast<int64_t>(indices.size()),
    static_cast<int64_t>(indices[0].size())}};

  // Flatten indices as from_blob reads garbage otherwise.
  std::vector<int64_t> acc;
  for (auto& vec: indices) {{
    acc.insert(acc.end(), vec.begin(), vec.end());
  }}

  // Clone as original data will be deallocated upon return.
  return at::from_blob(acc.data(), sizes, dtype).detach().clone();
}}

at::Tensor make_index_tensor(std::vector<std::vector<std::vector<int64_t>>> indices) {{
  at::ScalarType dtype = at::kInt;
  std::vector<int64_t> sizes = {{
    static_cast<int64_t>(indices.size()),
    static_cast<int64_t>(indices[0].size()),
    static_cast<int64_t>(indices[0][0].size())}};

  // Flatten indices as from_blob reads garbage otherwise.
  std::vector<int64_t> acc;
  for (auto& v: indices) {{
    for (auto& vv: v) {{
      acc.insert(acc.end(), vv.begin(), vv.end());
    }}
  }}

  // Clone as original data will be deallocated upon return.
  return at::from_blob(acc.data(), sizes, dtype).detach().clone();
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

    def add_suite(self, op_reg_name: str, f: NativeFunction, all_input_cases) -> None:
        suites_gen = TestSuiteGen(f, all_input_cases)
        self.suites_gens.append(suites_gen)
