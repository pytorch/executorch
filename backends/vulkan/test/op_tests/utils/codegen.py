# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from typing import Any, List, Optional, Union

from executorch.backends.vulkan.test.op_tests.utils.codegen_base import (
    AT_INT_ARRAY_REF,
    AT_SCALAR,
    AT_TENSOR,
    BOOL,
    CppTestFileGen,
    DOUBLE,
    INT,
    OPT_AT_TENSOR,
    OPT_BOOL,
    OPT_DEVICE,
    OPT_LAYOUT,
    OPT_SCALARTYPE,
    TestSuite,
    TestSuiteGen,
    THREE_TENSOR_TUPLE,
    TWO_TENSOR_TUPLE,
)
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup

from torchgen.gen import generate_static_dispatch_backend_call
from torchgen.model import NativeFunction

##################################
## Custom Test Suite Definition ##
##################################


@dataclass
class VkTestSuite(TestSuite):
    supports = {
        "storage_types": ["api::StorageType::TEXTURE_3D"],
        "layouts": [
            "api::GPUMemoryLayout::TENSOR_WIDTH_PACKED",
            "api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED",
        ],
    }


##########################
## Code Generator Class ##
##########################


@dataclass
class ATenArg:
    name: str
    cpp_type: str
    default: Optional[str]


@dataclass
class ValueRef:
    name: str
    src_cpp_name: str
    src_cpp_type: str
    is_in: bool = False
    is_out: bool = False
    requires_prepack: bool = False
    supports_prepack: bool = False


ValueRefList = Union[ValueRef, List[ValueRef]]


class ComputeGraphGen:
    def __init__(self, op_reg_name: str, f: NativeFunction, suite_def: TestSuite):
        self.op_reg_name = op_reg_name
        self.f = f
        self.suite_def = suite_def

        self.f_sig = CppSignatureGroup.from_native_function(
            self.f, method=False, fallback_binding=self.f.manual_cpp_binding
        ).most_faithful_signature()

        self.graph = "graph"
        self.dot = "->"

        self.args = []
        self.out = None
        self.refs = {}

        self.should_prepack = False

        for binding in self.f_sig.arguments():
            arg = binding.argument
            ctype = cpp.argumenttype_type(
                arg.type, mutable=arg.is_write, binds=arg.name
            )
            cpp_type = ctype.cpp_type(strip_ref=True)

            self.args.append(
                ATenArg(name=arg.name, cpp_type=cpp_type, default=arg.default)
            )

            requires_prepack = "weight" in arg.name or "bias" in arg.name
            supports_prepack = False
            if arg.name in self.suite_def.prepacked_args:
                supports_prepack = True

            self.refs[arg.name] = ValueRef(
                name=f"{arg.name}_ref",
                src_cpp_name=arg.name,
                src_cpp_type=cpp_type,
                is_in=(cpp_type == AT_TENSOR),
                requires_prepack=requires_prepack,
                supports_prepack=supports_prepack,
            )

        ret_type = cpp.returns_type(self.f.func.returns, symint=False).cpp_type()
        self.out = ATenArg(name="out", cpp_type=ret_type, default=None)
        if ret_type == AT_TENSOR:
            self.refs["out"] = ValueRef(
                name="out_ref", src_cpp_name="out", src_cpp_type=ret_type, is_out=True
            )
        elif ret_type == TWO_TENSOR_TUPLE:
            self.refs["out"] = [
                ValueRef(
                    name="out_ref_first",
                    src_cpp_name="std::get<0>(out)",
                    src_cpp_type="at::Tensor",
                    is_out=True,
                ),
                ValueRef(
                    name="out_ref_second",
                    src_cpp_name="std::get<1>(out)",
                    src_cpp_type="at::Tensor",
                    is_out=True,
                ),
                ValueRef(
                    name="out_ref",
                    src_cpp_name="out",
                    src_cpp_type=ret_type,
                    is_out=False,
                ),
            ]
        elif ret_type == THREE_TENSOR_TUPLE:
            self.refs["out"] = [
                ValueRef(
                    name="out_ref_first",
                    src_cpp_name="std::get<0>(out)",
                    src_cpp_type="at::Tensor",
                    is_out=True,
                ),
                ValueRef(
                    name="out_ref_second",
                    src_cpp_name="std::get<1>(out)",
                    src_cpp_type="at::Tensor",
                    is_out=True,
                ),
                ValueRef(
                    name="out_ref_third",
                    src_cpp_name="std::get<2>(out)",
                    src_cpp_type="at::Tensor",
                    is_out=True,
                ),
                ValueRef(
                    name="out_ref",
                    src_cpp_name="out",
                    src_cpp_type=ret_type,
                    is_out=False,
                ),
            ]

    ## ATen code generation

    def gen_decl(self, fn_name: str, ret_type: str = "void") -> str:
        cpp_args = [a.decl() for a in self.f_sig.arguments()]
        cpp_args_str = ", ".join(cpp_args)
        return f"{ret_type} {fn_name}({cpp_args_str})"

    def create_aten_fn_call(self) -> str:
        func_call = generate_static_dispatch_backend_call(
            self.f_sig, self.f, TestSuiteGen.backend_key
        )[7:].replace("::cpu", "")
        return func_call

    def create_out_src(self) -> str:
        return f"{self.out.cpp_type} out = " + self.create_aten_fn_call()

    ## Graph code generation utils

    def prepack_ref(self, ref: ValueRef) -> bool:
        if ref.requires_prepack:
            return True
        else:
            return ref.supports_prepack and self.should_prepack

    def create_value_for(self, ref: ValueRefList) -> str:  # noqa: C901
        if isinstance(ref, list):
            ret_str = ""
            for r in ref:
                ret_str += self.create_value_for(r)
            return ret_str

        prepack = self.prepack_ref(ref)

        cpp_type = "IOValueRef" if (ref.is_in and not prepack) else "ValueRef"

        if ref.src_cpp_type == OPT_AT_TENSOR:
            ret_str = f"{cpp_type} {ref.name} = "
            ret_str += f"!{ref.src_cpp_name}.has_value() ? "
            ret_str += f"{self.graph}{self.dot}add_none() : "
            if not prepack:
                ret_str += f"{self.graph}{self.dot}"
                ret_str += "add_input_tensor(" if ref.is_in else "add_tensor("
                ret_str += f"{ref.src_cpp_name}->sizes().vec(), "
                ret_str += f"from_at_scalartype({ref.src_cpp_name}->scalar_type())); \n"
            elif prepack:
                ret_str += f"{self.graph}{self.dot}"
                ret_str += f"add_tensorref({ref.src_cpp_name}->sizes().vec(), "
                ret_str += f"from_at_scalartype({ref.src_cpp_name}->scalar_type()), "
                ret_str += f"{ref.src_cpp_name}->const_data_ptr()); \n"
            return ret_str

        ret_str = f"{cpp_type} {ref.name} = {self.graph}{self.dot}"
        if ref.src_cpp_type == AT_TENSOR and not prepack:
            ret_str += "add_input_tensor(" if ref.is_in else "add_tensor("
            ret_str += f"{ref.src_cpp_name}.sizes().vec(), "
            ret_str += f"from_at_scalartype({ref.src_cpp_name}.scalar_type())); \n"
        elif ref.src_cpp_type == AT_TENSOR and prepack:
            ret_str += f"add_tensorref({ref.src_cpp_name}.sizes().vec(), "
            ret_str += f"from_at_scalartype({ref.src_cpp_name}.scalar_type()), "
            ret_str += f"{ref.src_cpp_name}.const_data_ptr()); \n"
        elif ref.src_cpp_type == AT_SCALAR:
            # TODO(ssjia): generalize this to work with all scalar types
            ret_str += f"add_scalar<double>({ref.src_cpp_name}.toDouble()); \n"
        elif ref.src_cpp_type == AT_INT_ARRAY_REF:
            ret_str += f"add_scalar_list({ref.src_cpp_name}.vec()); \n"
        elif ref.src_cpp_type == BOOL:
            ret_str += f"add_scalar<bool>({ref.src_cpp_name}); \n"
        elif ref.src_cpp_type == INT:
            ret_str += f"add_scalar<int64_t>({ref.src_cpp_name}); \n"
        elif ref.src_cpp_type == DOUBLE:
            ret_str += f"add_scalar<double>({ref.src_cpp_name}); \n"
        elif (
            ref.src_cpp_type == OPT_SCALARTYPE
            or ref.src_cpp_type == OPT_LAYOUT
            or ref.src_cpp_type == OPT_DEVICE
            or ref.src_cpp_type == OPT_BOOL
        ):
            ret_str += "add_none(); \n"
        elif ref.src_cpp_type == TWO_TENSOR_TUPLE:
            ret_str += f"add_value_list({{{ref.name}_first, {ref.name}_second}}); \n"
        elif ref.src_cpp_type == THREE_TENSOR_TUPLE:
            ret_str += f"add_value_list({{{ref.name}_first, {ref.name}_second, {ref.name}_third}}); \n"
        else:
            raise RuntimeError(f"Unsupported cpp type {ref.src_cpp_type}")

        return ret_str

    def create_op_call(self) -> str:
        deref = "*" if self.dot == "->" else ""
        op_create_code = f'VK_GET_OP_FN("{self.op_reg_name}")({deref}{self.graph}, {{'

        for aten_arg in self.args:
            ref = self.refs[aten_arg.name]
            op_create_code += (
                f"{ref.name}.value, "
                if (ref.is_in and not self.prepack_ref(ref)) or ref.is_out
                else f"{ref.name}, "
            )

        op_create_code += "out_ref});\n"
        return op_create_code

    def set_output(self, ref: ValueRefList) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.set_output(r)
            return ret_str

        assert ref.src_cpp_type == AT_TENSOR and ref.is_out
        ret_str = f"ValueRef {ref.name}_staging = {self.graph}{self.dot}"
        ret_str += f"set_output_tensor({ref.name});\n"
        return ret_str

    def virtual_resize(self, ref: ValueRefList) -> str:
        assert ref.src_cpp_type == AT_TENSOR and ref.is_in
        if self.prepack_ref(ref):
            return ""
        ret_str = f"{self.graph}{self.dot}get_tensor({ref.name}.value)"
        ret_str += f"->virtual_resize({ref.src_cpp_name}.sizes().vec());\n"
        return ret_str

    def copy_into_staging(self, ref: ValueRefList) -> str:
        assert ref.src_cpp_type == AT_TENSOR and ref.is_in
        if self.prepack_ref(ref):
            return ""
        ret_str = f"{self.graph}{self.dot}copy_into_staging("
        ret_str += f"{ref.name}.staging, "
        ret_str += f"{ref.src_cpp_name}.const_data_ptr(), "
        ret_str += f"{ref.src_cpp_name}.numel());\n"
        return ret_str

    def declare_vk_out_for(self, ref: Union[ValueRef, List[ValueRef]]) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.declare_vk_out_for(r)
            return ret_str

        return f"at::Tensor vk_{ref.name} = at::empty_like({ref.src_cpp_name});\n"

    def copy_from_staging(self, ref: ValueRefList) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.copy_from_staging(r)
            return ret_str

        assert ref.src_cpp_type == AT_TENSOR and ref.is_out
        ret_str = f"{self.graph}{self.dot}copy_from_staging({ref.name}_staging, "
        ret_str += f"vk_{ref.name}.mutable_data_ptr(), vk_{ref.name}.numel());\n"

        return ret_str

    ## Misc. code generation utilities

    def check_graph_out(self, ref: ValueRefList) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.check_graph_out(r)
            return ret_str

        return f"EXPECT_TRUE(check_close({ref.src_cpp_name}, vk_{ref.name}));\n"

    ## Top level code generation

    def gen_graph_build_code(self) -> str:
        graph_build = self.create_out_src()

        for aten_arg in self.args:
            graph_build += self.create_value_for(self.refs[aten_arg.name])

        graph_build += self.create_value_for(self.refs["out"])
        graph_build += self.create_op_call()

        graph_build += self.set_output(self.refs["out"])

        graph_build += f"{self.graph}{self.dot}prepare();\n"
        graph_build += f"{self.graph}{self.dot}encode_prepack();\n"
        graph_build += f"{self.graph}{self.dot}prepack();\n"
        graph_build += f"{self.graph}{self.dot}encode_execute();\n"

        return graph_build

    def gen_graph_exec_code(self) -> str:
        graph_exec = ""
        for aten_arg in self.args:
            ref = self.refs[aten_arg.name]
            if ref.is_in:
                graph_exec += self.virtual_resize(ref)
                graph_exec += self.copy_into_staging(ref)

        graph_exec += f"{self.graph}{self.dot}propagate_resize();\n"
        graph_exec += f"{self.graph}{self.dot}execute();\n"

        graph_exec += self.declare_vk_out_for(self.refs["out"])
        graph_exec += self.copy_from_staging(self.refs["out"])

        return graph_exec

    def gen_op_check_fn(self) -> str:
        op_name = self.f.func.name.unambiguous_name()
        op_check_fn = self.gen_decl(f"check_{op_name}") + " {"
        if self.should_prepack:
            op_check_fn = self.gen_decl(f"prepacked_check_{op_name}") + " {"
        op_check_fn += self.gen_graph_build_code()
        op_check_fn += self.gen_graph_exec_code()
        op_check_fn += self.check_graph_out(self.refs["out"])
        op_check_fn += "}\n"
        return op_check_fn


##################################
## Test Fixture Code Generation ##
##################################

test_fixture_template = """
class GeneratedOpsTest_{op_name} : public ::testing::TestWithParam< ::std::tuple<api::StorageType, api::GPUMemoryLayout>> {{
  protected:
    ComputeGraph* graph;
    at::ScalarType test_dtype = at::kFloat;

    void SetUp() override {{
        GraphConfig config;
        api::StorageType default_storage_type;
        api::GPUMemoryLayout default_memory_layout;
        std::tie(default_storage_type, default_memory_layout) = GetParam();
        config.setStorageTypeOverride(default_storage_type);
        config.setMemoryLayoutOverride(default_memory_layout);
        graph = new ComputeGraph(config);
    }}

    void TearDown() override {{
        delete graph;
        graph = nullptr;
    }}

    {check_fn}

    {prepacked_check_fn}

}};
"""


class VkTestSuiteGen(TestSuiteGen):
    def __init__(self, op_reg_name: str, f: NativeFunction, inputs: List[Any]):
        super().__init__(f, inputs)
        self.op_reg_name = op_reg_name
        self.generator = ComputeGraphGen(self.op_reg_name, self.f, self.suite_def)

    def generate_fixture_cpp(self) -> str:
        check_fn = ""
        if not self.suite_def.requires_prepack:
            check_fn = self.generator.gen_op_check_fn()

        prepacked_check_fn = ""
        if self.suite_def.supports_prepack():
            self.generator.should_prepack = True
            prepacked_check_fn = self.generator.gen_op_check_fn()

        return test_fixture_template.format(
            op_name=self.op_name,
            check_fn=check_fn,
            prepacked_check_fn=prepacked_check_fn,
        )

    def gen_parameterization(self) -> str:
        storage_types = self.suite_def.supports["storage_types"]
        layouts = self.suite_def.supports["layouts"]

        return f"""
        INSTANTIATE_TEST_SUITE_P(
            StorageLayoutCombos_{self.op_name},
            GeneratedOpsTest_{self.op_name},
            ::testing::Combine(
                ::testing::Values({', '.join(storage_types)}),
                ::testing::Values({', '.join(layouts)})));
        """


##############################
## Test File Code Generation ##
###############################

preamble_str = """
#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <tuple>

using namespace vkcompute;
using TensorOptions = at::TensorOptions;

api::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {
    switch(at_scalartype) {
        case c10::kFloat:
            return api::kFloat;
        case c10::kHalf:
            return api::kHalf;
        case c10::kInt:
            return api::kInt;
        case c10::kLong:
            return api::kInt;
        default:
            VK_THROW("Unsupported at::ScalarType!");
    }
}

#ifdef USE_VULKAN_FP16_INFERENCE
bool check_close(at::Tensor& t1, at::Tensor& t2, float rtol=1e-2, float atol=1e-2) {
#else
bool check_close(at::Tensor& t1, at::Tensor& t2, float rtol=1e-5, float atol=1e-5) {
#endif
    // Skip checking index tensors
    if (t1.scalar_type() == at::kLong || t2.scalar_type() == at::kLong) {
        return true;
    }
    bool is_close = at::allclose(t1, t2, rtol, atol);
    if (!is_close) {
        std::cout << "t1:" << t1 << std::endl;
        std::cout << "t2:" << t2 << std::endl;
    }
    return is_close;
}
"""


class VkCppTestFileGen(CppTestFileGen):
    def __init__(self, out_path: str):
        super().__init__(out_path)

    def generate_preamble(self) -> str:
        return preamble_str

    def add_suite(self, op_reg_name: str, f: NativeFunction, all_input_cases) -> None:
        suites_gen = VkTestSuiteGen(op_reg_name, f, all_input_cases)
        self.suites_gens.append(suites_gen)
