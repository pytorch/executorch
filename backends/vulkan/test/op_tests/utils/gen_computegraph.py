# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass
from typing import List, Optional, Union

from executorch.backends.vulkan.test.op_tests.utils.aten_types import (
    AT_INT_ARRAY_REF,
    AT_SCALAR,
    AT_TENSOR,
    AT_TENSOR_LIST,
    BOOL,
    DOUBLE,
    INT,
    OLD_STRING,
    OPT_AT_DOUBLE_ARRAY_REF,
    OPT_AT_INT_ARRAY_REF,
    OPT_AT_TENSOR,
    OPT_BOOL,
    OPT_DEVICE,
    OPT_INT64,
    OPT_LAYOUT,
    OPT_MEMORY_FORMAT,
    OPT_SCALAR_TYPE,
    STRING,
    TENSOR_VECTOR,
    THREE_TENSOR_TUPLE,
    TWO_TENSOR_TUPLE,
)
from executorch.backends.vulkan.test.op_tests.utils.test_suite import TestSuite

from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.gen import generate_static_dispatch_backend_call, translate_args
from torchgen.gen_aoti_c_shim import gen_static_dispatch_backend_call_signature
from torchgen.model import NativeFunction, Variant

###################################
## Compute Graph Code Generation ##
###################################


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
    # When is_dynamic_size is true, the underlying object size is not known
    # during code-gen. Example is the out value for aten.split where the out
    # value is a vector<Tensor>. In these cases, we need to use an additional
    # vector or at::TensorList to track these values.
    is_dynamic_size: bool = False

    @property
    def io_value_list_name(self):
        assert self.is_dynamic_size
        return f"{self.name}_io_value_list"

    @property
    def value_list_name(self):
        assert self.is_dynamic_size
        return f"{self.name}_value_list"

    @property
    def vk_out(self):
        assert self.is_out
        return f"vk_{self.name}"


ValueRefList = Union[ValueRef, List[ValueRef]]

InableCppType = frozenset([AT_TENSOR, AT_TENSOR_LIST])


class ComputeGraphGen:
    backend_key = None

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

            # These are the argument will be passed as a "weight" tensor, the
            # corresponding object will be TensorRef in the compute graph.
            requires_prepack = (
                "weight" in arg.name
                or "bias" in arg.name
                or "running_mean" in arg.name
                or "running_var" in arg.name
            )
            supports_prepack = False
            if arg.name in self.suite_def.prepacked_args:
                supports_prepack = True

            self.refs[arg.name] = ValueRef(
                name=f"{arg.name}_ref",
                src_cpp_name=arg.name,
                src_cpp_type=cpp_type,
                is_in=(cpp_type in InableCppType),
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
        elif ret_type == TENSOR_VECTOR:
            self.refs["out"] = ValueRef(
                name="out_ref",
                src_cpp_name="out",
                src_cpp_type=ret_type,
                is_out=True,
                is_dynamic_size=True,
            )
        else:
            raise NotImplementedError(
                f"ret_type: {ret_type} not supported for out value"
            )

    ## ATen code generation

    def gen_decl(self, fn_name: str, ret_type: str = "void") -> str:
        cpp_args = [a.decl() for a in self.f_sig.arguments()]
        cpp_args_str = ", ".join(cpp_args)
        return f"{ret_type} {fn_name}({cpp_args_str})"

    def create_aten_fn_call(self) -> str:
        func_call = generate_static_dispatch_backend_call(
            self.f_sig, self.f, ComputeGraphGen.backend_key
        )[7:].replace("::cpu", "")

        return func_call

    def create_aten_method_call(self) -> str:
        # For functions with only Method variant, we fallback to the function
        # declared in MethodOperators.h. The method is declared as
        # at::_ops::{name}::call(*), and ATEN_FN is a handly macro.
        cpp_sig = gen_static_dispatch_backend_call_signature(self.f_sig, self.f)
        exprs = translate_args(self.f_sig, cpp_sig)
        func_call = f"ATEN_FN({self.f_sig.name()})({exprs});"
        return func_call

    def create_out_src(self, include_declarations: bool = True) -> str:
        cpp_type = self.out.cpp_type if include_declarations else ""
        if Variant.function in self.f.variants:
            return f"{cpp_type} out = " + self.create_aten_fn_call() + "\n"
        else:
            return f"{cpp_type} out = " + self.create_aten_method_call() + "\n"

    ## Graph code generation utils

    def prepack_ref(self, ref: ValueRef) -> bool:
        if ref.requires_prepack:
            return True
        else:
            return ref.supports_prepack and self.should_prepack

    def create_value_decl_for(self, ref: ValueRefList) -> str:  # noqa: C901
        if isinstance(ref, list):
            ret_str = ""
            for r in ref:
                ret_str += self.create_value_decl_for(r)
            return ret_str

        cpp_type = "IOValueRef" if (ref.is_in or ref.requires_prepack) else "ValueRef"
        if ref.src_cpp_type == AT_TENSOR_LIST:
            ret_str = f"std::vector<IOValueRef> {ref.name}_io_value_refs;\n"
            ret_str += f"std::vector<ValueRef> {ref.name}_value_refs;\n"
            return ret_str
        elif ref.src_cpp_type == TENSOR_VECTOR:
            ret_str = f"std::vector<IOValueRef> {ref.io_value_list_name};\n"
            ret_str += f"std::vector<ValueRef> {ref.value_list_name};\n"
            return ret_str
        else:
            return f"{cpp_type} {ref.name};\n"

    def create_value_for(  # noqa: C901
        self, ref: ValueRefList, include_declarations: bool = True
    ) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref:
                ret_str += self.create_value_for(r)
            return ret_str

        prepack = self.prepack_ref(ref)
        ref_is_view = self.suite_def.is_view_op and ref.is_out

        cpp_type = "IOValueRef" if (ref.is_in and not prepack) else "ValueRef"
        if not include_declarations:
            cpp_type = ""

        if ref.src_cpp_type == OPT_AT_TENSOR:
            ret_str = f"{cpp_type} {ref.name} = "
            if prepack:
                ret_str = ""
                if include_declarations:
                    ret_str += f"IOValueRef {ref.name};\n"
                ret_str += f"{ref.name}.value = "
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
        elif ref.src_cpp_type == OPT_INT64:
            ret_str = f"{cpp_type} {ref.name} = "
            ret_str += f"!{ref.src_cpp_name}.has_value() ? "
            ret_str += f"{self.graph}{self.dot}add_none() : "
            ret_str += f"{self.graph}{self.dot}add_scalar<int64_t>"
            ret_str += f"({ref.src_cpp_name}.value());\n"
            return ret_str
        elif (
            ref.src_cpp_type == OPT_AT_DOUBLE_ARRAY_REF
            or ref.src_cpp_type == OPT_AT_INT_ARRAY_REF
        ):
            ret_str = f"{cpp_type} {ref.name} = "
            ret_str += f"!{ref.src_cpp_name}.has_value() ? "
            ret_str += f"{self.graph}{self.dot}add_none() : "
            ret_str += f"{self.graph}{self.dot}add_scalar_list"
            ret_str += f"({ref.src_cpp_name}->vec());\n"
            return ret_str
        elif ref.src_cpp_type == AT_TENSOR_LIST:
            assert ref.is_in, "AT_TENSOR_LIST must be an input"
            # This logic is a bit convoluted. We need to create a IOValueRef for
            # each tensor, to facilate staging. On the other hand, we will
            # use the .value tensor to create a ValueList, which will be passed
            # to the corresponding ops.
            ret_str = ""
            if include_declarations:
                ret_str += f"std::vector<IOValueRef> {ref.name}_io_value_refs;\n"
                ret_str += f"std::vector<ValueRef> {ref.name}_value_refs;\n"
            ret_str += f"for (int i=0; i < {ref.src_cpp_name}.size(); i++) {{\n"
            ret_str += (
                f"  IOValueRef io_value_ref = {self.graph}{self.dot}add_input_tensor(\n"
            )
            ret_str += f"      {ref.src_cpp_name}[i].sizes().vec(),\n"
            ret_str += (
                f"      from_at_scalartype({ref.src_cpp_name}[i].scalar_type())); \n"
            )
            ret_str += f"  {ref.name}_value_refs.emplace_back(io_value_ref.value);\n"
            ret_str += f"  {ref.name}_io_value_refs.emplace_back(io_value_ref);\n"
            ret_str += "}\n"
            ret_str += f"ValueRef {ref.name} = {self.graph}{self.dot}add_value_list(std::move({ref.name}_value_refs));\n"
            return ret_str
        elif ref.src_cpp_type == TENSOR_VECTOR:
            ret_str = ""
            if include_declarations:
                ret_str += f"std::vector<IOValueRef> {ref.io_value_list_name};\n"
                ret_str += f"std::vector<ValueRef> {ref.value_list_name};\n"
            ret_str += f"""
for (int i=0; i<out.size(); i++) {{
    const at::Tensor& cur = out[i];
    IOValueRef io_value_ref;
    io_value_ref.value = {self.graph}{self.dot}add_tensor(
        cur.sizes().vec(), from_at_scalartype(cur.scalar_type()));
    {ref.io_value_list_name}.emplace_back(io_value_ref);
    {ref.value_list_name}.emplace_back(io_value_ref.value);
}}
ValueRef out_ref = {self.graph}{self.dot}add_value_list(std::move({ref.value_list_name}));
"""
            return ret_str

        ret_str = f"{cpp_type} {ref.name} = {self.graph}{self.dot}"
        if prepack:
            ret_str = ""
            if include_declarations:
                ret_str = f"IOValueRef {ref.name};\n"
            ret_str += f"{ref.name}.value = {self.graph}{self.dot}"

        if ref.src_cpp_type == AT_TENSOR and ref_is_view:
            input_name = None
            for _name, ref in self.refs.items():
                if ref.is_in and ref.src_cpp_type == AT_TENSOR:
                    input_name = ref.name

            assert input_name is not None
            ret_str += f"add_tensor_view({input_name}.value);"
        elif ref.src_cpp_type == AT_TENSOR and not prepack:
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
            ref.src_cpp_type == OPT_SCALAR_TYPE
            or ref.src_cpp_type == OPT_LAYOUT
            or ref.src_cpp_type == OPT_DEVICE
            or ref.src_cpp_type == OPT_BOOL
            or ref.src_cpp_type == OPT_MEMORY_FORMAT
        ):
            ret_str += "add_none(); \n"
        elif ref.src_cpp_type == STRING or ref.src_cpp_type == OLD_STRING:
            ret_str += f"add_string(std::string({ref.src_cpp_name})); \n"
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
            if ref.src_cpp_type == AT_TENSOR_LIST:
                # Special case. Underlying tensors are input tensors, but the
                # container itself is just a normal value.
                op_create_code += f"{ref.name}, "
            else:
                op_create_code += (
                    f"{ref.name}.value, "
                    if ref.is_in or ref.requires_prepack or ref.is_out
                    else f"{ref.name}, "
                )
                # op_create_code += f"{ref.name}, "

        op_create_code += "out_ref});\n"
        return op_create_code

    def gen_output_staging_valueref_decl(self, ref: ValueRefList) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.gen_output_staging_valueref_decl(r)
            return ret_str
        elif ref.src_cpp_type == TENSOR_VECTOR:
            assert ref.is_out
            ret_str = ""
            return ret_str

        assert ref.src_cpp_type == AT_TENSOR and ref.is_out
        return f"ValueRef {ref.name}_staging;\n"

    def set_output(self, ref: ValueRefList, include_declarations: bool = True) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.set_output(r, include_declarations)
            return ret_str
        elif ref.src_cpp_type == TENSOR_VECTOR:
            assert ref.is_out
            ret_str = f"""
for (int i=0; i<out.size(); i++) {{
    {ref.io_value_list_name}[i].staging = {self.graph}{self.dot}set_output_tensor(
        {ref.io_value_list_name}[i].value);
}}
"""
            return ret_str

        assert ref.src_cpp_type == AT_TENSOR and ref.is_out
        cpptype = "ValueRef" if include_declarations else ""
        ret_str = f"{cpptype} {ref.name}_staging = {self.graph}{self.dot}"
        ret_str += f"set_output_tensor({ref.name});\n"
        return ret_str

    def virtual_resize(self, ref: ValueRefList) -> str:
        assert isinstance(ref, ValueRef)
        assert ref.src_cpp_type in InableCppType and ref.is_in
        if self.prepack_ref(ref):
            return ""

        if ref.src_cpp_type == AT_TENSOR:
            ret_str = f"{self.graph}{self.dot}get_tensor({ref.name}.value)"
            ret_str += f"->virtual_resize({ref.src_cpp_name}.sizes().vec());\n"
        elif ref.src_cpp_type == AT_TENSOR_LIST:
            ret_str = ""
            ret_str += f"for (int i=0; i < {ref.name}_io_value_refs.size(); i++) {{\n"
            ret_str += (
                f"  {self.graph}{self.dot}get_tensor({ref.name}_io_value_refs[i].value)"
            )
            ret_str += f"->virtual_resize({ref.src_cpp_name}[i].sizes().vec());\n"
            ret_str += "}\n"
        else:
            raise AssertionError(f"{ref.src_cpp_type} not expected")

        return ret_str

    def copy_into_staging(self, ref: ValueRefList) -> str:
        assert isinstance(ref, ValueRef)
        assert ref.src_cpp_type in InableCppType and ref.is_in

        if self.prepack_ref(ref):
            return ""

        if ref.src_cpp_type == AT_TENSOR:
            ret_str = f"{self.graph}{self.dot}copy_into_staging("
            ret_str += f"{ref.name}.staging, "
            ret_str += f"{ref.src_cpp_name}.const_data_ptr(), "
            ret_str += f"{ref.src_cpp_name}.numel());\n"
        elif ref.src_cpp_type == AT_TENSOR_LIST:
            ret_str = ""
            ret_str += f"for (int i=0; i < {ref.name}_io_value_refs.size(); i++) {{\n"
            ret_str += f"  {self.graph}{self.dot}copy_into_staging("
            ret_str += f"{ref.name}_io_value_refs[i].staging, "
            ret_str += f"{ref.src_cpp_name}[i].const_data_ptr(), "
            ret_str += f"{ref.src_cpp_name}[i].numel());\n"
            ret_str += "}\n"
        else:
            raise AssertionError(f"{ref.src_cpp_type} not expected")
        return ret_str

    def declare_vk_out_for(self, ref: Union[ValueRef, List[ValueRef]]) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.declare_vk_out_for(r)
            return ret_str
        elif ref.src_cpp_type == TENSOR_VECTOR:
            assert ref.is_out
            ret_str = f"""
std::vector<at::Tensor> {ref.vk_out};
for (int i=0; i<out.size(); i++) {{
    {ref.vk_out}.emplace_back(at::empty_like(out[i]).contiguous());
}}
"""
            return ret_str

        assert ref.src_cpp_type == AT_TENSOR and ref.is_out
        ret_str = f"at::Tensor vk_{ref.name} = at::empty_like({ref.src_cpp_name})"
        ret_str += ".contiguous();\n"
        return ret_str

    def copy_from_staging(self, ref: ValueRefList) -> str:
        if isinstance(ref, list):
            ret_str = ""
            for r in ref[:-1]:
                ret_str += self.copy_from_staging(r)
            return ret_str
        elif ref.src_cpp_type == TENSOR_VECTOR:
            assert ref.is_out
            ret_str = f"""
for (int i=0; i<out.size(); i++) {{
    {self.graph}{self.dot}copy_from_staging(
        {ref.io_value_list_name}[i].staging,
        {ref.vk_out}[i].mutable_data_ptr(),
        {ref.vk_out}[i].numel());
}}
"""
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
        elif ref.src_cpp_type == TENSOR_VECTOR:
            assert ref.is_out
            ret_str = f"""
for (int i=0; i<out.size(); i++) {{
    EXPECT_TRUE(check_close(out[i], {ref.vk_out}[i], rtol, atol));
}}
"""
            return ret_str

        return (
            f"EXPECT_TRUE(check_close({ref.src_cpp_name}, vk_{ref.name}, rtol, atol));"
        )

    ## Top level code generation

    def gen_arg_valueref_decls(self) -> str:
        ret_str = ""
        for aten_arg in self.args:
            ref = self.refs[aten_arg.name]
            ret_str += self.create_value_decl_for(ref)

        ret_str += self.create_value_decl_for(self.refs["out"])
        ret_str += f"{self.out.cpp_type} out;\n"
        ret_str += self.gen_output_staging_valueref_decl(self.refs["out"])
        return ret_str

    def gen_graph_build_code(self, include_declarations: bool = True) -> str:
        graph_build = self.create_out_src(include_declarations)
        for aten_arg in self.args:
            graph_build += self.create_value_for(
                self.refs[aten_arg.name], include_declarations
            )

        graph_build += self.create_value_for(self.refs["out"], include_declarations)
        graph_build += self.create_op_call()

        graph_build += self.set_output(self.refs["out"], include_declarations)

        graph_build += f"{self.graph}{self.dot}prepare();\n"
        graph_build += f"{self.graph}{self.dot}encode_prepack();\n"
        graph_build += f"{self.graph}{self.dot}prepack();\n"
        graph_build += f"{self.graph}{self.dot}encode_execute();\n"

        graph_build += "\n"
        return graph_build

    def gen_graph_exec_code(self, check_output=True) -> str:
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
        if check_output:
            graph_exec += self.check_graph_out(self.refs["out"])

        graph_exec = re.sub(r"^", "  ", graph_exec, flags=re.M)
        graph_exec = "{\n" + graph_exec + "\n}"

        return graph_exec

    def gen_conditional_skips(self, skip_str: str = "GTEST_SKIP();") -> str:
        fp16_skip = f"if (!{self.graph}{self.dot}context()->adapter_ptr()->has_full_float16_buffers_support()) {{\n"
        fp16_skip += f"  {skip_str}\n"
        fp16_skip += "}"
        fp16_skip = re.sub(r"^", "  ", fp16_skip, flags=re.M) + "\n"

        int8_skip = f"if (!{self.graph}{self.dot}context()->adapter_ptr()->has_full_int8_buffers_support()) {{\n"
        int8_skip += f"  {skip_str};\n"
        int8_skip += "}\n"

        skips = ""

        skips += "if (test_dtype == at::kHalf) {\n"
        skips += fp16_skip
        skips += "}\n"

        for _, dtype in self.suite_def.arg_dtype.items():
            if dtype == "at::kChar" or dtype == "at::kQInt8":
                skips += int8_skip
                continue

        skips += "\n"
        return skips

    def gen_op_check_fn(self) -> str:
        op_name = self.f.func.name.unambiguous_name()
        if self.suite_def.test_name_suffix is not None:
            op_name += "_" + self.suite_def.test_name_suffix

        op_check_fn = self.gen_decl(f"check_{op_name}") + " {\n"
        if self.should_prepack:
            op_check_fn = self.gen_decl(f"prepacked_check_{op_name}") + " {\n"

        op_check_fn_body = ""
        op_check_fn_body += self.gen_conditional_skips()
        op_check_fn_body += self.gen_graph_build_code()
        op_check_fn_body += self.gen_graph_exec_code()

        op_check_fn_body = re.sub(r"^", "    ", op_check_fn_body, flags=re.M)

        op_check_fn += op_check_fn_body
        op_check_fn += "\n  }"

        return op_check_fn

    def gen_build_graph_fn(self, include_declarations: bool = False) -> str:
        op_name = self.f.func.name.unambiguous_name()
        if self.suite_def.test_name_suffix is not None:
            op_name += "_" + self.suite_def.test_name_suffix
        op_build_graph_fn = self.gen_decl(f"build_graph_{op_name}") + " {\n"
        if self.should_prepack:
            op_build_graph_fn = (
                self.gen_decl(f"prepacked_build_graph_{op_name}") + " {\n"
            )

        op_build_graph_fn_body = ""
        op_build_graph_fn_body += self.gen_graph_build_code(include_declarations)

        op_build_graph_fn += op_build_graph_fn_body
        op_build_graph_fn += "\n  }"
        return op_build_graph_fn

    def gen_op_exec_graph_fn(self) -> str:
        op_name = self.f.func.name.unambiguous_name()
        if self.suite_def.test_name_suffix is not None:
            op_name += "_" + self.suite_def.test_name_suffix
        op_benchmark_fn = self.gen_decl(f"benchmark_{op_name}") + " {\n"
        if self.should_prepack:
            op_benchmark_fn = self.gen_decl(f"prepacked_benchmark_{op_name}") + " {\n"

        op_benchmark_fn_body = ""
        op_benchmark_fn_body += self.gen_graph_exec_code(False)

        op_benchmark_fn_body = re.sub(r"^", "    ", op_benchmark_fn_body, flags=re.M)

        op_benchmark_fn += op_benchmark_fn_body
        op_benchmark_fn += "\n  }"
        return op_benchmark_fn
