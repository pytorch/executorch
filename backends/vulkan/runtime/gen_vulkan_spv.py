#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import array
import codecs
import copy
import glob
import hashlib
import io
import os
import re
import shutil
import sys
from itertools import product
from multiprocessing.pool import ThreadPool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[assignment, misc]

CPP_H_NAME = "spv.h"
CPP_SRC_NAME = "spv.cpp"

# Basic configuration settings for shaders
DEFAULT_ENV: Dict[str, Any] = {
    "PRECISION": "highp",
    # B is shorthand for "binding". This is used to automatically increment the
    # layout binding index when declaring layout bindings. Note that a container
    # type is used because integers are immutable in Python.
    "B": [0],
    # C is shorthand for "constant_id". This is used to automatically increment the
    # constant_id index for specialization constants.
    # Note that it starts at 3, as 0-2 are reserved for local workgroup size ids.
    "C": [3],
}

# Establishes relationships between different tensor types and different GLSL types
TYPE_MAPPINGS: Dict[str, Any] = {
    "IMAGE_T": {
        3: {
            "double": "image3D",
            "float": "image3D",
            "half": "image3D",
            # integer dtypes
            "int8": "iimage3D",
            "uint8": "uimage3D",
            "int16": "iimage3D",
            "uint16": "uimage3D",
            "int32": "iimage3D",
            "uint32": "uimage3D",
            "int64": "iimage3D",
            "uint64": "uimage3D",
            # common dtype aliases
            "bool": "uimage3D",
            "int": "iimage3D",
            "uint": "uimage3D",
        },
        2: {
            "double": "image2D",
            "float": "image2D",
            "half": "image2D",
            # integer dtypes
            "int8": "iimage2D",
            "uint8": "uimage2D",
            "int16": "iimage2D",
            "uint16": "uimage2D",
            "int32": "iimage2D",
            "uint32": "uimage2D",
            "int64": "iimage2D",
            "uint64": "uimage2D",
            # common dtype aliases
            "bool": "uimage2D",
            "int": "iimage2D",
            "uint": "uimage2D",
        },
    },
    "SAMPLER_T": {
        3: {
            "double": "sampler3D",
            "float": "sampler3D",
            "half": "sampler3D",
            # integer dtypes
            "int8": "isampler3D",
            "uint8": "usampler3D",
            "int16": "isampler3D",
            "uint16": "usampler3D",
            "int32": "isampler3D",
            "uint32": "usampler3D",
            "int64": "isampler3D",
            "uint64": "usampler3D",
            # common dtype aliases
            "bool": "usampler3D",
            "int": "isampler3D",
            "uint": "usampler3D",
        },
        2: {
            "double": "sampler2D",
            "float": "sampler2D",
            "half": "sampler2D",
            # integer dtypes
            "int8": "isampler2D",
            "uint8": "usampler2D",
            "int16": "isampler2D",
            "uint16": "usampler2D",
            "int32": "isampler2D",
            "uint32": "usampler2D",
            "int64": "isampler2D",
            "uint64": "usampler2D",
            # common dtype aliases
            "bool": "usampler2D",
            "int": "isampler2D",
            "uint": "usampler2D",
        },
    },
    "IMAGE_FORMAT": {
        "double": "rgba32f",
        "float": "rgba32f",
        "half": "rgba16f",
        # integer dtypes
        "int8": "rgba8i",
        "uint8": "rgba8ui",
        "int16": "rgba16i",
        "uint16": "rgba16ui",
        "int32": "rgba32i",
        "uint32": "rgba32ui",
        "int64": "rgba32i",
        "uint64": "rgba32ui",
        # common dtype aliases
        "bool": "rgba8ui",
        "int": "rgba32i",
        "uint": "rgba32ui",
    },
}


def define_variable(name: str) -> str:
    if name in locals():
        return f"#define {name} {locals()[name]}"
    elif name in globals():
        return f"#define {name} {globals()[name]}"
    else:
        raise RuntimeError(f"{name} is not defined")


def buffer_scalar_type(dtype: str) -> str:
    if dtype == "half":
        return "float16_t"
    elif dtype == "float":
        return "float"
    elif dtype == "double":
        return "float64_t"
    # integer dtype alias conversion
    elif dtype == "bool":
        return "uint8_t"
    # we don't want to append _t for int32 or uint32 as int is already 32bit
    elif dtype == "int32" or dtype == "uint32":
        return "int" if dtype == "int32" else "uint"
    elif dtype[-1].isdigit():
        return dtype + "_t"
    return dtype


def buffer_gvec_type(dtype: str, n: int) -> str:
    if n == 1:
        return buffer_scalar_type(dtype)

    dtype_map = {
        "half": f"f16vec{n}",
        "float": f"vec{n}",
        "double": f"vec{n}",  # No 64bit image format support in GLSL
        "int8": f"i8vec{n}",
        "uint8": f"u8vec{n}",
        "int16": f"i16vec{n}",
        "uint16": f"u16vec{n}",
        "int32": f"ivec{n}",
        "int": f"ivec{n}",
        "uint32": f"uvec{n}",
        "uint": f"uvec{n}",
        "int64": f"ivec{n}",  # No 64bit image format support in GLSL
        "uint64": f"uvec{n}",  # No 64bit image format support in GLSL
        "bool": f"u8vec{n}",
    }

    vector_type = dtype_map.get(dtype)
    if vector_type is None:
        raise AssertionError(f"Invalid dtype: {dtype}")

    return vector_type


def texel_type(dtype: str) -> str:
    image_format = TYPE_MAPPINGS["IMAGE_FORMAT"][dtype]
    if image_format[-1:] == "f":
        return "vec4"
    elif image_format[-2:] == "ui":
        return "uvec4"
    elif image_format[-1:] == "i":
        return "ivec4"
    raise AssertionError(f"Invalid image format: {image_format}")


def gvec_type(dtype: str, n: int) -> str:
    gvec4_type = texel_type(dtype)
    return gvec4_type[:-1] + str(n)


def texel_component_type(dtype: str) -> str:
    vec4_type = texel_type(dtype)
    if vec4_type[:3] == "vec":
        return "float"
    elif vec4_type[:4] == "ivec":
        return "int"
    elif vec4_type[:4] == "uvec":
        return "uint"
    raise AssertionError(f"Invalid vec4 type: {vec4_type}")


def texel_load_type(dtype: str, storage_type: str) -> str:
    if storage_type.lower() == "buffer":
        return buffer_gvec_type(dtype, 4)
    else:
        return texel_type(dtype)


def texel_load_component_type(dtype: str, storage_type: str) -> str:
    if storage_type.lower() == "buffer":
        return buffer_scalar_type(dtype)
    else:
        return texel_component_type(dtype)


def get_access_qualifier(access_type: Optional[str]) -> str:
    if access_type is None:
        return ""
    if access_type.lower() == "r":
        return "readonly"
    if access_type.lower() == "w":
        return "writeonly"
    if access_type.lower() == "rw":
        return ""

    raise AssertionError(f"Invalid access type: {access_type}")


def get_slot_val(slot: Union[int, List[int]]) -> int:
    if isinstance(slot, list):
        return slot[0]
    return slot


def layout_declare_buffer(
    slot: Union[int, List[int]],
    access_type: str,
    var_name: str,
    dtype: str,
    precision: str = "PRECISION",
    is_scalar_array: bool = True,
) -> str:
    array_type = buffer_gvec_type(dtype, 4)
    if is_scalar_array:
        array_type = buffer_scalar_type(dtype)

    out_str = f"""
layout(set = 0, binding = {get_slot_val(slot)}) buffer {precision} restrict {get_access_qualifier(access_type)} {var_name}Buffer {{
    {array_type} {var_name}[];
}};
"""

    if isinstance(slot, list):
        slot[0] = slot[0] + 1
    return out_str


def layout_declare_image(
    slot: Union[int, List[int]],
    access_type: str,
    var_name: str,
    dtype: str,
    precision: str = "PRECISION",
    image_ndim: int = 3,
) -> str:
    image_format = TYPE_MAPPINGS["IMAGE_FORMAT"][dtype]
    image_type = TYPE_MAPPINGS["IMAGE_T"][image_ndim][dtype]

    ret_str = f"layout(set = 0, binding = {get_slot_val(slot)}, {image_format}) uniform {precision} restrict {get_access_qualifier(access_type)} {image_type} {var_name};"

    if isinstance(slot, list):
        slot[0] = slot[0] + 1
    return ret_str


def layout_declare_sampler(
    slot: Union[int, List[int]],
    access_type: str,
    var_name: str,
    dtype: str,
    precision: str = "PRECISION",
    access_qualifier: Optional[str] = None,
    image_ndim: int = 3,
) -> str:
    sampler_type = TYPE_MAPPINGS["SAMPLER_T"][image_ndim][dtype]

    ret_str = f"layout(set = 0, binding = {get_slot_val(slot)}) uniform {precision} {sampler_type} {var_name};"

    if isinstance(slot, list):
        slot[0] = slot[0] + 1
    return ret_str


def layout_declare_tensor(
    slot: Union[int, List[int]],
    access_type: str,
    var_name: str,
    dtype: str,
    storage_type: str,
    is_scalar_array: bool = True,
    precision: str = "PRECISION",
) -> str:
    assert storage_type.lower() in ["buffer", "texture3d", "texture2d"]

    image_ndim = 3
    if storage_type.lower() == "texture2d":
        image_ndim = 2

    # Create buffer binding
    if storage_type.lower() == "buffer":
        return layout_declare_buffer(
            slot,
            access_type,
            var_name,
            dtype,
            precision,
            is_scalar_array=is_scalar_array,
        )

    # Create image/sampler binding
    if access_type.lower() == "r":
        return layout_declare_sampler(
            slot, access_type, var_name, dtype, precision, image_ndim=image_ndim
        )
    else:
        return layout_declare_image(
            slot, access_type, var_name, dtype, precision, image_ndim=image_ndim
        )


def layout_declare_ubo(
    slot: Union[int, List[int]], *args, precision: str = "PRECISION"
) -> str:
    assert len(args) % 2 == 0

    var_list = list(zip(args[::2], args[1::2]))

    ubo_name = ""
    for _, var_name in var_list:
        ubo_name += var_name + "_"

    out_str = f"""
layout(set = 0, binding = {get_slot_val(slot)}) uniform {precision} restrict readonly {ubo_name}UBO {{
"""
    for type_name, var_name in var_list:
        out_str += f"  {type_name} {var_name};\n"
    out_str += "};"

    if isinstance(slot, list):
        slot[0] = slot[0] + 1
    return out_str


def layout_declare_spec_const(
    slot: Union[int, List[int]],
    type_name: str,
    var_name: str,
    initial_val: Optional[str] = None,
) -> str:
    assert type_name in ["int", "uint", "float", "bool"]

    out_str = f"layout(constant_id = {get_slot_val(slot)}) const {type_name} {var_name}"
    if initial_val is not None:
        out_str += f" = {initial_val}"
    out_str += ";"

    if isinstance(slot, list):
        slot[0] = slot[0] + 1
    return out_str


def define_active_storage_type(storage_type: str):
    if storage_type.lower() == "buffer":
        return "#define USING_BUFFER"
    elif storage_type.lower() == "texture3d":
        return "#define USING_TEXTURE3D"
    elif storage_type.lower() == "texture2d":
        return "#define USING_TEXTURE2D"
    else:
        raise AssertionError(f"Invalid storage type: {storage_type}")


def define_required_extensions(dtypes: Union[str, List[str]]):
    out_str = "\n"
    dtype_list = dtypes if isinstance(dtypes, list) else [dtypes]

    for dtype in dtype_list:
        nbit = None
        glsl_type = None
        if dtype == "half":
            nbit = "16bit"
            glsl_type = "float16"
        elif dtype == "double":
            # We only need to allow float64_t type usage
            glsl_type = "float64"
        elif dtype in ["int8", "uint8", "bool"]:
            nbit = "8bit"
            glsl_type = "int8"
        elif dtype in ["int16", "uint16"]:
            nbit = "16bit"
            glsl_type = "int16"
        elif dtype in ["int64", "uint64"]:
            # We only need to allow int64_t and uint64_t type usage
            glsl_type = "int64"

        if nbit is not None:
            out_str += f"#extension GL_EXT_shader_{nbit}_storage : require\n"
        if glsl_type is not None:
            out_str += f"#extension GL_EXT_shader_explicit_arithmetic_types_{glsl_type} : require\n"

    return out_str


UTILITY_FNS: Dict[str, Any] = {
    "macro_define": define_variable,
    "get_pos": {
        3: lambda pos: pos,
        2: lambda pos: f"{pos}.xy",
    },
    "buffer_scalar_type": buffer_scalar_type,
    "buffer_gvec_type": buffer_gvec_type,
    "texel_type": texel_type,
    "gvec_type": gvec_type,
    "texel_component_type": texel_component_type,
    "texel_load_type": texel_load_type,
    "texel_load_component_type": texel_load_component_type,
    "layout_declare_buffer": layout_declare_buffer,
    "layout_declare_image": layout_declare_image,
    "layout_declare_sampler": layout_declare_sampler,
    "layout_declare_tensor": layout_declare_tensor,
    "layout_declare_ubo": layout_declare_ubo,
    "layout_declare_spec_const": layout_declare_spec_const,
    "define_active_storage_type": define_active_storage_type,
    "define_required_extensions": define_required_extensions,
}


def extract_filename(path: str, keep_ext: bool = True) -> Any:
    if keep_ext:
        return os.path.basename(path)
    else:
        return os.path.basename(path).split(".")[0]


def extract_extension(path: str) -> str:
    return os.path.splitext(extract_filename(path))[1][1:]


############################
#  SPIR-V Code Generation  #
############################


# https://gist.github.com/pypt/94d747fe5180851196eb
class UniqueKeyLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            try:
                hash(key)
            except TypeError as e:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key ",
                    key_node.start_mark,
                ) from e
            # check for duplicate keys
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                )
            value = self.construct_object(value_node, deep=deep)  # type: ignore[no-untyped-call]
            mapping[key] = value
        return mapping


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def extract_leading_whitespace(line: str) -> str:
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def escape(line: str) -> str:
    output_parts = []
    while "${" in line:
        start_pos = line.index("${")
        end_pos = line.index("}", start_pos + 2)
        if start_pos != 0:
            output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
        output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
        line = line[end_pos + 1 :]
    if line:
        output_parts.append('"' + line.replace('"', '\\"') + '"')
    return " + ".join(output_parts)


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def preprocess(
    input_text: str, variables: Dict[str, Any], input_path: str = "codegen"
) -> str:
    # Workaround to handle source files using \ to extend mecros to a new line
    input_text = re.sub(r"\\$", r"\\\\", input_text, flags=re.MULTILINE)

    input_lines = input_text.splitlines()
    python_lines = []

    blank_lines = 0

    last_indent = ""

    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]

    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for input_line in input_lines:
        if input_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        if "LINT" in input_line:
            continue

        input_indent = extract_leading_whitespace(input_line)
        if python_block_start:
            assert input_indent.startswith(last_indent)
            extra_python_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            assert input_indent.startswith(indent_stack[-1][0])
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_input_line = input_line.strip()
        if stripped_input_line.startswith("$") and not stripped_input_line.startswith(
            "${"
        ):
            if stripped_input_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_input_line.replace("$", ""))
        else:
            assert input_line.startswith(python_indent)
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(
                python_indent
                + "print(%s, file=OUT_STREAM)"
                % escape(input_line[len(python_indent) :])
            )
        last_indent = input_indent

    while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(variables)
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream

    python_bytecode = compile("\n".join(python_lines), input_path, "exec")
    exec(python_bytecode, exec_globals)

    return output_stream.getvalue()


class SPVGenerator:
    def __init__(
        self,
        src_dir_paths: Union[str, List[str]],
        env: Dict[Any, Any],
        glslc_path: Optional[str],
        glslc_flags: str = "",
        replace_u16vecn: bool = False,
    ) -> None:
        if isinstance(src_dir_paths, str):
            self.src_dir_paths = [src_dir_paths]
        else:
            self.src_dir_paths = src_dir_paths

        self.env = env
        self.glslc_path = glslc_path
        self.glslc_flags = glslc_flags.split()
        self.glslc_flags_no_opt = self.glslc_flags.copy()
        if "-O" in self.glslc_flags_no_opt:
            self.glslc_flags_no_opt.remove("-O")
        if "-Os" in self.glslc_flags_no_opt:
            self.glslc_flags_no_opt.remove("-Os")
        self.replace_u16vecn = replace_u16vecn

        self.src_files: Dict[str, str] = {}
        self.template_yaml_files: List[str] = []

        self.addSrcAndYamlFiles(self.src_dir_paths)
        self.shader_template_params: Dict[Any, Any] = {}
        for yaml_file in self.template_yaml_files:
            self.parseTemplateYaml(yaml_file)

        self.output_file_map: Dict[str, Tuple[str, Dict[str, str]]] = {}
        self.constructOutputMap()

    def addSrcAndYamlFiles(self, src_dir_paths: List[str]) -> None:
        for src_path in src_dir_paths:
            # Collect glsl source files
            src_files_list = glob.glob(
                os.path.join(src_path, "**", "*.[gh]lsl*"), recursive=True
            ) + glob.glob(os.path.join(src_path, "**", "*.h"), recursive=True)
            for file in src_files_list:
                if len(file) > 1:
                    self.src_files[extract_filename(file, keep_ext=False)] = file
            # Collect template yaml files
            yaml_files = glob.glob(
                os.path.join(src_path, "**", "*.yaml"), recursive=True
            )
            for file in yaml_files:
                if len(file) > 1:
                    self.template_yaml_files.append(file)

    def generateVariantCombinations(  # noqa: C901
        self,
        iterated_params: Dict[str, Any],
        exclude_params: Optional[Set[str]] = None,
    ) -> List[Any]:
        if exclude_params is None:
            exclude_params = set()
        all_iterated_params = []
        for param_name, value_list in iterated_params.items():
            if re.match(r"^combination\d*$", param_name):
                param_values = []
                param_names = value_list["parameter_names"]
                combos = value_list["combos"]
                for combo in combos:
                    parameter_values = combo["parameter_values"]
                    if "suffix" in combo:
                        suffix = combo["suffix"]
                    else:
                        suffix = ""
                        for param_value in parameter_values:
                            if len(str(param_value)) > 0:
                                suffix += "_" + str(param_value)
                        suffix = suffix[1:]
                    param_values.append((param_names, suffix, parameter_values))

                all_iterated_params.append(param_values)

            elif param_name not in exclude_params:
                param_values = []
                for value in value_list:
                    if "RANGE" in value:
                        value_range = value["RANGE"]
                        suffix = value.get("SUFFIX", "")
                        if isinstance(value_range, list) and len(value_range) == 2:
                            for i in range(value_range[0], value_range[1] + 1):
                                curr_suffix = (
                                    suffix + "_" + str(i) if suffix else str(i)
                                )
                                param_values.append((param_name, curr_suffix, i))
                        else:
                            raise ValueError(
                                f"{value['RANGE']} is not a valid range. Must be in format [start, end] (inclusive)."
                            )

                    elif "VALUE" in value:
                        suffix = value.get("SUFFIX", value["VALUE"])
                        if value["VALUE"] in ["int", "uint"]:
                            raise ValueError(
                                f"Use int32 or uint32 instead of {value['VALUE']}"
                            )
                        param_values.append((param_name, suffix, value["VALUE"]))

                    else:
                        raise KeyError(
                            "Parameter must be 'VALUE: string' or 'RANGE: [a, b]'"
                        )

                all_iterated_params.append(param_values)

        return list(product(*all_iterated_params))

    def parseTemplateYaml(self, yaml_file: str) -> None:  # noqa: C901
        with open(yaml_file) as f:
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            for template_name, params_dict in contents.items():
                if template_name in self.shader_template_params:
                    raise KeyError(f"{template_name} params file is defined twice")

                default_params = params_dict["parameter_names_with_default_values"]
                default_params["YAML_SRC_FULLPATH"] = yaml_file
                params_names = set(default_params.keys()).union({"NAME"})

                self.shader_template_params[template_name] = []

                default_iterated_params = params_dict.get(
                    "generate_variant_forall", None
                )

                for variant in params_dict["shader_variants"]:
                    default_iterated_params_names = set(
                        default_iterated_params.keys()
                        if default_iterated_params is not None
                        else {}
                    )
                    variant_params_names = set(variant.keys())

                    invalid_keys = (
                        variant_params_names
                        - default_iterated_params_names
                        - params_names
                        - {"generate_variant_forall"}
                    )
                    assert len(invalid_keys) == 0

                    iterated_params = variant.get(
                        "generate_variant_forall", default_iterated_params
                    )

                    if iterated_params is not None:
                        variant_combinations = self.generateVariantCombinations(
                            iterated_params, variant_params_names
                        )

                        for combination in variant_combinations:
                            default_params_copy = copy.deepcopy(default_params)
                            for key in variant:
                                if key != "generate_variant_forall":
                                    default_params_copy[key] = variant[key]

                            variant_name = variant["NAME"]

                            for setting in combination:
                                param_names = setting[0]
                                suffix = setting[1]
                                param_values = setting[2]
                                if isinstance(param_names, list):
                                    for param_name, param_value in zip(
                                        param_names, param_values
                                    ):
                                        default_params_copy[param_name] = param_value
                                else:
                                    default_params_copy[param_names] = param_values

                                if len(str(suffix)) > 0:
                                    variant_name = f"{variant_name}_{suffix}"

                            default_params_copy["NAME"] = variant_name
                            default_params_copy["VARIANT_NAME"] = variant["NAME"]

                            self.shader_template_params[template_name].append(
                                default_params_copy
                            )
                    else:
                        default_params_copy = copy.deepcopy(default_params)
                        for key in variant:
                            default_params_copy[key] = variant[key]

                        self.shader_template_params[template_name].append(
                            default_params_copy
                        )

    def create_shader_params(
        self, variant_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        if variant_params is None:
            variant_params = {}
        shader_params = copy.deepcopy(self.env)
        for key, value in variant_params.items():
            shader_params[key] = value

        return shader_params

    def constructOutputMap(self) -> None:
        for src_name, params in self.shader_template_params.items():
            for variant in params:
                src_file_fullpath = self.src_files[src_name]

                self.output_file_map[variant["NAME"]] = (
                    src_file_fullpath,
                    self.create_shader_params(variant),
                )

        for src_name, src_file_fullpath in self.src_files.items():
            if src_name not in self.shader_template_params:
                self.output_file_map[src_name] = (
                    src_file_fullpath,
                    self.create_shader_params(),
                )

    def maybe_replace_u16vecn(self, input_text: str) -> str:
        """
        There is a latency benefit to using u16vecn variables to store texture position
        variables instead of ivecn, likely due to reduced register pressure. However,
        SwiftShader does not support 16 bit integer types in shaders, so this is a crude
        way to fallback to using ivecn to store texture positions so that testing with
        SwiftShader is still possible.
        """
        if not self.replace_u16vecn:
            return input_text
        if "codegen-nosub" in input_text:
            return input_text

        # Remove extension requirement so that generated ShaderInfo does not mark it
        input_text = input_text.replace(
            "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require", ""
        )
        input_text = input_text.replace("u16vec", "ivec")
        input_text = input_text.replace("uint16_t", "int")
        return input_text

    def get_md5_checksum(self, file_path: str) -> str:
        # Use a reasonably sized buffer for better performance with large files
        BUF_SIZE = 65536  # 64kb chunks

        md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)

        # Get the hexadecimal digest and compare
        file_md5 = md5.hexdigest()
        return file_md5

    def generateSPV(  # noqa: C901
        self,
        output_dir: str,
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, str]:
        # The key of this dictionary is the full path to a generated source file. The
        # value is a tuple that contains 3 entries:
        #
        # 1. A bool indicationg if the file has changed since the last compilation; this
        #    is determined by comparing against the cached version.
        # 2. List of other source files included by the generated file.
        gen_file_meta: Dict[str, Tuple[bool, List[str], str]] = {}

        # Return value of the function mapping the abspath of compiled SPIR-V binaries
        # to the abspath of the generated GLSL file they were compiled from.
        spv_to_glsl_map: Dict[str, str] = {}

        # Convert output_dir to absolute path
        assert os.path.exists(output_dir)
        output_dir = os.path.abspath(output_dir)

        if cache_dir is not None:
            assert os.path.exists(cache_dir)

        def get_glsl_includes(glsl_text):
            """
            Parse GLSL text content and return a list of included files.

            Args:
                glsl_text: String containing the GLSL file content to analyze

            Returns:
                List of included file names (e.g., ["random.h"])
            """
            includes = []
            for line in glsl_text.splitlines():
                # Look for #include directives with quoted filenames
                # Matches: #include "filename.h" or #include <filename.h>
                include_match = re.match(
                    r'^\s*#include\s+[<"]([^>"]+)[>"]', line.strip()
                )
                if include_match:
                    includes.append(include_match.group(1))

            return includes

        def file_has_changed(gen_file_path, cached_file_path):
            # If the file does not exist in the cache, then return True
            if not os.path.exists(cached_file_path):
                return True
            current_checksum = self.get_md5_checksum(gen_file_path)
            cached_checksum = self.get_md5_checksum(cached_file_path)
            return current_checksum != cached_checksum

        def any_sources_changed(gen_file_path, output_dir):
            """
            Given the path to a generated source file, check the gen_file_meta dict to
            determine if the ANY of the source files contributing to the compilation of
            this file were changed since the last successful compilation.
            """
            gen_file_changed, includes_list = gen_file_meta[gen_file_path]
            any_changed = gen_file_changed
            for included_file in includes_list:
                included_file_path = os.path.join(output_dir, included_file)
                any_changed = any_changed or any_sources_changed(
                    included_file_path, output_dir
                )

            return any_changed

        def generate_src_file(shader_paths_pair) -> Tuple[bool, List[str]]:
            """
            Given an input tuple containing the following items:
            (src_file_name, (template_file_path, codegen_params))

            This function generates src_file_name by processing
            template_file_path with the Python preprocessor using the
            parameters specified by codegen_params.

            Then, it returns a tuple containing:
            1. The path of the generated source file
            2. A bool indicating if the generated source file has changed since the last
               compilation.
            3. A list of files included by the generated source file
            """
            # name of .glsl, .glslh, or .h file to be generated
            src_file_name = shader_paths_pair[0]
            # path of template file used for codegen
            template_file_path = shader_paths_pair[1][0]
            # args to be used for codegen
            codegen_params = shader_paths_pair[1][1]

            # Assume that generated files will have the same file extension as the
            # source template file.
            out_file_ext = extract_extension(template_file_path)

            # Construct generated file name
            gen_out_path = os.path.join(output_dir, f"{src_file_name}.{out_file_ext}")
            # Construct path of cached generated file
            cached_gen_out_path = os.path.join(
                cache_dir, f"{src_file_name}.{out_file_ext}"
            )

            # Execute codegen to generate the output file
            with codecs.open(template_file_path, "r", encoding="utf-8") as input_file:
                input_text = input_file.read()
                input_text = self.maybe_replace_u16vecn(input_text)
                output_text = preprocess(input_text, codegen_params)

            included_files = get_glsl_includes(output_text)

            with codecs.open(gen_out_path, "w", encoding="utf-8") as output_file:
                output_file.write(output_text)

            file_changed = (
                file_has_changed(gen_out_path, cached_gen_out_path) or force_rebuild
            )

            # Save the generated file to cache so it can be used for future checks
            if cache_dir is not None and file_changed:
                shutil.copyfile(gen_out_path, cached_gen_out_path)

            return gen_out_path, file_changed, included_files

        def compile_spirv(shader_paths_pair) -> Tuple[str, str]:
            """
            Given an input tuple containing the following items:
            (src_file_name, (template_file_path, codegen_params))

            Infer the path of the GLSL source file generated by generate_src_file and
            compile a SPIR-V binary from it. Returns the path of the compiled SPIR-V
            binary and the path of the source file used to compile it.

            This function also utilizes a caching mechanism; if generate_src_file
            reported that the source file was unchanged since the last successful
            compilation, AND if the SPIR-V from the last successful compilation was
            stored in the cache, then directly use the cached SPIR-V without triggering
            a re-compilation.
            """
            # name of generated .glsl, .glslh, or .h from generate_src_file
            src_file_name = shader_paths_pair[0]
            # path of template file used for codegen
            template_file_path = shader_paths_pair[1][0]
            # args used for codegen
            codegen_params = shader_paths_pair[1][1]

            # Assume that generated files will have the same file extension as the
            # source template file.
            out_file_ext = extract_extension(template_file_path)

            # Infer name of generated file (created by generate_src_file)
            gen_out_path = os.path.join(output_dir, f"{src_file_name}.{out_file_ext}")

            # Only proceed if GLSL -> SPIR-V compilation is required for this file
            if out_file_ext != "glsl":
                return (None, gen_out_path)

            # Validate that the source file actually exists
            assert os.path.exists(gen_out_path) and gen_out_path in gen_file_meta

            # Construct name of SPIR-V file to be compiled
            spv_out_path = os.path.join(output_dir, f"{src_file_name}.spv")

            if cache_dir is not None:
                # Construct the file names of cached SPIR-V file to check if they exist
                # in the cache.
                cached_spv_out_path = os.path.join(cache_dir, f"{src_file_name}.spv")

                can_use_cached = not any_sources_changed(gen_out_path, output_dir)
                if can_use_cached and os.path.exists(cached_spv_out_path):
                    shutil.copyfile(cached_spv_out_path, spv_out_path)
                    return (spv_out_path, gen_out_path)

            vk_version = codegen_params.get("VK_VERSION", "1.1")
            # Only proceed if a GLSL compiler was specified
            if self.glslc_path is not None:
                cmd_base = [
                    self.glslc_path,
                    "-fshader-stage=compute",
                    gen_out_path,
                    "-o",
                    spv_out_path,
                    "--target-env=vulkan{}".format(vk_version),
                    "-Werror",
                    "-I",
                    output_dir,
                ]
                cmd = cmd_base + self.glslc_flags

                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    opt_fail = "compilation succeeded but failed to optimize"
                    err_msg_base = f"Failed to compile {os.getcwd()}/{gen_out_path}: "
                    if opt_fail in e.stderr or opt_fail in e.stdout:
                        cmd_no_opt = cmd_base + self.glslc_flags_no_opt
                        try:
                            subprocess.run(cmd_no_opt, check=True, capture_output=True)
                        except subprocess.CalledProcessError as e_no_opt:
                            # Delete any existing cached SPIR-V file if it exists
                            if os.path.exists(cached_spv_out_path):
                                os.remove(cached_spv_out_path)

                            raise RuntimeError(
                                f"{err_msg_base} {e_no_opt.stderr}"
                            ) from e_no_opt

                    else:
                        # Delete any existing cached SPIR-V file if it exists
                        if os.path.exists(cached_spv_out_path):
                            os.remove(cached_spv_out_path)

                        raise RuntimeError(f"{err_msg_base} {e.stderr}") from e

                # If compilation was successful, store the compiled SPIR-V file in the
                # cache for future use.
                if cache_dir is not None:
                    shutil.copyfile(spv_out_path, cached_spv_out_path)

            return (spv_out_path, gen_out_path)

        # Run codegen serially to ensure that all .glsl, .glslh, and .h files are up to
        # date before compilation
        for generated_file_tuple in self.output_file_map.items():
            gen_out_path, file_changed, include_list = generate_src_file(
                generated_file_tuple
            )
            gen_file_meta[gen_out_path] = (file_changed, include_list)

        # Parallelize SPIR-V compilation to optimize build time
        with ThreadPool(os.cpu_count()) as pool:
            for spv_out_path, glsl_out_path in pool.map(
                compile_spirv, self.output_file_map.items()
            ):
                spv_to_glsl_map[spv_out_path] = glsl_out_path

        return spv_to_glsl_map


##############################################
#  Shader Info and Shader Registry Handling  #
##############################################


@dataclass
class ShaderInfo:
    tile_size: List[int]
    layouts: List[str]
    weight_storage_type: str = ""
    bias_storage_type: str = ""
    register_for: Optional[Tuple[str, List[str]]] = None
    requires_shader_int16_ext: bool = False
    requires_16bit_storage_ext: bool = False
    requires_8bit_storage_ext: bool = False
    requires_integer_dot_product_ext: bool = False
    requires_shader_int64_ext: bool = False
    requires_shader_float64_ext: bool = False


def getName(filePath: str) -> str:
    return os.path.basename(filePath).replace("/", "_").replace(".", "_")


def isDescriptorLine(lineStr: str) -> bool:
    descriptorLineId = r"^layout\(set"
    return re.search(descriptorLineId, lineStr) is not None


def isTileSizeLine(lineStr: str) -> bool:
    tile_size_id = r"^ \* TILE_SIZE = \("
    return re.search(tile_size_id, lineStr) is not None


def findTileSizes(lineStr: str) -> List[int]:
    tile_size_id = r"^ \* TILE_SIZE = \(([0-9]+), ([0-9]+), ([0-9]+)\)"
    matches = re.search(tile_size_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in findTileSizes")
    return [int(matches.group(1)), int(matches.group(2)), int(matches.group(3))]


def isWeightStorageTypeLine(lineStr: str) -> bool:
    weight_storage_id = r"^ \* WEIGHT_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None


def getWeightStorageType(lineStr: str) -> str:
    weight_storage_id = r"^ \* WEIGHT_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getWeightStorageType")
    return matches.group(1)


def isBiasStorageTypeLine(lineStr: str) -> bool:
    weight_storage_id = r"^ \* BIAS_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None


def getBiasStorageType(lineStr: str) -> str:
    weight_storage_id = r"^ \* BIAS_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    return matches.group(1)


def isRegisterForLine(lineStr: str) -> bool:
    # Check for Shader Name and a list of at least one Registry Key
    register_for_id = (
        r"^ \* REGISTER_FOR = \('([A-Za-z0-9_]+)'\s*,\s*\['([A-Za-z0-9_]+)'.*\]\)"
    )
    return re.search(register_for_id, lineStr) is not None


def findRegisterFor(lineStr: str) -> Tuple[str, List[str]]:
    register_for_pattern = r"'([A-Za-z0-9_]+)'"
    matches = re.findall(register_for_pattern, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    matches_list = list(matches)
    return (matches_list[0], matches_list[1:])


def isExtensionRequireLine(lineStr: str) -> bool:
    extension_require_id = r"^#extension ([A-Za-z0-9_]+)\s*:\s*require"
    return re.search(extension_require_id, lineStr) is not None


typeIdMapping = {
    r"image[123]D\b": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    r"sampler[123]D\b": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    r"\bbuffer\b": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    r"\buniform\b": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
}


def determineDescriptorType(lineStr: str) -> str:
    for identifier, typeNum in typeIdMapping.items():
        if re.search(identifier, lineStr):
            return typeNum
    raise AssertionError(
        "No matching descriptor type for " + lineStr + " in determineDescriptorType"
    )


def getShaderInfo(srcFilePath: str) -> ShaderInfo:  # noqa: C901
    shader_info = ShaderInfo([], [], "")
    with open(srcFilePath) as srcFile:
        for line in srcFile:
            if isDescriptorLine(line):
                shader_info.layouts.append(determineDescriptorType(line))
            if isTileSizeLine(line):
                shader_info.tile_size = findTileSizes(line)
            if isWeightStorageTypeLine(line):
                shader_info.weight_storage_type = getWeightStorageType(line)
            if isBiasStorageTypeLine(line):
                shader_info.bias_storage_type = getBiasStorageType(line)
            if isRegisterForLine(line):
                shader_info.register_for = findRegisterFor(line)
            if isExtensionRequireLine(line):
                if "GL_EXT_shader_explicit_arithmetic_types_int16" in line:
                    shader_info.requires_shader_int16_ext = True
                if "GL_EXT_shader_16bit_storage" in line:
                    shader_info.requires_16bit_storage_ext = True
                if "GL_EXT_shader_8bit_storage" in line:
                    shader_info.requires_8bit_storage_ext = True
                if "GL_EXT_integer_dot_product" in line:
                    shader_info.requires_integer_dot_product_ext = True
                if "GL_EXT_shader_explicit_arithmetic_types_int64" in line:
                    shader_info.requires_shader_int64_ext = True
                if "GL_EXT_shader_explicit_arithmetic_types_float64" in line:
                    shader_info.requires_shader_float64_ext = True

    return shader_info


##########################
#  C++ File Generation  #
#########################

cpp_template = """
#include <executorch/backends/vulkan/runtime/api/ShaderRegistry.h>
#include <stdint.h>
#include <vector>

using namespace vkcompute;

namespace at {{
namespace native {{
namespace vulkan {{

namespace {{

{spv_bin_arrays}

}}

static void register_fn() {{

{register_shader_infos}

{shader_info_registry}

}}

static const api::ShaderRegisterInit register_shaders(&register_fn);

}}
}}
}}

"""


def generateSpvBinStr(spvPath: str, name: str) -> Tuple[int, str]:
    with open(spvPath, "rb") as fr:
        next_bin = array.array("I", fr.read())
        sizeBytes = 4 * len(next_bin)
        spv_bin_str = "const uint32_t {}_bin[] = {{\n{}\n}};".format(
            name,
            textwrap.indent(",\n".join(str(x) for x in next_bin), "  "),
        )

    return sizeBytes, spv_bin_str


def generateShaderInfoStr(shader_info: ShaderInfo, name: str, sizeBytes: int) -> str:
    tile_size = (
        f"{{{', '.join(str(x) for x in shader_info.tile_size)}}}"
        if (len(shader_info.tile_size) > 0)
        else "{1, 1, 1}"
    )

    shader_info_layouts = "{{{}}}".format(",\n ".join(shader_info.layouts))

    def to_cpp_str(val: bool):
        return "true" if val else "false"

    shader_info_args = [
        f'"{name}"',
        f"{name}_bin",
        str(sizeBytes),
        shader_info_layouts,
        tile_size,
        to_cpp_str(shader_info.requires_shader_int16_ext),
        to_cpp_str(shader_info.requires_16bit_storage_ext),
        to_cpp_str(shader_info.requires_8bit_storage_ext),
        to_cpp_str(shader_info.requires_integer_dot_product_ext),
        to_cpp_str(shader_info.requires_shader_int64_ext),
        to_cpp_str(shader_info.requires_shader_float64_ext),
    ]

    shader_info_str = textwrap.indent(
        "api::shader_registry().register_shader(\n  vkapi::ShaderInfo(\n{args}));\n".format(
            args=textwrap.indent(",\n".join(shader_info_args), "     "),
        ),
        "    ",
    )

    return shader_info_str


def generateShaderDispatchStr(shader_info: ShaderInfo, name: str) -> str:
    if shader_info.register_for is None:
        return ""

    (op_name, registry_keys) = shader_info.register_for
    shader_dispatch_str = ""
    for registry_key in registry_keys:
        shader_dispatch_str = textwrap.indent(
            f'api::shader_registry().register_op_dispatch("{op_name}", api::DispatchKey::{registry_key.upper()}, "{name}");',
            "    ",
        )

    return shader_dispatch_str


def genCppFiles(
    spv_files: Dict[str, str], cpp_header_path: str, cpp_src_file_path: str
) -> None:
    spv_bin_strs = []
    register_shader_info_strs = []
    shader_registry_strs = []

    for spvPath, srcPath in spv_files.items():
        if spvPath is None:
            continue

        name = getName(spvPath).replace("_spv", "")

        sizeBytes, spv_bin_str = generateSpvBinStr(spvPath, name)
        spv_bin_strs.append(spv_bin_str)

        shader_info = getShaderInfo(srcPath)

        register_shader_info_strs.append(
            generateShaderInfoStr(shader_info, name, sizeBytes)
        )

        if shader_info.register_for is not None:
            shader_registry_strs.append(generateShaderDispatchStr(shader_info, name))

    spv_bin_arrays = "\n".join(spv_bin_strs)
    register_shader_infos = "\n".join(register_shader_info_strs)
    shader_info_registry = "\n".join(shader_registry_strs)

    cpp = cpp_template.format(
        spv_bin_arrays=spv_bin_arrays,
        register_shader_infos=register_shader_infos,
        shader_info_registry=shader_info_registry,
    )

    with open(cpp_src_file_path, "w") as fw:
        fw.write(cpp)


##########
#  Main  #
##########


def parse_arg_env(items: Dict[Any, Any]) -> Dict[Any, Any]:
    d = {}
    if items:
        for item in items:
            tokens = item.split("=")
            key = tokens[0].strip()
            value = tokens[1].strip()
            d[key] = value
    return d


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--glsl-paths",
        nargs="+",
        help='List of paths to look for GLSL source files, separated by spaces. Ex: --glsl-paths "path1 path2 path3"',
        default=["."],
    )
    parser.add_argument("-c", "--glslc-path", required=True, help="")
    parser.add_argument(
        "-t", "--tmp-dir-path", required=True, help="/tmp/vulkan_shaders/"
    )
    parser.add_argument("-o", "--output-path", required=True, help="")
    parser.add_argument("-f", "--force-rebuild", action="store_true", default=False)
    parser.add_argument("--replace-u16vecn", action="store_true", default=False)
    parser.add_argument("--optimize_size", action="store_true", help="")
    parser.add_argument("--optimize", action="store_true", help="")
    parser.add_argument("--spv_debug", action="store_true", default=False)
    parser.add_argument(
        "--env", metavar="KEY=VALUE", nargs="*", help="Set a number of key-value pairs"
    )
    options = parser.parse_args()

    env = DEFAULT_ENV
    env.update(TYPE_MAPPINGS)
    env.update(UTILITY_FNS)

    for key, value in parse_arg_env(options.env).items():
        env[key] = value

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    if not os.path.exists(options.tmp_dir_path):
        os.makedirs(options.tmp_dir_path)

    glslc_flags = []
    if options.optimize_size:
        glslc_flags.append("-Os")
    elif options.optimize:
        glslc_flags.append("-O")

    if options.spv_debug:
        glslc_flags.append("-g")

    glslc_flags_str = " ".join(glslc_flags)

    shader_generator = SPVGenerator(
        options.glsl_paths,
        env,
        options.glslc_path,
        glslc_flags=glslc_flags_str,
        replace_u16vecn=options.replace_u16vecn,
    )
    output_spv_files = shader_generator.generateSPV(
        options.output_path, options.tmp_dir_path, options.force_rebuild
    )

    genCppFiles(
        output_spv_files,
        f"{options.output_path}/{CPP_H_NAME}",
        f"{options.output_path}/{CPP_SRC_NAME}",
    )

    return 0


def invoke_main() -> None:
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
