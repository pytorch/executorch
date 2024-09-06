#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import array
import codecs
import copy
import glob
import io
import os
import re
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
}

# Establishes relationships between different tensor types and different GLSL types
TYPE_MAPPINGS: Dict[str, Any] = {
    "IMAGE_T": {
        3: {
            "float": "image3D",
            "half": "image3D",
            "int": "iimage3D",
            "uint": "uimage3D",
            "int8": "iimage3D",
            "uint8": "uimage3D",
        },
        2: {
            "float": "image2D",
            "half": "image2D",
            "int": "iimage2D",
            "uint": "uimage2D",
            "int8": "iimage2D",
            "uint8": "uimage2D",
        },
    },
    "SAMPLER_T": {
        3: {
            "float": "sampler3D",
            "half": "sampler3D",
            "int": "isampler3D",
            "uint": "usampler3D",
            "int8": "isampler3D",
            "uint8": "usampler3D",
        },
        2: {
            "float": "sampler2D",
            "half": "sampler2D",
            "int": "isampler2D",
            "uint": "usampler2D",
            "int8": "isampler2D",
            "uint8": "usampler2D",
        },
    },
    "IMAGE_FORMAT": {
        "float": "rgba32f",
        "half": "rgba16f",
        "int": "rgba32i",
        "uint": "rgba32ui",
        "int8": "rgba8i",
        "uint8": "rgba8ui",
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
    elif dtype[-1] == "8":
        return dtype + "_t"

    return dtype


def buffer_gvec_type(dtype: str, n: int) -> str:
    if n == 1:
        return buffer_scalar_type(dtype)

    if dtype == "float":
        return f"vec{n}"
    elif dtype == "half":
        return f"f16vec{n}"
    elif dtype == "int":
        return f"ivec{n}"
    elif dtype == "int8":
        return f"i8vec{n}"
    elif dtype == "uint8":
        return f"u8vec{n}"

    raise AssertionError(f"Invalid dtype: {dtype}")


def texel_type(dtype: str) -> str:
    image_format = TYPE_MAPPINGS["IMAGE_FORMAT"][dtype]
    if image_format[-1] == "f":
        return "vec4"
    elif image_format[-2] == "ui":
        return "uvec4"
    elif image_format[-1] == "i":
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
        out_str += f"{type_name} {var_name};\n"
    out_str += "};"

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


def define_required_extensions(dtype: str):
    out_str = "\n"
    nbit = None
    glsl_type = None

    if dtype == "half":
        nbit = "16bit"
        glsl_type = "float16"
    if dtype == "int8":
        nbit = "8bit"
        glsl_type = "int8"

    if nbit is not None and glsl_type is not None:
        out_str += f"#extension GL_EXT_shader_{nbit}_storage : require\n"
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
    "define_active_storage_type": define_active_storage_type,
    "define_required_extensions": define_required_extensions,
}


def extract_filename(path: str, keep_ext: bool = True) -> Any:
    if keep_ext:
        return os.path.basename(path)
    else:
        return os.path.basename(path).split(".")[0]


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
    ) -> None:
        if isinstance(src_dir_paths, str):
            self.src_dir_paths = [src_dir_paths]
        else:
            self.src_dir_paths = src_dir_paths

        self.env = env
        self.glslc_path = glslc_path
        self.glslc_flags = glslc_flags

        self.glsl_src_files: Dict[str, str] = {}
        self.template_yaml_files: List[str] = []

        self.addSrcAndYamlFiles(self.src_dir_paths)
        self.shader_template_params: Dict[Any, Any] = {}
        for yaml_file in self.template_yaml_files:
            self.parseTemplateYaml(yaml_file)

        self.output_shader_map: Dict[str, Tuple[str, Dict[str, str]]] = {}
        self.constructOutputMap()

    def addSrcAndYamlFiles(self, src_dir_paths: List[str]) -> None:
        for src_path in src_dir_paths:
            # Collect glsl source files
            glsl_files = glob.glob(
                os.path.join(src_path, "**", "*.glsl*"), recursive=True
            )
            for file in glsl_files:
                if len(file) > 1:
                    self.glsl_src_files[extract_filename(file, keep_ext=False)] = file
            # Collect template yaml files
            yaml_files = glob.glob(
                os.path.join(src_path, "**", "*.yaml"), recursive=True
            )
            for file in yaml_files:
                if len(file) > 1:
                    self.template_yaml_files.append(file)

    def generateVariantCombinations(
        self,
        iterated_params: Dict[str, Any],
        exclude_params: Optional[Set[str]] = None,
    ) -> List[Any]:
        if exclude_params is None:
            exclude_params = set()
        all_iterated_params = []
        for param_name, value_list in iterated_params.items():
            if param_name not in exclude_params:
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
                        param_values.append((param_name, suffix, value["VALUE"]))

                    else:
                        raise KeyError(
                            "Parameter must be 'VALUE: string' or 'RANGE: [a, b]'"
                        )

                all_iterated_params.append(param_values)

        return list(product(*all_iterated_params))

    def parseTemplateYaml(self, yaml_file: str) -> None:
        with open(yaml_file) as f:
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            for template_name, params_dict in contents.items():
                if template_name in self.shader_template_params:
                    raise KeyError(f"{template_name} params file is defined twice")

                default_params = params_dict["parameter_names_with_default_values"]
                params_names = set(default_params.keys()).union({"NAME"})

                self.shader_template_params[template_name] = []

                default_iterated_params = params_dict.get(
                    "generate_variant_forall", None
                )

                for variant in params_dict["shader_variants"]:
                    variant_params_names = set(variant.keys())
                    invalid_keys = (
                        variant_params_names
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
                            for param_value in combination:
                                default_params_copy[param_value[0]] = param_value[2]
                                if len(str(param_value[1])) > 0:
                                    variant_name = f"{variant_name}_{param_value[1]}"

                            default_params_copy["NAME"] = variant_name

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
        for shader_name, params in self.shader_template_params.items():
            for variant in params:
                source_glsl = self.glsl_src_files[shader_name]

                self.output_shader_map[variant["NAME"]] = (
                    source_glsl,
                    self.create_shader_params(variant),
                )

        for shader_name, source_glsl in self.glsl_src_files.items():
            if shader_name not in self.shader_template_params:
                self.output_shader_map[shader_name] = (
                    source_glsl,
                    self.create_shader_params(),
                )

    def generateSPV(self, output_dir: str) -> Dict[str, str]:
        output_file_map = {}

        def process_shader(shader_paths_pair):
            shader_name = shader_paths_pair[0]

            source_glsl = shader_paths_pair[1][0]
            shader_params = shader_paths_pair[1][1]

            with codecs.open(source_glsl, "r", encoding="utf-8") as input_file:
                input_text = input_file.read()
                output_text = preprocess(input_text, shader_params)

            glsl_out_path = os.path.join(output_dir, f"{shader_name}.glsl")
            with codecs.open(glsl_out_path, "w", encoding="utf-8") as output_file:
                output_file.write(output_text)

            # If no GLSL compiler is specified, then only write out the generated GLSL shaders.
            # This is mainly for testing purposes.
            if self.glslc_path is not None:
                spv_out_path = os.path.join(output_dir, f"{shader_name}.spv")

                cmd = (
                    [
                        self.glslc_path,
                        "-fshader-stage=compute",
                        glsl_out_path,
                        "-o",
                        spv_out_path,
                        "--target-env=vulkan1.1",
                        "-Werror",
                    ]
                    + [
                        arg
                        for src_dir_path in self.src_dir_paths
                        for arg in ["-I", src_dir_path]
                    ]
                    + self.glslc_flags.split()
                )

                subprocess.check_call(cmd)

                return (spv_out_path, glsl_out_path)

        # Parallelize shader compilation as much as possible to optimize build time.
        with ThreadPool(os.cpu_count()) as pool:
            for spv_out_path, glsl_out_path in pool.map(
                process_shader, self.output_shader_map.items()
            ):
                output_file_map[spv_out_path] = glsl_out_path

        return output_file_map


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


def getShaderInfo(srcFilePath: str) -> ShaderInfo:
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

    shader_info_args = [
        f'"{name}"',
        f"{name}_bin",
        str(sizeBytes),
        shader_info_layouts,
        tile_size,
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
    parser.add_argument("-t", "--tmp-dir-path", required=True, help="/tmp")
    parser.add_argument("-o", "--output-path", required=True, help="")
    parser.add_argument("--optimize_size", action="store_true", help="")
    parser.add_argument("--optimize", action="store_true", help="")
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

    glslc_flags = ""
    if options.optimize_size:
        glslc_flags += "-Os"
    elif options.optimize:
        glslc_flags += "-O"

    shader_generator = SPVGenerator(
        options.glsl_paths, env, options.glslc_path, glslc_flags
    )
    output_spv_files = shader_generator.generateSPV(options.tmp_dir_path)

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
