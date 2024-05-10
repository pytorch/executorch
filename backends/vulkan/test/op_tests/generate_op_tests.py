# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from typing import Dict

from executorch.backends.vulkan.test.op_tests.cases import test_suites

from executorch.backends.vulkan.test.op_tests.utils.codegen import VkCppTestFileGen
from executorch.backends.vulkan.test.op_tests.utils.codegen_base import (
    TestSuite,
    TestSuiteGen,
)
from torchgen import local

from torchgen.gen import parse_native_yaml, ParsedYaml
from torchgen.model import DispatchKey, NativeFunction


def registry_name(f: NativeFunction) -> str:
    name = str(f.namespace) + "." + str(f.func.name)
    if len(f.func.name.overload_name) == 0:
        name += ".default"
    return name


def construct_f_map(parsed_yaml: ParsedYaml) -> Dict[str, NativeFunction]:
    f_map: Dict[str, NativeFunction] = {}
    for f in parsed_yaml.native_functions:
        f_map[registry_name(f)] = f
    return f_map


def process_test_suites(
    cpp_generator: VkCppTestFileGen,
    f_map: Dict[str, NativeFunction],
    test_suites: Dict[str, TestSuite],
) -> None:
    for registry_name, op_test_suite in test_suites.items():
        f = f_map[registry_name]
        cpp_generator.add_suite(registry_name, f, op_test_suite)


@local.parametrize(
    use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
)
def generate_cpp(
    native_functions_yaml_path: str, tags_path: str, output_dir: str
) -> None:
    output_file = os.path.join(output_dir, "op_tests.cpp")
    cpp_generator = VkCppTestFileGen(output_file)

    parsed_yaml = parse_native_yaml(native_functions_yaml_path, tags_path)
    f_map = construct_f_map(parsed_yaml)

    TestSuiteGen.backend_key = parsed_yaml.backend_indices[DispatchKey.CPU]

    process_test_suites(cpp_generator, f_map, test_suites)

    with open(output_file, "w") as file:
        file.write(cpp_generator.generate_cpp())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple Hello World C++ program."
    )
    parser.add_argument(
        "--aten-yaml-path",
        help="path to native_functions.yaml file.",
    )
    parser.add_argument(
        "--tags-path",
        help="Path to tags.yaml. Required by yaml parsing in codegen system.",
    )
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    args = parser.parse_args()
    generate_cpp(args.aten_yaml_path, args.tags_path, args.output)
