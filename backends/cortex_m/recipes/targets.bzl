# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    python_library(
        name = "recipes",
        srcs = [
            "__init__.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":cortex_m_recipe_provider",
            ":cortex_m_recipe_types",
            "//executorch/export:recipe_registry",
        ],
    )

    python_library(
        name = "cortex_m_recipe_provider",
        srcs = [
            "cortex_m_recipe_provider.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":cortex_m_recipe_types",
            "//caffe2:torch",
            "//executorch/backends/cortex_m:cmsis_nn",
            "//executorch/backends/cortex_m:edge_compile_config",
            "//executorch/backends/cortex_m:op_backend",
            "//executorch/backends/cortex_m:target_config",
            "//executorch/backends/cortex_m/quantizer:quantizer",
            "//executorch/export:lib",
        ],
    )

    python_library(
        name = "cortex_m_recipe_types",
        srcs = [
            "cortex_m_recipe_types.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/export:recipe",
        ],
    )
