load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/backends/vulkan:targets.bzl",
    "get_platforms",
    "vulkan_spv_shader_lib",
)

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    # Shader library from GLSL files
    runtime.filegroup(
        name = "custom_ops_shaders",
        srcs = native.glob([
            "test_ops/glsl/*.glsl",
            "test_ops/glsl/*.yaml",
        ]),
        visibility = ["PUBLIC"],
    )

    vulkan_spv_shader_lib(
        name = "custom_ops_shaderlib",
        spv_filegroups = {
            ":custom_ops_shaders": "test_ops/glsl",
        },
        is_fbcode = is_fbcode,
    )

    # Prototyping utilities library
    runtime.cxx_library(
        name = "prototyping_utils",
        srcs = [
            "framework/cm_utils.cpp",
            "framework/config.cpp",
            "framework/conv2d_utils.cpp",
            "framework/device_info.cpp",
            "framework/registry.cpp",
            "framework/results.cpp",
            "framework/runner.cpp",
            "framework/test_case.cpp",
            "framework/value_spec.cpp",
            "framework/weight_utils.cpp",
        ],
        headers = [
            "framework/cm_utils.h",
            "framework/config.h",
            "framework/conv2d_utils.h",
            "framework/device_info.h",
            "framework/labels.h",
            "framework/registry.h",
            "framework/results.h",
            "framework/runner.h",
            "framework/test_case.h",
            "framework/utils.h",
            "framework/value_spec.h",
            "framework/weight_utils.h",
        ],
        exported_headers = [
            "framework/cm_utils.h",
            "framework/config.h",
            "framework/conv2d_utils.h",
            "framework/device_info.h",
            "framework/labels.h",
            "framework/registry.h",
            "framework/results.h",
            "framework/runner.h",
            "framework/test_case.h",
            "framework/utils.h",
            "framework/value_spec.h",
            "framework/weight_utils.h",
        ],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan:vulkan_graph_runtime",
        ],
        visibility = ["PUBLIC"],
    )

    # Operator implementations library
    runtime.cxx_library(
        name = "operator_implementations",
        srcs = native.glob([
            "test_ops/impl/*.cpp",
        ]),
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            ":custom_ops_shaderlib",
        ],
        visibility = ["PUBLIC"],
        link_whole = True,
    )

    runtime.cxx_binary(
        name = "etvk_bench",
        srcs = ["etvk_bench.cpp"] + native.glob(["cases/**/*.cpp"]),
        headers = native.glob(["cases/**/*.h"]),
        platforms = get_platforms(),
        define_static_target = False,
        deps = [
            ":custom_ops_shaderlib",
            ":operator_implementations",
            ":prototyping_utils",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
        ],
        external_deps = [
            "gflags",
        ],
    )
