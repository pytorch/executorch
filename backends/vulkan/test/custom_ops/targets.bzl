load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/backends/vulkan:targets.bzl",
    "get_platforms",
    "vulkan_spv_shader_lib",
)

def define_custom_op_test_binary(custom_op_name, extra_deps = [], src_file = None):
    deps_list = [
        ":prototyping_utils",
        ":operator_implementations",
        ":custom_ops_shaderlib",
        "//executorch/backends/vulkan:vulkan_graph_runtime",
        runtime.external_dep_location("libtorch"),
    ] + extra_deps

    src_file_str = src_file if src_file else "{}.cpp".format(custom_op_name)

    runtime.cxx_binary(
        name = custom_op_name,
        srcs = [
            src_file_str,
        ],
        platforms = get_platforms(),
        define_static_target = False,
        deps = deps_list,
    )

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    # Shader library from GLSL files
    runtime.filegroup(
        name = "custom_ops_shaders",
        srcs = native.glob([
            "glsl/*.glsl",
            "glsl/*.yaml",
        ]),
        visibility = [
            "//executorch/backends/vulkan/test/custom_ops/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    vulkan_spv_shader_lib(
        name = "custom_ops_shaderlib",
        spv_filegroups = {
            ":custom_ops_shaders": "glsl",
        },
        is_fbcode = is_fbcode,
    )

    # Prototyping utilities library
    runtime.cxx_library(
        name = "prototyping_utils",
        srcs = [
            "utils.cpp",
        ],
        headers = [
            "utils.h",
        ],
        exported_headers = [
            "utils.h",
        ],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan:vulkan_graph_runtime",
        ],
        visibility = [
            "//executorch/backends/vulkan/test/custom_ops/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Operator implementations library
    runtime.cxx_library(
        name = "operator_implementations",
        srcs = native.glob([
            "impl/*.cpp",
        ]),
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            ":custom_ops_shaderlib",
        ],
        visibility = [
            "//executorch/backends/vulkan/test/custom_ops/...",
            "@EXECUTORCH_CLIENTS",
        ],
        link_whole = True,
    )

    define_custom_op_test_binary("add")
    define_custom_op_test_binary("q8csw_linear")
    define_custom_op_test_binary("q8csw_conv2d")
    define_custom_op_test_binary("choose_qparams_per_row")
    define_custom_op_test_binary("q4gsw_linear")
    define_custom_op_test_binary("qdq8ta_conv2d_activations")
