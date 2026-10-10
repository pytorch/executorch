load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/backends/vulkan:targets.bzl",
    "get_platforms",
    "vulkan_spv_shader_lib",
)

def define_custom_op_test_binary(custom_op_name, subdir, extra_deps = [], include_torch = False):
    deps_list = [
        ":prototyping_utils",
        ":operator_implementations",
        ":custom_ops_shaderlib",
        "//executorch/backends/vulkan:vulkan_graph_runtime",
    ] + ([runtime.external_dep_location("libtorch")] if include_torch else []) + extra_deps

    runtime.cxx_binary(
        name = custom_op_name,
        srcs = [
            "{}/{}.cpp".format(subdir, custom_op_name),
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
            "test_ops/glsl/**/*",
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
            "framework/config.cpp",
            "framework/conv2d_utils.cpp",
            "framework/results.cpp",
            "framework/runner.cpp",
            "framework/test_case.cpp",
            "framework/value_spec.cpp",
            "framework/weight_utils.cpp",
        ],
        headers = [
            "framework/config.h",
            "framework/conv2d_utils.h",
            "framework/labels.h",
            "framework/results.h",
            "framework/runner.h",
            "framework/test_case.h",
            "framework/utils.h",
            "framework/value_spec.h",
            "framework/weight_utils.h",
        ],
        exported_headers = [
            "framework/config.h",
            "framework/conv2d_utils.h",
            "framework/labels.h",
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




    define_custom_op_test_binary("test_add", "cases/add")
    define_custom_op_test_binary("test_q8csw_linear", "cases/q8csw_linear")
    define_custom_op_test_binary("test_q8csw_conv2d", "cases/q8csw_conv2d")
    define_custom_op_test_binary("test_choose_qparams_per_row", "cases/choose_qparams_per_row")
    define_custom_op_test_binary("test_q4gsw_linear", "cases/q4gsw_linear")
    define_custom_op_test_binary("test_q8ta_qdq", "cases/q8ta/qdq")
    define_custom_op_test_binary("test_q8ta_clone", "cases/q8ta/clone")
    define_custom_op_test_binary("test_q8ta_binary", "cases/q8ta/binary")
    define_custom_op_test_binary("test_q8ta_conv2d", "cases/q8ta/conv2d")
    define_custom_op_test_binary("test_q8ta_conv2d_pw", "cases/q8ta/conv2d")
    define_custom_op_test_binary("test_q8ta_conv2d_dw", "cases/q8ta/conv2d")
    define_custom_op_test_binary("test_q8ta_linear", "cases/q8ta/linear")
    define_custom_op_test_binary("test_q8ta_conv2d_transposed", "cases/q8ta/conv2d_transposed")
    define_custom_op_test_binary("test_q8ta_pixel_shuffle", "cases/q8ta/pixel_shuffle")
    define_custom_op_test_binary("test_q8ta_unary", "cases/q8ta/unary")
    define_custom_op_test_binary("test_mm", "cases/mm")
    define_custom_op_test_binary("test_conv2d", "cases/conv2d")
    define_custom_op_test_binary("test_conv2d_pw", "cases/conv2d")
    define_custom_op_test_binary("test_conv2d_dw", "cases/conv2d")
    define_custom_op_test_binary("test_embedding_q4gsw", "cases/embedding_q4gsw")
    define_custom_op_test_binary("test_conv1d_pw", "cases/conv1d")
    define_custom_op_test_binary("test_conv1d_dw", "cases/conv1d")
    define_custom_op_test_binary("test_fpa_q4gsw_linear", "cases/q4gsw_linear")
    define_custom_op_test_binary("test_sdpa", "cases/sdpa")
