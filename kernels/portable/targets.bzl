load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "operators",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/portable/cpu:cpu",
            # "//executorch/kernels/portable/cpu:edge_dialect_ops",
        ],
    )

    runtime.cxx_library(
        name = "operators_aten",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/portable/cpu:cpu_aten",
            # "//executorch/kernels/portable/cpu:edge_dialect_ops_aten",
        ],
    )

    runtime.cxx_library(
        name = "edge_dialect_operators",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/portable/cpu:edge_dialect_ops",
        ],
    )

    runtime.cxx_library(
        name = "edge_dialect_operators_aten",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/portable/cpu:edge_dialect_ops_aten",
        ],
    )

    runtime.export_file(
        name = "functions.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.export_file(
        name = "custom_ops.yaml",
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "executorch_all_ops",
        include_all_operators = True,
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.export_file(
        name = "edge_dialect_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "executorch_aten_ops",
        ops_schema_yaml_target = "//executorch/kernels/portable:functions.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "executorch_custom_ops",
        ops_schema_yaml_target = "//executorch/kernels/portable:custom_ops.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "executorch_edge_dialect_ops",
        ops_schema_yaml_target = "//executorch/kernels/portable:edge_dialect_ops.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    generated_lib_common_args = {
        "custom_ops_aten_kernel_deps": [
            "//executorch/kernels/portable:operators_aten",
        ],
        "custom_ops_yaml_target": "//executorch/kernels/portable:custom_ops.yaml",
        # size_test expects _static targets to be available for these libraries.
        "define_static_targets": True,
        "functions_yaml_target": "//executorch/kernels/portable:functions.yaml",
        "visibility": [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    }

    executorch_generated_lib(
        name = "generated_lib",
        deps = [
            ":executorch_aten_ops",
            ":executorch_custom_ops",
            ":executorch_edge_dialect_ops",
        ],
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        **generated_lib_common_args
    )

    executorch_generated_lib(
        name = "generated_lib_aten",
        deps = [
            ":executorch_aten_ops",
            ":executorch_custom_ops",
            "//executorch/kernels/portable:operators_aten",
        ],
        custom_ops_aten_kernel_deps = [
            "//executorch/kernels/portable:operators_aten",
        ],
        custom_ops_yaml_target = "//executorch/kernels/portable:custom_ops.yaml",
        aten_mode = True,
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        define_static_targets = True,
    )

    executorch_generated_lib(
        name = "generated_lib_edge_dialect_ops",
        deps = [
            ":executorch_edge_dialect_ops",
            ":edge_dialect_operators",
        ],
        custom_ops_yaml_target = "//executorch/kernels/portable:edge_dialect_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        define_static_targets = True,
    )

    executorch_generated_lib(
        name = "generated_lib_edge_dialect_ops_aten",
        deps = [
            ":executorch_edge_dialect_ops",
            ":edge_dialect_operators_aten",
        ],
        custom_ops_aten_kernel_deps = [
            ":edge_dialect_operators_aten",
        ],
        custom_ops_yaml_target = "//executorch/kernels/portable:edge_dialect_ops.yaml",
        aten_mode = True,
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        define_static_targets = True,
    )
