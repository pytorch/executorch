load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    runtime.export_file(
        name = "ops.yaml",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "torchvision_ops",
        ops_schema_yaml_target = ":ops.yaml",
        define_static_targets = True,
    )

    runtime.cxx_library(
        name = "torchvision_operators",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/torchvision/cpu:torchvision_cpu",
        ],
    )

    executorch_generated_lib(
        name = "generated_lib",
        deps = [
            ":torchvision_ops",
            ":torchvision_operators",
        ],
        custom_ops_yaml_target = ":ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        define_static_targets = True,
    )
