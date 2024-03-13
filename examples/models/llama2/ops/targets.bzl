load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "quantized_aot_lib",
        srcs = [
            "quantized_ops.py",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.export_file(
        name = "quantized.yaml",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "all_quantized_ops",
        define_static_targets = True,
        ops_schema_yaml_target = ":quantized.yaml",
    )

    executorch_generated_lib(
        name = "generated_lib",
        custom_ops_yaml_target = ":quantized.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            ":all_quantized_ops",
            "//executorch/kernels/quantized:quantized_operators",
        ],
    )
