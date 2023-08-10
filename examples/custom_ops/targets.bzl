load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.export_file(
        name = "custom_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # ~~~ START of custom ops 1 `my_ops::mul3` library definitions ~~~
    et_operator_library(
        name = "select_custom_ops_1",
        ops = [
            "my_ops::mul3.out",
        ],
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "custom_ops_1",
        srcs = ["custom_ops_1_out.cpp"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    executorch_generated_lib(
        name = "lib_1",
        deps = [
            ":select_custom_ops_1",
            ":custom_ops_1",
        ],
        custom_ops_yaml_target = ":custom_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # ~~~ END of custom ops 1 `my_ops::mul3` library definitions ~~~
