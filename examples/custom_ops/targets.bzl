load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib", "exir_custom_ops_aot_lib")

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
    # ~~~ START of custom ops 2 `my_ops::mul4` library definitions ~~~

    et_operator_library(
        name = "select_custom_ops_2",
        ops = [
            "my_ops::mul4.out",
        ],
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "custom_ops_2",
        srcs = ["custom_ops_2_out.cpp"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "custom_ops_2_aten",
        srcs = [
            "custom_ops_2.cpp",
            "custom_ops_2_out.cpp",
        ],
        deps = [
            "//executorch/runtime/kernel:kernel_includes_aten",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        external_deps = ["libtorch"],
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        # WARNING: using a deprecated API to avoid being built into a shared
        # library. In the case of dynamically loading so library we don't want
        # it to depend on other so libraries because that way we have to
        # specify library directory path.
        force_static = True,
    )

    exir_custom_ops_aot_lib(
        name = "custom_ops_aot_lib_2",
        yaml_target = ":custom_ops.yaml",
        visibility = ["//executorch/..."],
        kernels = [":custom_ops_2_aten"],
        deps = [
            ":select_custom_ops_2",
        ],
    )

    executorch_generated_lib(
        name = "lib_2",
        deps = [
            ":select_custom_ops_2",
            ":custom_ops_2",
        ],
        custom_ops_yaml_target = ":custom_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
    # ~~~ END of custom ops 2 `my_ops::mul4` library definitions ~~~
