load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
load(":lib_defs.bzl", "define_libs")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    define_libs()

    runtime.export_file(
        name = "optimized.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.export_file(
        name = "optimized-oss.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "optimized_operators",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/optimized/cpu:cpu_optimized",
        ],
    )

    et_operator_library(
        name = "optimized_oplist",
        ops_schema_yaml_target = ":optimized.yaml",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Used mainly for operator testing. In practice, a generated lib specific
    # to a project should be created that contains only the required operators
    # for a particular model.
    executorch_generated_lib(
        name = "generated_lib",
        deps = [
            ":optimized_oplist",
            ":optimized_operators",
        ],
        functions_yaml_target = ":optimized.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
