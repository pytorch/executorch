load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.export_file(
        name = "functions.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "executorch_aten_ops",
        ops_schema_yaml_target = ":functions.yaml",
        define_static_targets = True,
    )

    executorch_generated_lib(
        name = "generated_lib",
        aten_mode = True,
        deps = [
            ":executorch_aten_ops",
        ],
        functions_yaml_target = None,
        define_static_targets = True,
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
