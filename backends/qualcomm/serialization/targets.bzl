load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:export_files.bzl", "export_file")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Used by the INPUT_SCHEMA genrule in targets.bzl for OSS builds,
    # where cross-package relative paths don't work in genrule srcs.
    export_file(
        name = "qc_compiler_spec.fbs",
        visibility = ["PUBLIC"],
    )

    export_file(
        name = "qc_compiler_spec_schema",
        src = "qc_compiler_spec.fbs",
        visibility = ["PUBLIC"],
    )

    runtime.python_library(
        name = "serialization",
        srcs = glob([
            "*.py",
        ]),
        resources = {
            ":qc_compiler_spec_schema": "qc_compiler_spec.fbs",
        },
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/exir/backend:backend_details",
            "//executorch/exir/backend:compile_spec_schema",
        ],
    )
