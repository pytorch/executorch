load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:export_files.bzl", "export_file")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    
    export_file(
        name = "qnn_schema",
        src = "schema.fbs",
        visibility = ["//executorch/backends/qualcomm/serialization/..."],
    )

    runtime.python_library(
        name = "serialization",
        srcs = glob([
            "*.py",
        ]),
        resources = {
            ":qnn_schema": "schema.fbs",
        },
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/exir/backend:backend_details",
            "//executorch/exir/backend:compile_spec_schema",
        ],
    )
