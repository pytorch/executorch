load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    
    runtime.python_library(
        name = "utils",
        srcs = glob([
            "*.py",
        ]),
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/exir/backend:backend_details",
            "//executorch/exir/backend:compile_spec_schema",
            "//executorch/backends/qualcomm/serialization:serialization",
        ],
    )
