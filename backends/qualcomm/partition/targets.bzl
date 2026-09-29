load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")

def define_common_targets():
    if not is_fbcode():
        return

    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "partition",
        srcs = glob([
            "*.py",
        ]),
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/exir/backend:backend_details",
            "//executorch/exir/backend:compile_spec_schema",
            "//executorch/backends/qualcomm/builders:builders",
            "//executorch/backends/qualcomm:preprocess",
            "//executorch/exir/backend/canonical_partitioners:canonical_partitioner_lib",
        ],
    )
