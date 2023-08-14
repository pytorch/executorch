"""Client build configurations.

This package contains useful build targets for executorch clients, assembling
common collections of deps into self-contained targets.
"""

load("@fbsource//xplat/executorch/backends:backends.bzl", "get_all_cpu_backend_targets")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # An extended executor library that includes all CPU backend targets and
    # helper deps.
    runtime.cxx_library(
        name = "executor_cpu_optimized",
        exported_deps = [
            "//executorch/backends/xnnpack/threadpool:threadpool",
        ] + get_all_cpu_backend_targets(),
        visibility = [
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
