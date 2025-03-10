load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "thread_parallel",
        srcs = [
            "thread_parallel.cpp",
        ],
        exported_headers = [
            "thread_parallel.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/extension/threadpool:threadpool",
            "//executorch/runtime/core:core",
        ],
    )
