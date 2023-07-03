load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/profiler:profiler.bzl", "get_profiling_flags")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "profiler",
        srcs = [
            "profiler.cpp",
        ] + select({
            "DEFAULT": ["linux_hooks.cpp"],
            "ovr_config//cpu:xtensa": ["xtensa_executorch_hooks.cpp"],
        }),
        exported_preprocessor_flags = get_profiling_flags(),
        exported_headers = [
            "profiler.h",
            "hooks.h",
        ],
        deps = [
            "//executorch/core:abort",
            "//executorch/core:log",
        ],
        visibility = [
            "//executorch/backends/...",
            "//executorch/codegen/...",
            "//executorch/core/...",
            "//executorch/executor/...",
            "//executorch/kernels/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
