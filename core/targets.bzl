load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "core",
        exported_headers = [
            "ArrayRef.h",
            "Constants.h",
            "Error.h",
            "FunctionRef.h",
            "Result.h",
            "macros.h",
            "span.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/profiler:profiler",
        ],
        exported_deps = [
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.cxx_library(
        name = "freeable_buffer",
        exported_headers = ["FreeableBuffer.h"],
        visibility = [
            "//executorch/backends/...",
            "//executorch/core/test/...",
            "//executorch/executor/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [],
    )

    runtime.cxx_library(
        name = "data_loader",
        exported_headers = [
            "DataLoader.h",
        ],
        exported_deps = [
            "//executorch/runtime/platform:platform",
            "//executorch/core:core",
            ":freeable_buffer",
        ],
        visibility = [
            "//executorch/core/test/...",
            "//executorch/executor/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
