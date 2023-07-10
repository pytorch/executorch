load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "core",
        exported_headers = [
            "array_ref.h",  # TODO(T157717874): Migrate all users to span and then move this to portable_type
            "data_loader.h",
            "error.h",
            "freeable_buffer.h",
            "function_ref.h",
            "result.h",
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
            "//executorch/core:core",  # for legacy clients that need Constants.h or macros.h TODO remove this
        ],
    )
