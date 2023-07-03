load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "core",
        srcs = [
            "Runtime.cpp",
            "String.cpp",
        ],
        exported_headers = [
            "ArrayRef.h",
            "Constants.h",
            "Error.h",
            "FunctionRef.h",
            "Result.h",
            "Runtime.h",
            "String.h",
            "macros.h",
            "span.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            ":log",
            "//executorch/profiler:profiler",
        ],
        exported_deps = [
            ":abort",
            ":log",
            "//executorch/compiler:compiler",
            "//executorch/platform:platform",
            # Must be exported to include the weak symbols it defines.
            "//executorch/platform:platform_private",
        ],
    )

    runtime.cxx_library(
        name = "abort",
        srcs = [
            "Abort.cpp",
        ],
        exported_headers = [
            "Assert.h",
            "Abort.h",
        ],
        deps = [
            "//executorch/compiler:compiler",
            "//executorch/platform:platform_private",
        ],
        exported_deps = [
            ":log",
            "//executorch/compiler:compiler",
            "//executorch/platform:platform",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        reexport_all_header_dependencies = False,
    )

    runtime.cxx_library(
        name = "log",
        srcs = [
            "Log.cpp",
        ],
        exported_headers = [
            "Log.h",
        ],
        deps = [
            "//executorch/compiler:compiler",
            "//executorch/platform:platform",
            "//executorch/platform:platform_private",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
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
            "//executorch/compiler:compiler",
            "//executorch/core:core",
            ":freeable_buffer",
        ],
        visibility = [
            "//executorch/core/test/...",
            "//executorch/executor/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "operator_registry",
        srcs = ["OperatorRegistry.cpp"],
        exported_headers = ["OperatorRegistry.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            ":core",
            "//executorch/core/values:executor_values",
        ],
    )
