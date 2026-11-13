load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # rpcmem I/O buffers for the LLM runner's IOManagers. Registers itself on
    # load, so link it whole.
    runtime.cxx_library(
        name = "io_buffer_allocator",
        srcs = ["io_buffer_allocator.cpp"],
        link_whole = True,
        platforms = [ANDROID],
        deps = [
            "//executorch/backends/qualcomm/runtime:runtime_android_build",
            "//executorch/extension/llm/runner/io_manager:io_manager",
        ],
        visibility = ["PUBLIC"],
    )
