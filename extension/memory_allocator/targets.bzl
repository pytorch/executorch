load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "malloc_memory_allocator",
        exported_headers = [
            "malloc_memory_allocator.h",
            "memory_allocator_utils.h",
        ],
        exported_deps = [
            "//executorch/runtime/core:memory_allocator",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "cpu_caching_allocator",
        srcs = [
            "cpu_caching_malloc_allocator.cpp",
        ],
        exported_headers = [
            "cpu_caching_malloc_allocator.h",
            "memory_allocator_utils.h",
        ],
        exported_deps = [
            "//executorch/runtime/core:memory_allocator",
        ],
        visibility = ["PUBLIC"],
    )
