load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "malloc_memory_allocator_test",
        srcs = [
            "malloc_memory_allocator_test.cpp",
        ],
        deps = [
            "//executorch/extension/memory_allocator:malloc_memory_allocator",
        ],
    )
