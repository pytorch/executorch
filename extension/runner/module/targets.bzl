load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "module",
        srcs = [
            "module.cpp",
        ],
        exported_headers = [
            "module.h",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/extension/memory_allocator:malloc_memory_allocator",
            "//executorch/extension/data_loader:mmap_data_loader",
            "//executorch/extension/runner:runner",
        ],
    )

    runtime.cxx_library(
        name = "module_aten",
        srcs = [
            "module.cpp",
        ],
        exported_headers = [
            "module.h",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/extension/memory_allocator:malloc_memory_allocator",
            "//executorch/extension/data_loader:mmap_data_loader",
            "//executorch/extension/runner:runner_aten",
        ],
    )
