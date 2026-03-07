load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "file_backend_cache_test",
        srcs = ["file_backend_cache_test.cpp"],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/extension/backend_cache:file_backend_cache",
        ],
    )
