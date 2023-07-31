load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "memory_utils_test",
        srcs = [
            "memory_utils_test.cpp",
        ],
        deps = [
            "//executorch/runtime/platform:compiler",
            "//executorch/util:memory_utils",
        ],
    )
