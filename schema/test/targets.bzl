load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "extended_header_test",
        srcs = [
            "extended_header_test.cpp",
        ],
        deps = [
            "//executorch/schema:extended_header",
        ],
    )
