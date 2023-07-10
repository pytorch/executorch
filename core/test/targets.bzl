load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "logging_test",
        srcs = [
            "LoggingTest.cpp",
        ],
        deps = [
            "//executorch/runtime/core:core",
        ],
        compiler_flags = [
            # Turn on debug logging.
            "-DET_MIN_LOG_LEVEL=Debug",
        ],
    )
