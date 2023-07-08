load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "arrayref_test",
        srcs = ["ArrayRefTest.cpp"],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "span_test",
        srcs = ["span_test.cpp"],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "StringTest",
        srcs = ["StringTest.cpp"],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "logging_test",
        srcs = [
            "LoggingTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
        compiler_flags = [
            # Turn on debug logging.
            "-DET_MIN_LOG_LEVEL=Debug",
        ],
    )

    runtime.cxx_test(
        name = "error_handling_test",
        srcs = [
            "ErrorHandlingTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "function_ref_test",
        srcs = [
            "FunctionRefTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "freeable_buffer_test",
        srcs = [
            "FreeableBufferTest.cpp",
        ],
        deps = [
            "//executorch/core:freeable_buffer",
        ],
    )
