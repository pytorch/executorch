load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "platform_test",
        srcs = [
            "executor_pal_test.cpp",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.cxx_test(
        name = "platform_death_test",
        srcs = [
            "executor_pal_death_test.cpp",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )

    # This is an example of a target that provides a PAL implementation. Note
    # the `link_whole = True` parameter, which is necessary to ensure that the
    # symbols make their way into the top-level binary. If this target were to
    # be added to a library instead of directly to a binary, it would need to be
    # in that library's `exported_deps`.
    runtime.cxx_library(
        name = "stub_platform",
        srcs = [
            "stub_platform.cpp",
        ],
        exported_headers = [
            "stub_platform.h",
        ],
        deps = [
            "//executorch/runtime/platform:compiler",
            "//executorch/runtime/platform:platform",
            "//executorch/test/utils:utils",  # gtest.h
        ],
        visibility = [],
    )

    runtime.cxx_test(
        name = "platform_override_test",
        srcs = [
            "executor_pal_override_test.cpp",
        ],
        deps = [
            # This must come first to ensure that the weak platform
            # calls are overriden.
            # buildifier: do not sort
            ":stub_platform",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.cxx_test(
        name = "logging_test",
        srcs = [
            "logging_test.cpp",
        ],
        deps = [
            "//executorch/runtime/platform:platform",
        ],
        compiler_flags = [
            # Turn on debug logging.
            "-DET_MIN_LOG_LEVEL=Debug",
        ],
    )

    runtime.cxx_test(
        name = "clock_test",
        srcs = [
            "clock_test.cpp",
        ],
        deps = [
            # This must come first to ensure that the weak platform
            # calls are overriden.
            # buildifier: do not sort
            ":stub_platform",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )
