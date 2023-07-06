load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "platform_test",
        srcs = [
            "ExecutorPalTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "platform_death_test",
        srcs = [
            "ExecutorPalDeathTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
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
            "StubPlatform.cpp",
        ],
        exported_headers = [
            "StubPlatform.h",
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
            "ExecutorPalOverrideTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
            ":stub_platform",
        ],
    )
