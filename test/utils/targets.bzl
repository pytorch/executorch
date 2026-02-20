load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "utils" + aten_suffix,
            srcs = [
                "UnitTestMain.cpp",
            ],
            exported_headers = [
                "alignment.h",
                "DeathTest.h",
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/runtime/platform:platform",
                "//executorch/runtime/core:core",
            ],
            exported_external_deps = [
                "gtest" + aten_suffix,
                "gmock" + aten_suffix,
            ],
        )

        runtime.cxx_test(
            name = "alignment_test" + aten_suffix,
            srcs = [
                "alignment_test.cpp",
            ],
            deps = [
                ":utils" + aten_suffix,
            ],
        )
