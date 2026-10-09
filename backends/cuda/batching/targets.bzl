load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.cxx_library(
        name = "step_plan",
        exported_headers = [
            "step_plan.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/platform:platform",
        ],
    )

    cpp_unittest(
        name = "test_step_plan",
        srcs = [
            "test/test_step_plan.cpp",
        ],
        deps = [
            ":step_plan",
            "//executorch/test/utils:utils",
        ],
    )
