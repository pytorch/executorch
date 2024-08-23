load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "wrappers",
        srcs = glob([
            "*.cpp",
        ]),
        exported_headers = glob([
            "*.h",
        ]),
        define_static_target = True,
        platforms = [ANDROID],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "fbsource//third-party/qualcomm/qnn:api",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
        exported_deps = [
            "//executorch/backends/qualcomm/runtime:logging",
        ],
    )
