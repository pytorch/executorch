load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/qnn_version.bzl", "get_qnn_library_version")

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
        visibility = ["PUBLIC"],
        deps = [
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_version()),
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:app_sources".format(get_qnn_library_version()),
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
        exported_deps = [
            "//executorch/backends/qualcomm/runtime:logging",
        ],
    )
