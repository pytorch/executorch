load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/third-party:third_party_libs.bzl", "qnn_third_party_dep")

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
            qnn_third_party_dep("api"),
            qnn_third_party_dep("app_sources"),
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
        exported_deps = [
            "//executorch/backends/qualcomm/runtime:logging",
        ],
    )
