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
        name = "logging",
        srcs = [
            "Logging.cpp",
        ],
        exported_headers = [
            "Logging.h",
        ],
        define_static_target = True,
        platforms = [ANDROID],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "fbsource//third-party/qualcomm/qnn:api",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
        exported_deps = [
            "//executorch/backends/qualcomm:schema",
        ],
    )

    runtime.cxx_library(
        name = "runtime",
        srcs = glob(
            [
                "*.cpp",
                "backends/*.cpp",
                "backends/htpbackend/*.cpp",
                "backends/htpbackend/aarch64/*.cpp",
            ],
            exclude = ["Logging.cpp"],
        ),
        exported_headers = glob(
            [
                "*.h",
                "backends/*.h",
                "backends/htpbackend/*.h",
            ],
            exclude = ["Logging.h"],
        ),
        define_static_target = True,
        link_whole = True,  # needed for executorch/examples/models/llama2:main to register QnnBackend
        platforms = [ANDROID],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "fbsource//third-party/qualcomm/qnn:api",
            ":logging",
            "//executorch/backends/qualcomm:schema",
            "//executorch/backends/qualcomm/aot/ir:qcir_utils",
            "//executorch/backends/qualcomm/aot/wrappers:wrappers",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/extension/tensor:tensor",
        ],
    )
