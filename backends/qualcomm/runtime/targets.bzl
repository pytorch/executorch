load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/qnn_version.bzl", "get_qnn_library_verision")

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
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_verision()),
            "//executorch/runtime/backend:interface",
        ],
        exported_deps = [
            "//executorch/backends/qualcomm:schema",
            "//executorch/runtime/core:core",
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
        link_whole = True,  # needed for executorch/examples/models/llama:main to register QnnBackend
        platforms = [ANDROID],
        visibility = ["@EXECUTORCH_CLIENTS"],
        resources = {
            "qnn_lib": "fbsource//third-party/qualcomm/qnn/qnn-{0}:qnn_offline_compile_libs".format(get_qnn_library_verision()),
        },
        deps = [
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_verision()),
            ":logging",
            "//executorch/backends/qualcomm:schema",
            "//executorch/backends/qualcomm/aot/ir:qcir_utils",
            "//executorch/backends/qualcomm/aot/wrappers:wrappers",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/extension/tensor:tensor",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core:event_tracer",
        ],
    )
