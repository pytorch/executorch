load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/extension/pybindings:pybindings.bzl", "MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "executor_backend",
        srcs = [
            "ExecutorBackend.cpp",
        ],
        exported_headers = [
            "ExecutorBackend.h",
        ],
        platforms = [ANDROID, CXX],
        deps = [
            "//executorch/runtime/executor:program",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/runtime/backend:interface",
            "//executorch/extension/data_loader:buffer_data_loader",
        ] + MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB,
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "executor_backend_register",
        srcs = [
            "ExecutorBackendRegister.cpp",
        ],
        visibility = [
            "//executorch/exir/backend/test/...",
        ],
        deps = [
            ":executor_backend",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
        platforms = [ANDROID, CXX],
    )
