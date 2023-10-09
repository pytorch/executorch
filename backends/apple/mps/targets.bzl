load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "mps_backend",
        srcs = native.glob([
            "runtime/*.mm",
        ]),
        headers = native.glob([
            "runtime/*.h",
        ]),
        visibility = [
            "//executorch/exir/backend:backend_lib",
            "//executorch/exir/backend/test/...",
            "//executorch/backends/apple/mps/test/...",
            "//executorch/extension/pybindings/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/backends/apple/mps/runtime:MPSBackend",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        define_static_target = True,
        link_whole = True,
    )
