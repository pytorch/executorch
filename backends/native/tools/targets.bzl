load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_binary(
        name = "native_executor",
        srcs = ["native_executor.cpp"],
        deps = [
            "//executorch/backends/native/runtime:runtime",
        ],
        external_deps = ["gflags"],
        visibility = ["PUBLIC"],
    )
