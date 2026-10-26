load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "runner_utils",
        srcs = ["runner_utils.cpp"],
        exported_headers = ["runner_utils.h"],
        visibility = ["//executorch/backends/native/examples/..."],
    )

    runtime.cxx_test(
        name = "runner_utils_test",
        srcs = ["test_runner_utils.cpp"],
        deps = [":runner_utils"],
    )
