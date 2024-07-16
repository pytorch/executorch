load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "test_backend_compatibility",
        srcs = [
            "test_backend_compatibility.cpp",
        ],
        deps = [
            "fbsource//third-party/googletest:gtest_main",
            "//executorch/runtime/backend:interface",
            "//executorch/exir/backend/test/demos/rpc:executor_backend",
            "//executorch/exir/backend/test/demos/rpc:executor_backend_register",
        ],
    )
