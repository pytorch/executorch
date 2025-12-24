load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define test targets for SlimTensor c10 core module."""

    runtime.cxx_test(
        name = "test_device",
        srcs = [
            "test_device.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/c10/core:device",
            "//executorch/backends/aoti/slim/c10/core:device_type",
        ],
    )

    runtime.cxx_test(
        name = "test_scalar_type",
        srcs = [
            "test_scalar_type.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/c10/core:scalar_type",
        ],
    )
