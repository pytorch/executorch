load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define test targets for SlimTensor core module."""

    runtime.cxx_test(
        name = "test_storage_cpu",
        srcs = [
            "test_storage_cpu.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/core:storage",
        ],
    )

    runtime.cxx_test(
        name = "test_slimtensor_basic",
        srcs = [
            "test_slimtensor_basic.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/core:storage",
        ],
    )

    runtime.cxx_test(
        name = "test_slimtensor_copy",
        srcs = [
            "test_slimtensor_copy.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/core:storage",
        ],
    )
