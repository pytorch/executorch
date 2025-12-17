load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")

def cuda_runtime_cpp_unittest(name):
    cpp_unittest(
        name = "test_" + name,
        srcs = [
            "test_" + name + ".cpp",
        ],
        deps = [
            "//executorch/backends/cuda/runtime:runtime_shims",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/platform:platform",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    cuda_runtime_cpp_unittest("cuda_guard")
    cuda_runtime_cpp_unittest("cuda_stream_guard")
