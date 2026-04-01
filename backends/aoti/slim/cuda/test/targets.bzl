load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")
load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")

def cuda_slim_cpp_unittest(name):
    cpp_unittest(
        name = "test_" + name,
        srcs = [
            "test_" + name + ".cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/cuda:guard",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/platform:platform",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
        ),
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    cuda_slim_cpp_unittest("cuda_guard")
    cuda_slim_cpp_unittest("cuda_stream_guard")
