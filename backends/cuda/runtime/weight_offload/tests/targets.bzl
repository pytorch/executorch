load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")
load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Host-level: parses the on-wire payload only. No CUDA, no GPU.
    cpp_unittest(
        name = "payload_test",
        srcs = ["payload_test.cpp"],
        deps = [
            "//executorch/backends/cuda/runtime:cuda_backend",
            "//executorch/runtime/core:core",
        ],
    )

    # Session state machine (create/serve/evict/prefetch/pinned). Drives
    # the eviction + prefetch branches at unit granularity. Requires a
    # GPU; runs on the gpu-remote-execution platform.
    cpp_unittest(
        name = "session_test",
        srcs = ["session_test.cpp"],
        deps = [
            "//executorch/backends/cuda/runtime:cuda_backend",
            "//executorch/backends/cuda/runtime:runtime_shims",
            "//executorch/backends/aoti:aoti_common_slim",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
        preprocessor_flags = ["-DCUDA_AVAILABLE=1"],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
        ),
    )
