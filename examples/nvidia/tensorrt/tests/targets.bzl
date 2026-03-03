load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Export correctness tests: exports each supported model with TensorRT
    # and compares inference outputs against eager PyTorch on GPU.
    #   buck2 test $GPU_FLAGS fbcode//executorch/examples/nvidia/tensorrt/tests:test_export
    #   buck2 test $GPU_FLAGS fbcode//executorch/examples/nvidia/tensorrt/tests:test_export -- test_add
    runtime.python_test(
        name = "test_export",
        srcs = ["test_export.py"],
        labels = ["long_running"],
        preload_deps = [
            "//executorch/backends/nvidia/tensorrt/runtime:tensorrt_backend",
        ],
        deps = [
            "//executorch/examples/nvidia/tensorrt:tensorrt_example_lib",
        ],
    )
