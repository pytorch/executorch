load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "tensorrt_example_lib",
        srcs = [
            "__init__.py",
            "export.py",
            "runner.py",
        ],
        visibility = [
            "//executorch/examples/nvidia/tensorrt/...",
            "//executorch/backends/nvidia/tensorrt/...",
        ],
        deps = [
            "//caffe2:torch",
            "//deeplearning/trt/python:py_tensorrt",
            "//executorch/backends/nvidia/tensorrt:serialization",
            "//executorch/backends/nvidia/tensorrt/partitioner:partitioner",
            "//executorch/examples/models:models",
            "//executorch/exir:lib",
            "//executorch/extension/export_util:export_util",
            "//executorch/extension/pybindings:portable_lib",
            "//executorch/extension/pytree:pytree",
        ],
    )

    runtime.python_binary(
        name = "export",
        main_module = "executorch.examples.nvidia.tensorrt.export",
        preload_deps = [
            "//executorch/backends/nvidia/tensorrt/runtime:tensorrt_backend",
        ],
        deps = [
            ":tensorrt_example_lib",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.python_binary(
        name = "runner",
        main_module = "executorch.examples.nvidia.tensorrt.runner",
        deps = [
            ":tensorrt_example_lib",
        ],
        visibility = ["PUBLIC"],
    )

    # C++ executor runner for TensorRT models.
    # Requires TensorRT and CUDA (NVIDIA GPUs). Not for mobile builds.
    runtime.cxx_binary(
        name = "tensorrt_executor_runner",
        srcs = ["tensorrt_executor_runner.cpp"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/runner_util:inputs",
            "//executorch/runtime/executor:program",
            "//executorch/backends/nvidia/tensorrt/runtime:tensorrt_backend",
            "//executorch/kernels/portable:generated_lib",
        ],
    )
