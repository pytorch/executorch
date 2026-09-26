load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbcode_macros//build_defs:python_unittest_remote_gpu.bzl", "python_unittest_remote_gpu")
load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    python_unittest_remote_gpu(
        name = "test_cuda_export",
        srcs = [
            "test_cuda_export.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:cuda_backend",
            "//executorch/backends/cuda:cuda_partitioner",
            "//executorch/exir:lib",
            "//executorch/exir/backend:backend_api",
            "//executorch/exir/backend:compile_spec_schema",
            "//executorch/examples/models/toy_model:toy_model",
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
            subplatform = "A100-exclusive",
        ),
    )

    python_unittest_remote_gpu(
        name = "test_triton_sdpa_splitk",
        srcs = [
            "test_triton_sdpa_splitk.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:triton_kernels",
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
            subplatform = "A100-exclusive",
        ),
    )

    python_unittest(
        name = "test_cuda_partitioner",
        srcs = [
            "test_cuda_partitioner.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:cuda_partitioner",
            "//executorch/backends/cuda:cuda_backend",
            "//executorch/exir:lib",
            "//executorch/exir/backend:compile_spec_schema",
        ],
    )
