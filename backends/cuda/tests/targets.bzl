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

    runtime.python_library(
        name = "autotune_test_utils",
        srcs = [
            "autotune_test_utils.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:autotune",
        ],
    )

    python_unittest_remote_gpu(
        name = "test_autotune_inputs",
        srcs = [
            "test_autotune_inputs.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:autotune",
            "//executorch/backends/cuda:cuda_backend",
            "//executorch/backends/cuda:cuda_partitioner",
            "//executorch/backends/cuda:triton_kernels",
            "//executorch/exir:lib",
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
            subplatform = "A100-exclusive",
        ),
    )

    python_unittest_remote_gpu(
        name = "test_cuda_graph_timing",
        srcs = [
            "test_cuda_graph_timing.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            ":autotune_test_utils",
            "//caffe2:torch",
            "//executorch/backends/cuda:autotune",
            "//executorch/backends/cuda:cuda_backend",
            "//executorch/backends/cuda:cuda_partitioner",
            "//executorch/exir:lib",
            "//executorch/exir/backend:compile_spec_schema",
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

    python_unittest_remote_gpu(
        name = "test_int4_quantized_gemm",
        srcs = [
            "test_int4_quantized_gemm.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            ":autotune_test_utils",
            "//caffe2:torch",
            "//executorch/backends/cuda:coalesced_int4_tensor",
            "//executorch/backends/cuda:cuda_backend",
            "//executorch/backends/cuda:cuda_partitioner",
            "//executorch/backends/cuda:quantize_op_dispatch",
            "//executorch/backends/cuda:triton_kernels",
            "//executorch/exir:lib",
            "//executorch/extension/llm/export:int4",
            "//executorch/extension/llm/export:quant",
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
            subplatform = "A100-exclusive",
        ),
    )

    python_unittest_remote_gpu(
        name = "test_offgraph_kv",
        srcs = [
            "test_offgraph_kv.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:cuda_passes",
            "//executorch/exir:lib",
            "//executorch/exir/dialects:lib",
            # The oracle is the neutral op itself, not a reference rebuilt here.
            "//executorch/extension/llm/cache:cache",
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
            subplatform = "A100-exclusive",
        ),
    )

    python_unittest(
        name = "test_quantized_gemm_family",
        srcs = [
            "test_quantized_gemm_family.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:triton_kernels",
        ],
    )

    python_unittest(
        name = "test_gemm_family_dispatch",
        srcs = [
            "test_gemm_family_dispatch.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:quantize_op_dispatch",
            "//executorch/backends/cuda:triton_kernels",
        ],
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
