load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")
load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")

def cuda_shim_cpp_unittest(name):
    cpp_unittest(
        name = "test_" + name,
        srcs = [
            "test_" + name + ".cpp",
        ],
        deps = [
            "//executorch/backends/cuda/runtime:runtime_shims",
            "//executorch/backends/aoti:aoti_common_slim",
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

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    cuda_shim_cpp_unittest("aoti_torch_empty_strided")
    cuda_shim_cpp_unittest("aoti_torch_delete_tensor_object")
    cuda_shim_cpp_unittest("aoti_torch_create_tensor_from_blob_v2")
    cuda_shim_cpp_unittest("aoti_torch__reinterpret_tensor")
    cuda_shim_cpp_unittest("aoti_torch_copy_")
    cuda_shim_cpp_unittest("aoti_torch_cuda_guard")
    cuda_shim_cpp_unittest("aoti_torch_cuda__weight_int4pack_mm")
    cuda_shim_cpp_unittest("aoti_torch_cuda_rand")
    cuda_shim_cpp_unittest("aoti_torch_new_tensor_handle")
    cuda_shim_cpp_unittest("aoti_torch_item_bool")
    cuda_shim_cpp_unittest("aoti_torch_assign_tensors_out")

    cpp_unittest(
        name = "test_op__device_copy",
        srcs = ["test_op__device_copy.cpp"],
        deps = [
            "//executorch/backends/cuda/runtime:cuda_backend",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/kernels/portable:generated_lib_headers",
            "//executorch/kernels/portable/cpu:op__device_copy",
            "//executorch/runtime/core:device_allocator",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/portable_type:portable_type",
            "//executorch/runtime/kernel:kernel_runtime_context",
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
