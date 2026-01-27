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
            "//executorch/backends/aoti:common_shims",
            "//executorch/backends/cuda/runtime:runtime_shims",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
        keep_gpu_sections = True,
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
        ),
    )

def cuda_shim_slim_cpp_unittest(name):
    """Unittest for SlimTensor-based shim functions."""
    cpp_unittest(
        name = "test_" + name + "_slim",
        srcs = [
            "test_" + name + "_slim.cpp",
        ],
        deps = [
            "//executorch/backends/cuda/runtime:runtime_shims_slim",
            "//executorch/backends/aoti:common_shims",
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/factory:empty",
            "//executorch/backends/aoti/slim/factory:from_blob",
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
    # Original ETensor-based shim tests, will be removed after migration
    cuda_shim_cpp_unittest("aoti_torch_empty_strided")
    cuda_shim_cpp_unittest("aoti_torch_delete_tensor_object")
    cuda_shim_cpp_unittest("aoti_torch_create_tensor_from_blob_v2")
    cuda_shim_cpp_unittest("aoti_torch__reinterpret_tensor")
    cuda_shim_cpp_unittest("aoti_torch_copy_")
    cuda_shim_cpp_unittest("aoti_torch_cuda_guard")
    cuda_shim_cpp_unittest("aoti_torch_cuda__weight_int4pack_mm")
    cuda_shim_cpp_unittest("aoti_torch_new_tensor_handle")
    cuda_shim_cpp_unittest("aoti_torch_item_bool")
    cuda_shim_cpp_unittest("aoti_torch_assign_tensors_out")

    # SlimTensor-based shim tests
    cuda_shim_slim_cpp_unittest("aoti_torch_create_tensor_from_blob_v2")
