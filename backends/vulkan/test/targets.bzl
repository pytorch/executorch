load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load(":compute_api_tests.bzl", "define_compute_api_test_targets")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Combined fbcode + xplat targets for backends/vulkan/test."""
    if is_fbcode:
        python_unittest(
            name = "test_vulkan_delegate",
            srcs = [
                "test_vulkan_delegate.py",
            ],
            preload_deps = [
                "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_fbcode",
                "//executorch/backends/vulkan:vulkan_backend_lib",
                "//executorch/kernels/portable:custom_ops_generated_lib",
            ],
            deps = [
                ":test_utils",
                "//caffe2:torch",
                "//executorch/backends/transforms:convert_dtype_pass",
                "//executorch/backends/vulkan:vulkan_preprocess",
                "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
                "//executorch/exir:lib",
                "//executorch/extension/pybindings:portable_lib",  # @manual
                "//executorch/extension/pytree:pylib",
                "//executorch/kernels/portable:custom_ops_generated_lib",
            ],
        )

        python_unittest(
            name = "test_vulkan_passes",
            srcs = [
                "test_vulkan_passes.py",
            ],
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan/_passes:vulkan_passes",
                "//executorch/backends/vulkan:vulkan_preprocess",
                "//executorch/backends/xnnpack/quantizer:xnnpack_quantizer",
                "//pytorch/ao:torchao",  # @manual
            ]
        )

        python_unittest(
            name = "test_vulkan_delegate_header",
            srcs = [
                "test_vulkan_delegate_header.py",
            ],
            deps = [
                "//executorch/backends/vulkan:vulkan_preprocess",
            ],
        )

        python_unittest(
            name = "test_vulkan_compile_options",
            srcs = [
                "test_vulkan_compile_options.py",
            ],
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan:vulkan_preprocess",
                "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
                "//executorch/exir/_serialize:lib",
                "//executorch/exir:lib",
            ],
        )

        python_unittest(
            name = "test_serialization",
            srcs = [
                "test_serialization.py",
            ],
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan:vulkan_preprocess",
            ],
        )

        python_unittest(
            name = "test_vulkan_tensor_repr",
            srcs = [
                "test_vulkan_tensor_repr.py",
            ],
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan:vulkan_preprocess",
            ],
        )

        runtime.python_library(
            name = "tester",
            srcs = ["tester.py"],
            deps = [
                "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
                "//executorch/backends/vulkan:vulkan_preprocess",
            ]
        )

        runtime.python_library(
            name = "test_utils",
            srcs = [
                "utils.py",
            ],
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan:vulkan_preprocess",
                "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
                "//executorch/backends/xnnpack:xnnpack_preprocess",
                "//executorch/backends/xnnpack/quantizer:xnnpack_quantizer",
                "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
                "//executorch/devtools:lib",
                "//executorch/devtools/bundled_program/serialize:lib",
                "//executorch/exir:lib",
                "//executorch/extension/pybindings:portable_lib",  # @manual
                "//executorch/extension/pytree:pylib",
            ],
        )
    else:
        fb_native.filegroup(
            name = "test_shaders",
            srcs = glob([
                "glsl/*",
            ]),
            visibility = [
                "PUBLIC",
            ],
        )

        define_compute_api_test_targets()
