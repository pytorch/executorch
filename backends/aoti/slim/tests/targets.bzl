load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")

def slim_tensor_cpp_unittest(name, extra_deps = []):
    cpp_unittest(
        name = "test_" + name,
        srcs = [
            "test_" + name + ".cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim:slim_tensor_cpu",
        ] + extra_deps,
    )

def slim_tensor_cuda_cpp_unittest(name):
    cpp_unittest(
        name = "test_" + name,
        srcs = [
            "test_" + name + ".cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim:slim_tensor",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )

def define_common_targets():
    """Define test targets for SlimTensor library."""
    slim_tensor_cpp_unittest("slim_tensor_basic")
    slim_tensor_cuda_cpp_unittest("slim_tensor_cuda")
