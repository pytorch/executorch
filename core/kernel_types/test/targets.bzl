load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "tensor_shape_dynamism_test_aten",
        srcs = ["TensorShapeDynamismAtenTest.cpp"],
        deps = [
            "//executorch/core/kernel_types:kernel_types_aten",
        ],
    )
