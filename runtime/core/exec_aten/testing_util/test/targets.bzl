load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "tensor_util_test",
        srcs = ["tensor_util_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "tensor_factory_test",
        srcs = ["tensor_factory_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "tensor_factory_test_aten",
        srcs = ["tensor_factory_test.cpp"],
        preprocessor_flags = ["-DUSE_ATEN_LIB"],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util_aten",
        ],
    )
