load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "aten_bridge_test",
        srcs = [
            "aten_bridge_test.cpp",
            "make_aten_functor_from_et_functor_test.cpp",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:operator_registry",
            "//executorch/extension/aten_util:aten_bridge",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
        external_deps = [
            "libtorch",
            "gtest_aten",
        ],
    )
