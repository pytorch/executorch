load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "ivalue_util_test",
        srcs = ["ivalue_util_test.cpp"],
        deps = ["//executorch/extension/pytree/aten_util:ivalue_util"],
        fbcode_deps = [
            "//caffe2:torch-cpp",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_all_ops_et",
        ],
    )
