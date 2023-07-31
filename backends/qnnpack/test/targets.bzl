load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "APPLE",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "qnnpack_utils_test",
        srcs = ["test_utils.cpp"],
        fbcode_deps = [
            "//caffe2:ATen-cpu",
        ],
        xplat_deps = [
            "//caffe2:aten_cpu",
        ],
        platforms = [ANDROID, APPLE, CXX],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/extension/aten_util:aten_bridge",
            "//executorch/backends/qnnpack:qnnpack_utils",
        ],
    )
