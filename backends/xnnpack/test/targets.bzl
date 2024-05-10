load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "dynamic_quant_utils_test",
        srcs = ["runtime/test_runtime_utils.cpp"],
        fbcode_deps = [
            "//caffe2:ATen-cpu",
        ],
        xplat_deps = [
            "//caffe2:aten_cpu",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/extension/aten_util:aten_bridge",
            "//executorch/backends/xnnpack:dynamic_quant_utils",
        ],
    )

    runtime.cxx_test(
        name = "xnnexecutor_test",
        srcs = ["runtime/test_xnnexecutor.cpp"],
        deps = [
            third_party_dep("XNNPACK"),
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/backends/xnnpack:xnnpack_backend",
        ],
    )
