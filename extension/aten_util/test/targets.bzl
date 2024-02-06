load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "aten_bridge_test",
        srcs = ["aten_bridge_test.cpp"],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/extension/aten_util:aten_bridge",
        ],
        fbcode_deps = [
            "//caffe2:ATen-core",
            "//caffe2:ATen-cpu",
            "//caffe2/c10:c10",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_core",
            "//xplat/caffe2/c10:c10",
            # Dont really like this but without this I dont have aten::empty
            # And havent figured out a more minimal target
            "//xplat/caffe2:torch_mobile_all_ops_et",
        ],
    )
