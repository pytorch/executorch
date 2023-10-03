load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "ivalue_util",
        srcs = ["ivalue_util.cpp"],
        exported_headers = ["ivalue_util.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/extension/pytree:pytree",
            "//executorch/runtime/platform:platform",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        fbcode_deps = [
            "//caffe2:ATen-core",
            "//caffe2:ATen-cpu",
            "//caffe2/c10:c10",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_core",
            "//xplat/caffe2/c10:c10",
        ],
    )
