load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    common_deps = [
        "//executorch/runtime/kernel:kernel_includes",
    ]

    runtime.cxx_library(
        name = "kernels",
        srcs = ["kernels.cpp"],
        exported_headers = [
            "kernels.h",
        ],
        deps = common_deps,
        compatible_with = ["ovr_config//cpu:xtensa"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib_common",
        ],
        platforms = CXX,
    )
