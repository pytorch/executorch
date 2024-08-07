load("@fbsource//tools/build_defs:fb_xplat_cxx_binary.bzl", "fb_xplat_cxx_binary")
load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")

def define_common_targets():
    fb_xplat_cxx_binary(
        name = "cadence_runner",
        srcs = ["cadence_runner.cpp"],
        headers = [],
        platforms = CXX,
        visibility = ["PUBLIC"],
        deps = [
            "fbsource//arvr/third-party/gflags:gflags",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/runtime/executor:program",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/data_loader:buffer_data_loader",
            "//executorch/util:util",
            "//executorch/sdk/etdump:etdump_flatcc",
            "//executorch/sdk/bundled_program:runtime",
        ],
    )
