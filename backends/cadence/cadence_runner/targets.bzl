load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("@fbsource//tools/build_defs:fb_xplat_cxx_binary.bzl", "fb_xplat_cxx_binary")
load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")

def define_common_targets():
    fb_native.export_file(
        name = "cadence_runner.cpp",
        src = "cadence_runner.cpp",
        visibility = [
            "PUBLIC",
        ],
    )

    fb_xplat_cxx_binary(
        name = "cadence_runner",
        srcs = ["cadence_runner.cpp"],
        headers = [],
        platforms = CXX,
        visibility = ["PUBLIC"],
        deps = [
            "fbsource//arvr/third-party/gflags:gflags",
            "fbsource//xplat/executorch/devtools/etdump:etdump_flatcc",
            "fbsource//xplat/executorch/devtools/bundled_program:runtime",
            "fbsource//xplat/executorch/extension/data_loader:file_data_loader",
            "fbsource//xplat/executorch/extension/data_loader:buffer_data_loader",
            "fbsource//xplat/executorch/kernels/portable:generated_lib",
            "fbsource//xplat/executorch/runtime/executor:program",
        ],
    )
