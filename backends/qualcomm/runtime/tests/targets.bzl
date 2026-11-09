load("@fbsource//tools/build_defs:default_platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/backends/qualcomm/third-party:third_party_libs.bzl", "qnn_third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "shared_buffer_test",
        srcs = ["SharedBufferTest.cpp"],
        compatible_with = [] if runtime.is_oss else ["ovr_config//os:linux"],
        platforms = [CXX],
        deps = [
            "//executorch/backends/qualcomm/runtime:runtime",
            qnn_third_party_dep("api"),
        ],
    )
