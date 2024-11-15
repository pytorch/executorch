load("@fbcode//target_determinator/macros:ci.bzl", "ci")
load("@fbsource//tools/build_defs:fb_xplat_cxx_binary.bzl", "fb_xplat_cxx_binary")
load("@fbsource//tools/build_defs:fb_xplat_cxx_test.bzl", "fb_xplat_cxx_test")
load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID", "MACOSX", "CXX")
load(
    "@fbsource//xplat/executorch/backends/vulkan:targets.bzl",
    "vulkan_spv_shader_lib",
)

def define_compute_api_test_targets():
    for no_volk in [True, False]:
        suffix = "_no_volk" if no_volk else ""
        platforms = [ANDROID, CXX]
        labels = []
        if no_volk:
            platforms = [ANDROID]
            labels = ci.labels(ci.linux(ci.mode("fbsource//arvr/mode/android/mac/dbg")))

        vulkan_spv_shader_lib(
            name = "test_shader_lib{}".format(suffix),
            spv_filegroups = {
                ":test_shaders": "glsl",
            },
            no_volk = no_volk,
        )

        fb_xplat_cxx_binary(
            name = "vulkan_compute_api_test_bin{}".format(suffix),
            srcs = [
                "utils/test_utils.cpp",
                "vulkan_compute_api_test.cpp",
            ],
            headers = [
                "utils/test_utils.h",
            ],
            apple_sdks = MACOSX,
            labels = labels,
            platforms = platforms,
            visibility = ["PUBLIC"],
            deps = [
                ":test_shader_lib{}".format(suffix),
                "//third-party/googletest:gtest_main",
                "//xplat/executorch/backends/vulkan:vulkan_graph_runtime{}".format(suffix),
                "//xplat/executorch/runtime/core/exec_aten:lib",
            ],
        )

        fb_xplat_cxx_test(
            name = "vulkan_compute_api_test{}".format(suffix),
            srcs = [
                "utils/test_utils.cpp",
                "vulkan_compute_api_test.cpp",
            ],
            headers = [
                "utils/test_utils.h",
            ],
            contacts = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"],
            fbandroid_additional_loaded_sonames = [
                "test_shader_lib",
                "vulkan_graph_runtime",
                "vulkan_graph_runtime_shaderlib",
            ],
            platforms = [ANDROID],
            use_instrumentation_test = True,
            visibility = ["PUBLIC"],
            deps = [
                ":test_shader_lib",
                "//third-party/googletest:gtest_main",
                "//xplat/executorch/backends/vulkan:vulkan_graph_runtime{}".format(suffix),
                "//xplat/executorch/runtime/core/exec_aten:lib",
            ],
        )
