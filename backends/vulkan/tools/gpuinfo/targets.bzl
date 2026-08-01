load("@fbcode_macros//build_defs:native_rules.bzl", "buck_filegroup")
load("@fbsource//tools/build_defs:fb_xplat_cxx_binary.bzl", "fb_xplat_cxx_binary")
load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID")
load("@fbsource//xplat/executorch/backends/vulkan:targets.bzl", "vulkan_spv_shader_lib")

def define_common_targets(is_fbcode = False):
    """Combined fbcode + xplat targets for backends/vulkan/tools/gpuinfo."""
    buck_filegroup(
        name = "gpuinfo_shaders",
        srcs = native.glob([
            "glsl/*",
        ]),
        visibility = [
            "PUBLIC",
        ],
    )

    vulkan_spv_shader_lib(
        name = "gpuinfo_shader_lib",
        is_fbcode = is_fbcode,
        spv_filegroups = {
            ":gpuinfo_shaders": "glsl",
        },
    )

    if is_fbcode:
        fb_xplat_cxx_binary(
            name = "vulkan_gpuinfo",
            srcs = native.glob([
                "**/*.cpp",
            ]),
            headers = native.glob([
                "**/*.h",
            ]),
            header_namespace = "/include",
            include_directories = ["/include"],
            platforms = ANDROID,
            raw_headers = native.glob([
                "**/*.h",
            ]),
            deps = [
                ":gpuinfo_shader_lib",
                "//executorch/backends/vulkan:vulkan_graph_runtime",
            ],
        )
    else:
        fb_xplat_cxx_binary(
            name = "vulkan_gpuinfo",
            srcs = native.glob([
                "**/*.cpp",
            ]),
            headers = native.glob([
                "**/*.h",
            ]),
            compiler_flags = select({
                "DEFAULT": [
                    "-Wno-header-hygiene",
                ],
                "ovr_config//compiler:cl": [],
            }),
            header_namespace = "/include",
            include_directories = ["/include"],
            platforms = ANDROID,
            raw_headers = native.glob([
                "**/*.h",
            ]),
            deps = [
                ":gpuinfo_shader_lib",
                "//arvr/third-party/opencl:headers",
                "//arvr/third-party/opencl:runtime",
                "//xplat/executorch/backends/vulkan:vulkan_graph_runtime",
                "//xplat/folly:json",
            ],
        )
