load("//third-party:glob_defs.bzl", "subdir_glob")

TORCH_VULKAN_BASE_PATH = "pytorch/aten/src/ATen/native/vulkan/"

def aten_vulkan_path(suffix_path: str):
    return TORCH_VULKAN_BASE_PATH + suffix_path

def get_glsl_image_format():
    if read_config("pt", "vulkan_full_precision", "0") == "0":
        return "rgba16f"
    return "rgba32f"

def get_glsl_paths():
    base_path = "$(location :aten_vulkan_glsl_src_path)"
    sub_path = aten_vulkan_path("glsl")
    return base_path + "/" + sub_path

def define_torch_vulkan_targets():
    native.filegroup(
        name = "aten_vulkan_glsl_src_path",
        srcs = glob([aten_vulkan_path("glsl/*")] + [
            aten_vulkan_path("glsl/templates/*"),
        ]),
        visibility = [
            "PUBLIC",
        ],
    )

    native.python_library(
        name = "gen_aten_vulkan_spv_lib",
        srcs = [
            "pytorch/tools/gen_vulkan_spv.py",
        ],
        base_module = "",
    )

    native.python_binary(
        name = "gen_aten_vulkan_spv_bin",
        main_module = "pytorch.tools.gen_vulkan_spv",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_aten_vulkan_spv_lib",
        ],
    )

    native.http_archive(
        name = "glslc_archive",
        sha1 = "e2b3ba52827342e6922fcd309e5f9ec3656d70d4",
        urls = ["https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/linux/continuous_gcc_release/447/20240212-124340/install.tgz"],
        type = "tar.gz",
    )

    native.genrule(
        name = "glslc",
        out = "glslc",
        cmd = "cp $(location :glslc_archive)/install/bin/glslc $OUT",
        executable = True,
    )

    native.genrule(
        name = "gen_aten_vulkan_spv",
        outs = {
            "spv.cpp": ["spv.cpp"],
            "spv.h": ["spv.h"],
        },
        cmd = "$(exe :gen_aten_vulkan_spv_bin) " +
              "--glsl-paths " + get_glsl_paths() + " " +
              "--output-path $OUT --env FLOAT_IMAGE_FORMAT=" + get_glsl_image_format() + " " +
              "--glslc-path=$(exe :glslc) " +
              "--tmp-dir-path=$OUT ",
        default_outs = ["."],
        labels = ["uses_dotslash"],
    )

    native.cxx_library(
        name = "torch_vulkan_api",
        srcs = glob([aten_vulkan_path("api/*.cpp")]),
        header_namespace = "",
        exported_headers = subdir_glob([
            ("pytorch/aten/src", "ATen/native/vulkan/api/*.h"),
        ]),
        exported_preprocessor_flags = [
            "-DUSE_VULKAN_API",
            "-DUSE_VULKAN_WRAPPER",
            "-DUSE_VULKAN_VOLK",
        ],
        exported_deps = [
            "//backends/vulkan/third-party:volk",
            "//backends/vulkan/third-party:VulkanMemoryAllocator",
        ],
        visibility = ["PUBLIC"],
    )

    native.cxx_library(
        name = "torch_vulkan_spv",
        srcs = [
            ":gen_aten_vulkan_spv[spv.cpp]",
        ],
        header_namespace = "ATen/native/vulkan",
        exported_headers = {
            "spv.h": ":gen_aten_vulkan_spv[spv.h]",
        },
        exported_deps = [
            ":torch_vulkan_api",
        ],
        visibility = ["PUBLIC"],
    )

    native.cxx_library(
        name = "torch_vulkan_ops",
        srcs = glob([aten_vulkan_path("impl/*.cpp")]),
        header_namespace = "",
        exported_headers = subdir_glob([
            ("pytorch/aten/src", "ATen/native/vulkan/impl/*.h"),
        ]),
        exported_deps = [
            ":torch_vulkan_spv",
        ],
        visibility = ["PUBLIC"],
    )
