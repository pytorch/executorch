load("@fbcode//target_determinator/macros:ci.bzl", "ci")
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID", "CXX", "FBCODE")


def get_vulkan_compiler_flags():
    return ["-Wno-missing-prototypes", "-Wno-global-constructors"]

def get_labels(no_volk):
    if no_volk:
        return ci.labels(ci.linux(ci.mode("fbsource//arvr/mode/android/mac/dbg")))
    else:
        return []

def get_platforms(no_volk):
    if no_volk:
        return [ANDROID]
    else:
        return [ANDROID, CXX]

def vulkan_spv_shader_lib(name, spv_filegroups, is_fbcode = False, no_volk = False):
    gen_vulkan_spv_target = "//xplat/executorch/backends/vulkan:gen_vulkan_spv_bin"
    glslc_path = "//xplat/caffe2/fb/vulkan/dotslash:glslc"

    if is_fbcode:
        gen_vulkan_spv_target = "//executorch/backends/vulkan:gen_vulkan_spv_bin"
        glslc_path = "//caffe2/fb/vulkan/tools:glslc"

    glsl_paths = []

    # TODO(ssjia): remove the need for subpath once subdir_glob is enabled in OSS
    for target, subpath in spv_filegroups.items():
        glsl_paths.append("$(location {})/{}".format(target, subpath))

    genrule_cmd = (
        "$(exe {}) ".format(gen_vulkan_spv_target) +
        "--glsl-paths {} ".format(" ".join(glsl_paths)) +
        "--output-path $OUT " +
        "--glslc-path=$(exe {}) ".format(glslc_path) +
        "--tmp-dir-path=$OUT " +
        select({
            "DEFAULT": "",
            "ovr_config//os:android": "--optimize",
            "ovr_config//os:linux": "--replace-u16vecn",
        })
    )

    genrule_name = "gen_{}_cpp".format(name)
    buck_genrule(
        name = genrule_name,
        outs = {
            "{}.cpp".format(name): ["spv.cpp"],
        },
        cmd = genrule_cmd,
        default_outs = ["."],
        labels = ["uses_dotslash"],
    )

    suffix = "_no_volk" if no_volk else ""
    runtime.cxx_library(
        name = name,
        srcs = [
            ":{}[{}.cpp]".format(genrule_name, name),
        ],
        compiler_flags = get_vulkan_compiler_flags(),
        labels = get_labels(no_volk),
        platforms = get_platforms(no_volk),
        define_static_target = False,
        # Static initialization is used to register shaders to the global shader registry,
        # therefore link_whole must be True to make sure unused symbols are not discarded.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        # Define a soname that can be used for dynamic loading in Java, Python, etc.
        soname = "lib{}.$(ext)".format(name),
        exported_deps = [
            "//executorch/backends/vulkan:vulkan_compute_api{}".format(suffix),
        ],
    )

def define_common_targets(is_fbcode = False):
    runtime.python_library(
        name = "gen_vulkan_spv_lib",
        srcs = [
            "runtime/gen_vulkan_spv.py",
        ],
        base_module = "",
        external_deps = ["torchgen"],
    )

    runtime.python_binary(
        name = "gen_vulkan_spv_bin",
        main_module = "runtime.gen_vulkan_spv",
        visibility = [
            "//executorch/backends/vulkan/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            ":gen_vulkan_spv_lib",
        ],
    )

    runtime.filegroup(
        name = "vulkan_graph_runtime_shaders",
        srcs = native.glob([
            "runtime/graph/ops/glsl/*",
        ]),
    )

    for no_volk in [True, False]:
        # No volk builds only available on xplat to build for Android
        if no_volk and is_fbcode:
            continue

        suffix = "_no_volk" if no_volk else ""

        VK_API_PREPROCESSOR_FLAGS = []
        VK_API_DEPS = [
            "fbsource//third-party/VulkanMemoryAllocator/3.0.1:VulkanMemoryAllocator_xplat",
        ]

        default_deps = []
        android_deps = ["fbsource//third-party/toolchains:android"]
        default_flags = []
        android_flags = []

        if no_volk:
            android_deps.append("fbsource//third-party/toolchains:vulkan")
        else:
            for deps in [default_deps, android_deps]:
                deps.append("fbsource//third-party/volk:volk")
            for flags in [default_flags, android_flags]:
                flags.append("-DUSE_VULKAN_WRAPPER")
                flags.append("-DUSE_VULKAN_VOLK")
            android_flags.append("-DVK_ANDROID_external_memory_android_hardware_buffer")

        if is_fbcode:
            VK_API_DEPS += [
                "fbsource//third-party/swiftshader:swiftshader_vk_headers",
                "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_fbcode",
                "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_so",
            ]
        else:
            VK_API_DEPS += select({
                "DEFAULT": default_deps,
                "ovr_config//os:android": android_deps,
                "ovr_config//os:macos": [
                    "//third-party/khronos:moltenVK_static"
                ],
            })
            VK_API_PREPROCESSOR_FLAGS += select({
                "DEFAULT": default_flags,
                "ovr_config//os:android": android_flags,
                "ovr_config//os:macos": []
            })

        runtime.cxx_library(
            name = "vulkan_compute_api{}".format(suffix),
            compiler_flags = get_vulkan_compiler_flags(),
            srcs = native.glob([
                "runtime/api/**/*.cpp",
                "runtime/utils/**/*.cpp",
                "runtime/vk_api/**/*.cpp",
            ]),
            exported_headers = native.glob([
                "runtime/api/**/*.h",
                "runtime/utils/**/*.h",
                "runtime/vk_api/**/*.h",
            ]),
            labels = get_labels(no_volk),
            platforms = get_platforms(no_volk),
            visibility = [
                "//executorch/backends/vulkan/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_preprocessor_flags = VK_API_PREPROCESSOR_FLAGS,
            exported_deps = VK_API_DEPS,
        )

        runtime.cxx_library(
            name = "vulkan_graph_runtime{}".format(suffix),
            srcs = native.glob([
                "runtime/graph/**/*.cpp",
            ]),
            compiler_flags = get_vulkan_compiler_flags(),
            exported_headers = native.glob([
                "runtime/graph/**/*.h",
            ]),
            labels = get_labels(no_volk),
            platforms = get_platforms(no_volk),
            visibility = [
                "//executorch/backends/...",
                "//executorch/extension/pybindings/...",
                "//executorch/test/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":vulkan_graph_runtime_shaderlib{}".format(suffix),
            ],
            define_static_target = False,
            # Static initialization is used to register operators to the global operator registry,
            # therefore link_whole must be True to make sure unused symbols are not discarded.
            # @lint-ignore BUCKLINT: Avoid `link_whole=True`
            link_whole = True,
            # Define an soname that can be used for dynamic loading in Java, Python, etc.
            soname = "libvulkan_graph_runtime.$(ext)",
        )

        vulkan_spv_shader_lib(
            name = "vulkan_graph_runtime_shaderlib{}".format(suffix),
            spv_filegroups = {
                ":vulkan_graph_runtime_shaders": "runtime/graph/ops/glsl",
            },
            is_fbcode = is_fbcode,
            no_volk = no_volk,
        )


        runtime.cxx_library(
            name = "vulkan_backend_lib{}".format(suffix),
            srcs = native.glob([
                "runtime/*.cpp",
            ]),
            compiler_flags = get_vulkan_compiler_flags(),
            headers = native.glob([
                "runtime/*.h",
            ]),
            labels = get_labels(no_volk),
            platforms = get_platforms(no_volk),
            visibility = [
                "//executorch/backends/...",
                "//executorch/extension/pybindings/...",
                "//executorch/test/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                ":vulkan_graph_runtime{}".format(suffix),
                "//executorch/backends/vulkan/serialization:vk_delegate_schema",
                "//executorch/runtime/core:event_tracer",
                "//executorch/runtime/backend:interface",
                "//executorch/runtime/core/exec_aten/util:tensor_util",
            ],
            define_static_target = False,
            # VulkanBackend.cpp needs to compile with executor as whole
            # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
            link_whole = True,
        )

    ##
    ## AOT targets
    ##
    if is_fbcode:
        runtime.python_library(
            name = "utils_lib",
            srcs = [
                "utils.py",
            ],
            visibility = [
                "//executorch/backends/vulkan/...",
            ],
            deps = [
                "//caffe2:torch",
                "//executorch/exir:tensor",
                "//executorch/backends/vulkan/serialization:lib",
            ]
        )

        runtime.python_library(
            name = "custom_ops_lib",
            srcs = [
                "custom_ops_lib.py"
            ],
            visibility = [
                "//executorch/...",
                "//executorch/vulkan/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//caffe2:torch",
            ]
        )

        runtime.python_library(
            name = "op_registry",
            srcs = [
                "op_registry.py",
            ],
            visibility = [
                "//executorch/...",
                "//executorch/vulkan/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                ":custom_ops_lib",
                ":utils_lib",
                "//caffe2:torch",
                "//executorch/exir/dialects:lib",
                "//executorch/backends/vulkan/serialization:lib",
            ]
        )

        runtime.python_library(
            name = "vulkan_preprocess",
            srcs = [
                "vulkan_preprocess.py",
            ],
            visibility = [
                "//executorch/...",
                "//executorch/vulkan/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/backends/transforms:addmm_mm_to_linear",
                "//executorch/backends/transforms:fuse_batch_norm_with_conv",
                "//executorch/backends/transforms:fuse_conv_with_clamp",
                "//executorch/backends/transforms:fuse_dequant_linear",
                "//executorch/backends/transforms:fuse_view_copy",
                "//executorch/backends/transforms:remove_clone_ops",
                "//executorch/backends/vulkan/_passes:vulkan_passes",
                "//executorch/backends/vulkan/serialization:lib",
                "//executorch/exir/backend:backend_details",
            ],
        )
