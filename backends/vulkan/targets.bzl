load("@fbsource//tools/target_determinator/macros:ci.bzl", "ci")
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID", "CXX", "FBCODE", "APPLE")


def get_vulkan_compiler_flags():
    return select({
        "DEFAULT": [
            "-Wno-global-constructors",
            "-Wno-missing-prototypes",
        ],
        "ovr_config//os:windows": [],
    })

def get_vulkan_preprocessor_flags(no_volk, is_fbcode):
    VK_API_PREPROCESSOR_FLAGS = []

    default_flags = []
    android_flags = []

    debug_mode = read_config("etvk", "debug", "0") == "1"

    if not no_volk:
        for flags in [default_flags, android_flags]:
            flags.append("-DUSE_VULKAN_WRAPPER")
            flags.append("-DUSE_VULKAN_VOLK")
            flags.append("-DUSE_VOLK_HEADER_ONLY")
        android_flags.append("-DVK_ANDROID_external_memory_android_hardware_buffer")

    if not is_fbcode:
        link_moltenvk = no_volk and read_config("etvk", "link_moltenvk", "1") == "1"
        mac_flags = default_flags
        if link_moltenvk:
            mac_flags = []

        if debug_mode:
            mac_flags.append("-DETVK_BOOST_STACKTRACE_AVAILABLE")
            default_flags.append("-DETVK_BOOST_STACKTRACE_AVAILABLE")

        VK_API_PREPROCESSOR_FLAGS += select({
            "DEFAULT": default_flags,
            "ovr_config//os:android": android_flags,
            "ovr_config//os:macos": mac_flags,
        }) + select({
            "//third-party/cuda:windows-cuda-11": [
                "-DVK_USE_PLATFORM_WIN32_KHR",
            ],
            "DEFAULT": [],
            "ovr_config//os:android": [
                "-DVK_USE_PLATFORM_ANDROID_KHR",
            ],
            "ovr_config//os:linux": [
                "-DVK_USE_PLATFORM_XLIB_KHR",
            ],
            "ovr_config//os:macos": [
                "-DVK_USE_PLATFORM_MACOS_MVK",
            ],
            "ovr_config//os:windows": [
                "-DVK_USE_PLATFORM_WIN32_KHR",
            ],
        })

        etvk_default_cache_path = read_config("etvk", "default_cache_path", "")
        if etvk_default_cache_path != "":
            VK_API_PREPROCESSOR_FLAGS += ["-DETVK_DEFAULT_CACHE_PATH={}".format(etvk_default_cache_path)]

        if debug_mode:
            VK_API_PREPROCESSOR_FLAGS += ["-DVULKAN_DEBUG"]

    return VK_API_PREPROCESSOR_FLAGS

def get_labels(no_volk):
    if no_volk:
        return ci.labels(ci.linux(ci.mode("fbsource//arvr/mode/android/mac/dbg")))
    else:
        return []

def get_platforms():
    return [ANDROID, APPLE, CXX]

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

    nthreads = read_config("etvk", "shader_compile_nthreads", "-1")

    genrule_cmd = (
        "$(exe {}) ".format(gen_vulkan_spv_target) +
        "--glsl-paths {} ".format(" ".join(glsl_paths)) +
        "--output-path $OUT " +
        "--glslc-path=$(exe {}) ".format(glslc_path) +
        "--tmp-dir-path=shader_cache " +
        "--nthreads {} ".format(nthreads) +
        ("-f " if read_config("etvk", "force_shader_rebuild", "0") == "1" else " ") +
        select({
            "DEFAULT": "",
            "ovr_config//os:android": "--optimize",
            "ovr_config//os:linux": "--replace-u16vecn",
            "ovr_config//os:windows": "--optimize --spv_debug",
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
        platforms = get_platforms(),
        define_static_target = True,
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
    debug_mode = read_config("etvk", "debug", "0") == "1"

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
        visibility = ["PUBLIC"],
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

        VK_API_DEPS = [
            "fbsource//third-party/VulkanMemoryAllocator/3.0.1:VulkanMemoryAllocator_xplat",
        ]

        default_deps = []
        android_deps = ["fbsource//third-party/toolchains:android"]

        if no_volk:
            for deps in [default_deps, android_deps]:
                deps.append("fbsource//third-party/toolchains:vulkan")
                deps.append("fbsource//third-party/khronos:vulkan-headers")
        else:
            for deps in [default_deps, android_deps]:
                deps.append("fbsource//third-party/volk:volk-header")
                deps.append("fbsource//third-party/volk:volk-implementation")

        if is_fbcode:
            VK_API_DEPS += [
                "fbsource//third-party/swiftshader:swiftshader_vk_headers",
                "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_fbcode",
                "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_so",
            ]
        else:
            link_moltenvk = no_volk and read_config("etvk", "link_moltenvk", "1") == "1"
            mac_deps = default_deps
            if link_moltenvk:
                mac_deps = [
                    "//third-party/khronos:moltenVK_static"
                ]

            if debug_mode:
                mac_deps.append("fbsource//third-party/boost:boost")
                default_deps.append("fbsource//third-party/boost:boost")

            VK_API_DEPS += select({
                "DEFAULT": default_deps,
                "ovr_config//os:android": android_deps,
                "ovr_config//os:macos": mac_deps,
            }) + select({
                "DEFAULT": [],
                "ovr_config//os:linux": [
                    "//arvr/third-party/libX11:libX11",
                ]
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
            platforms = get_platforms(),
            visibility = ["PUBLIC"],
            fbobjc_frameworks = select({
                "DEFAULT": [],
                "ovr_config//os:macos": [
                    "$SDKROOT/System/Library/Frameworks/CoreGraphics.framework",
                    "$SDKROOT/System/Library/Frameworks/Foundation.framework",
                    "$SDKROOT/System/Library/Frameworks/AppKit.framework",
                    "$SDKROOT/System/Library/Frameworks/Metal.framework",
                    "$SDKROOT/System/Library/Frameworks/QuartzCore.framework",
                ],
            }),
            exported_preprocessor_flags = get_vulkan_preprocessor_flags(no_volk, is_fbcode),
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
            platforms = get_platforms(),
            visibility = ["PUBLIC"],
            exported_deps = [
                ":vulkan_graph_runtime_shaderlib{}".format(suffix),
                "//executorch/runtime/backend:interface",
            ],
            define_static_target = True,
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
            platforms = get_platforms(),
            visibility = ["PUBLIC"],
            deps = [
                ":vulkan_graph_runtime{}".format(suffix),
                "//executorch/backends/vulkan/serialization:vk_delegate_schema",
                "//executorch/runtime/core:event_tracer",
                "//executorch/runtime/core/exec_aten/util:tensor_util",
                "//executorch/runtime/core:named_data_map",
            ],
            define_static_target = True,
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
                "//executorch/exir/backend/canonical_partitioners:config_partitioner_lib",
                "//executorch/backends/vulkan/serialization:lib",
            ],
        )

        runtime.python_library(
            name = "custom_ops_lib",
            srcs = [
                "custom_ops_lib.py"
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan/patterns:vulkan_patterns",
            ]
        )

        runtime.python_library(
            name = "op_registry",
            srcs = [
                "op_registry.py",
            ],
            visibility = ["PUBLIC"],
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
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/backends/transforms:addmm_mm_to_linear",
                "//executorch/backends/transforms:fuse_batch_norm_with_conv",
                "//executorch/backends/transforms:fuse_conv_with_clamp",
                "//executorch/backends/transforms:fuse_view_copy",
                "//executorch/backends/transforms:remove_clone_ops",
                "//executorch/backends/transforms:view_copy_to_squeeze_unsqueeze",
                "//executorch/backends/vulkan/_passes:vulkan_passes",
                "//executorch/backends/vulkan/serialization:lib",
                "//executorch/backends/transforms:remove_getitem_op",
                "//executorch/backends/xnnpack/_passes:xnnpack_passes",
                "//executorch/exir/backend:backend_details",
            ],
        )
