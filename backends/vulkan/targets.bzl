load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def get_vulkan_compiler_flags():
    return ["-Wno-missing-prototypes", "-Wno-global-constructors"]

def vulkan_spv_shader_lib(name, spv_filegroups, is_fbcode = False):
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

    runtime.cxx_library(
        name = name,
        srcs = [
            ":{}[{}.cpp]".format(genrule_name, name),
        ],
        compiler_flags = get_vulkan_compiler_flags(),
        define_static_target = False,
        # Static initialization is used to register shaders to the global shader registry,
        # therefore link_whole must be True to make sure unused symbols are not discarded.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        # Define a soname that can be used for dynamic loading in Java, Python, etc.
        soname = "lib{}.$(ext)".format(name),
        exported_deps = [
            "//executorch/backends/vulkan:vulkan_compute_api",
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

    vulkan_spv_shader_lib(
        name = "vulkan_graph_runtime_shaderlib",
        spv_filegroups = {
            ":vulkan_graph_runtime_shaders": "runtime/graph/ops/glsl",
        },
        is_fbcode = is_fbcode,
    )

    VK_API_PREPROCESSOR_FLAGS = []
    VK_API_DEPS = [
        "fbsource//third-party/VulkanMemoryAllocator/3.0.1:VulkanMemoryAllocator_xplat",
    ]

    if is_fbcode:
        VK_API_DEPS += [
            "fbsource//third-party/swiftshader:swiftshader_vk_headers",
            "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_fbcode",
            "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_so",
        ]
    else:
        VK_API_DEPS += select({
            "DEFAULT": [
                "fbsource//third-party/volk:volk",
            ],
            "ovr_config//os:android": [
                "fbsource//third-party/volk:volk",
                "fbsource//third-party/toolchains:android"
            ],
            "ovr_config//os:macos-arm64": [
                "//third-party/khronos:moltenVK"
            ],
        })
        VK_API_PREPROCESSOR_FLAGS += select({
            "DEFAULT": [
                "-DUSE_VULKAN_WRAPPER",
                "-DUSE_VULKAN_VOLK",
            ],
            "ovr_config//os:android": [
                "-DUSE_VULKAN_WRAPPER",
                "-DUSE_VULKAN_VOLK",
                "-DVK_ANDROID_external_memory_android_hardware_buffer"
            ],
            "ovr_config//os:macos-arm64": []
        })

    runtime.cxx_library(
        name = "vulkan_compute_api",
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
        visibility = [
            "//executorch/backends/vulkan/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_preprocessor_flags = VK_API_PREPROCESSOR_FLAGS,
        exported_deps = VK_API_DEPS,
    )

    runtime.cxx_library(
        name = "vulkan_graph_runtime",
        srcs = native.glob([
            "runtime/graph/**/*.cpp",
        ]),
        compiler_flags = get_vulkan_compiler_flags(),
        exported_headers = native.glob([
            "runtime/graph/**/*.h",
        ]),
        visibility = [
            "//executorch/backends/...",
            "//executorch/extension/pybindings/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            ":vulkan_graph_runtime_shaderlib",
        ],
        define_static_target = False,
        # Static initialization is used to register operators to the global operator registry,
        # therefore link_whole must be True to make sure unused symbols are not discarded.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        # Define an soname that can be used for dynamic loading in Java, Python, etc.
        soname = "libvulkan_graph_runtime.$(ext)",
    )

    runtime.cxx_library(
        name = "vulkan_backend_lib",
        srcs = native.glob([
            "runtime/*.cpp",
        ]),
        compiler_flags = get_vulkan_compiler_flags(),
        headers = native.glob([
            "runtime/*.h",
        ]),
        visibility = [
            "//executorch/backends/...",
            "//executorch/extension/pybindings/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            ":vulkan_graph_runtime",
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
