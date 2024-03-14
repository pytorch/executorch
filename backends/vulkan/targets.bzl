load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def vulkan_spv_shader_lib(name, spv_filegroups, is_fbcode = False):
    gen_aten_vulkan_spv_target = "//caffe2/tools:gen_aten_vulkan_spv_bin"
    glslc_path = "//caffe2/fb/vulkan/dotslash:glslc"
    if is_fbcode:
        gen_aten_vulkan_spv_target = "//caffe2:gen_vulkan_spv_bin"
        glslc_path = "//caffe2/fb/vulkan/tools:glslc"

    glsl_paths = []

    # TODO(ssjia): remove the need for subpath once subdir_glob is enabled in OSS
    for target, subpath in spv_filegroups.items():
        glsl_paths.append("$(location {})/{}".format(target, subpath))

    genrule_cmd = [
        "$(exe {})".format(gen_aten_vulkan_spv_target),
        "--glsl-paths {}".format(" ".join(glsl_paths)),
        "--output-path $OUT",
        "--glslc-path=$(exe {})".format(glslc_path),
        "--tmp-dir-path=$OUT",
    ]

    genrule_name = "gen_{}_cpp".format(name)
    runtime.genrule(
        name = genrule_name,
        outs = {
            "{}.cpp".format(name): ["spv.cpp"],
        },
        cmd = " ".join(genrule_cmd),
        default_outs = ["."],
        labels = ["uses_dotslash"],
    )

    runtime.cxx_library(
        name = name,
        srcs = [
            ":{}[{}.cpp]".format(genrule_name, name),
        ],
        define_static_target = False,
        # Static initialization is used to register shaders to the global shader registry,
        # therefore link_whole must be True to make sure unused symbols are not discarded.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        # Define a soname that can be used for dynamic loading in Java, Python, etc.
        soname = "lib{}.$(ext)".format(name),
        exported_deps = [
            "//caffe2:torch_vulkan_api",
        ],
    )

def define_common_targets(is_fbcode = False):
    runtime.genrule(
        name = "gen_vk_delegate_schema",
        srcs = [
            "serialization/schema.fbs",
        ],
        # We're only generating a single file, so it seems like we could use
        # `out`, but `flatc` takes a directory as a parameter, not a single
        # file. Use `outs` so that `${OUT}` is expanded as the containing
        # directory instead of the file itself.
        outs = {
            "schema_generated.h": ["schema_generated.h"],
        },
        cmd = " ".join([
            "$(exe {})".format(runtime.external_dep_location("flatc")),
            "--cpp",
            "--cpp-std c++11",
            "--scoped-enums",
            "-o ${OUT}",
            "${SRCS}",
        ]),
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "vk_delegate_schema",
        srcs = [],
        visibility = [
            "//executorch/backends/vulkan/...",
        ],
        exported_headers = {
            "schema_generated.h": ":gen_vk_delegate_schema[schema_generated.h]",
        },
        exported_external_deps = [
            "flatbuffers-api",
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

    runtime.cxx_library(
        name = "vulkan_graph_runtime",
        srcs = native.glob([
            "runtime/graph/**/*.cpp",
        ]),
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
            ":vk_delegate_schema",
            ":vulkan_graph_runtime",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        define_static_target = False,
        # VulkanBackend.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )
