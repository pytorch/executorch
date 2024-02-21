load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
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
            "//caffe2:torch_vulkan_graph",
            "//executorch/runtime/backend:interface",
        ],
        # VulkanBackend.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )
