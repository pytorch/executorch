load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    runtime.genrule(
        name = "gen_vk_delegate_schema",
        srcs = ["schema.fbs"],
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

    if is_fbcode:
        runtime.python_library(
            name = "lib",
            srcs = [
                "vulkan_graph_builder.py",
                "vulkan_graph_schema.py",
                "vulkan_graph_serialize.py",
            ],
            resources = [
                "schema.fbs",
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/exir:graph_module",
                "//executorch/exir/_serialize:_bindings",
                "//executorch/exir/_serialize:lib",
            ],
        )
