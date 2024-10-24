load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
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
        visibility = [
            "//executorch/...",
            "//executorch/vulkan/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/exir:graph_module",
            "//executorch/exir/_serialize:_bindings",
            "//executorch/exir/_serialize:lib",
        ],
    )
