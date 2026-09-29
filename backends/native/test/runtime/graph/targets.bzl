load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "graph_test",
        srcs = ["test_graph.cpp"],
        deps = [
            "//executorch/backends/native/runtime/graph:graph",
        ],
    )

    runtime.cxx_test(
        name = "graph_utils_test",
        srcs = ["test_graph_utils.cpp"],
        deps = [
            "//executorch/backends/native/runtime/graph:graph_utils",
        ],
    )
