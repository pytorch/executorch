load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.python_test(
        name = "test_runtime",
        srcs = ["test_runtime.py"],
        deps = [
            "//executorch/extension/pybindings/test:make_test",
            "//executorch/runtime:runtime",
            "//executorch/devtools/etdump:serialize",
        ],
    )

    runtime.python_test(
        name = "test_runtime_etdump_gen",
        srcs = ["test_runtime_etdump_gen.py"],
        deps = [
            "//executorch/extension/pybindings/test:make_test",
            "//executorch/runtime:runtime",
            "//executorch/devtools/etdump:serialize",
        ],
    )

    runtime.python_test(
        name = "test_runtime_xnnpack",
        srcs = ["test_runtime_xnnpack.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
            "//executorch/exir:lib",
            "//executorch/runtime:runtime",
        ],
    )
