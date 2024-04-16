load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.python_unittest(
        name = "test_simple_sdpa",
        srcs = [
            "test_simple_sdpa.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/llama2:export_library",
            "//executorch/examples/models/llama2:llama_transformer",
        ],
    )
