load("@fbcode_macros//build_defs:build_file_migration.bzl", "fbcode_target")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    fbcode_target(_kind = runtime.python_test,
        name = "test_speech_transform",
        srcs = ["test_speech_transform.py"],
        deps = [
            "//executorch/examples/models/gemma4:speech_transform",
            "fbsource//third-party/pypi/transformers:transformers",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_webgpu_rewrite_pass",
        srcs = ["test_webgpu_rewrite_pass.py"],
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:custom_ops_lib",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
        ],
    )
