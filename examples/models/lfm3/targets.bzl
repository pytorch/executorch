load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.python_library(
        name = "lfm3",
        srcs = [
            "__init__.py",
            "convert_weights.py",
        ],
        _is_external_target = True,
        base_module = "executorch.examples.models.lfm3",
        resources = {
            "config/1_7b_config.json": "config/1_7b_config.json",
        },
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models:checkpoint",
            "fbsource//third-party/pypi/safetensors:safetensors",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.python_library(
        name = "test_utils",
        srcs = ["test_utils.py"],
        _is_external_target = True,
        base_module = "executorch.examples.models.lfm3",
        deps = [
            ":lfm3",
            "//caffe2:torch",
            "//executorch/examples/models/llama:llama_transformer",
            "//executorch/examples/models/llama:transformer_modules",
        ],
        visibility = ["PUBLIC"],
    )

    python_unittest(
        name = "test_lfm3",
        srcs = ["test_lfm3.py"],
        base_module = "executorch.examples.models.lfm3",
        deps = [
            ":lfm3",
            ":test_utils",
            "//caffe2:torch",
            "//executorch/examples/models/llama:llama_transformer",
            "//executorch/examples/models/llama:transformer_modules",
            "//executorch/extension/llm/export:export_lib",
        ],
    )

    python_unittest(
        name = "test_lfm3_static",
        srcs = ["test_lfm3_static.py"],
        base_module = "executorch.examples.models.lfm3",
        deps = [
            ":lfm3",
            ":test_utils",
            "//caffe2:torch",
            "//executorch/backends/qualcomm/_passes:passes",
            "//executorch/backends/qualcomm/builders:builders",
            "//executorch/backends/qualcomm/quantizer:quantizer",
            "//executorch/examples/models/llama:transformer_modules",
            "//executorch/examples/qualcomm/oss_scripts/llama:static_llama",
        ],
    )
