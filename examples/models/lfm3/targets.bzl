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
