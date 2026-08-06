load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.python_library(
        name = "phi_4_mini",
        srcs = [
            "__init__.py",
            "convert_weights.py",
        ],
        _is_external_target = True,
        base_module = "executorch.examples.models.phi_4_mini",
        resources = {
            "config/config.json": "config/config.json",
            "config/phi_4_mini_xnnpack.yaml": "config/phi_4_mini_xnnpack.yaml",
        },
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models:model_base",
            "//executorch/examples/models/llama:llama2_model",
            "fbcode//pytorch/torchtune:lib",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.python_binary(
        name = "convert_phi4_mini_weights",
        main_function = "executorch.examples.models.phi_4_mini.convert_weights.main",
        deps = [
            ":phi_4_mini",
        ],
    )
