load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    python_library(
        name = "resnet_model",
        srcs = [
            "__init__.py",
            "model.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models:model_base",
            "//pytorch/vision:torchvision",  # @manual
        ],
    )
