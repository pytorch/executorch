load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "model",
        srcs = ["model.py"],
        visibility = [],  # Private
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.python_library(
        name = "utils",
        srcs = ["utils.py"],
        visibility = [],  # Private
        deps = [
            "//caffe2:torch",
            "fbsource//third-party/pypi/tqdm:tqdm",
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",  # @manual
            "//pytorch/vision:torchvision",
        ],
    )

    runtime.python_binary(
        name = "main",
        srcs = ["main.py"],
        main_function = "executorch.extension.training.examples.CIFAR.main.main",
        deps = [
            ":model",
            ":utils",
            "fbsource//third-party/pypi/tqdm:tqdm",
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",  # @manual
            "//pytorch/vision:torchvision",
        ],
    )

    runtime.cxx_binary(
        name = "train",
        srcs = ["train.cpp"],
        deps = [
            "//executorch/extension/training/module:training_module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/training/optimizer:sgd",
            "//executorch/runtime/executor:program",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/extension/flat_tensor/serialize:serialize_cpp",
        ],
        external_deps = ["gflags"],
        define_static_target = True,
    )
