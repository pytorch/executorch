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
        name = "data_utils",
        srcs = ["data_utils.py"],
        deps = [
            "//caffe2:torch",
            "//pytorch/vision:torchvision",
        ],
    )

    runtime.python_binary(
        name = "data_processing",
        srcs = ["data_utils.py"],
        main_function = "executorch.extension.training.examples.CIFAR.data_utils.main",
        deps = [
            "//caffe2:torch",
            "//pytorch/vision:torchvision",
        ],
    )

    runtime.python_library(
        name = "train_utils",
        srcs = ["train_utils.py"],
        visibility = [],  # Private
        deps = [
            "//caffe2:torch",
            "fbsource//third-party/pypi/tqdm:tqdm",
            "//executorch/extension/pybindings:portable_lib",
            "//executorch/extension/training:lib",
        ],
    )

    runtime.python_binary(
        name = "model_export",
        srcs = ["export.py"],
        main_function = "executorch.extension.training.examples.CIFAR.export.main",
        deps = [
            ":model",
            ":data_utils",
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",
        ],
    )

    runtime.python_library(
        name = "export",
        srcs = ["export.py"],
        visibility = [],  # Private
        deps = [
            ":model",
            ":data_utils",
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",
        ],
    )

    runtime.python_binary(
        name = "main",
        srcs = ["main.py"],
        main_function = "executorch.extension.training.examples.CIFAR.main.main",
        deps = [
            ":model",
            ":data_utils",
            ":export",
            ":train_utils",
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
