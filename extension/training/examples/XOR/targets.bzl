load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_binary(
        name = "train_xor",
        srcs = ["train.cpp"],
        deps = [
            "//executorch/extension/training/module:training_module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/training/optimizer:sgd",
            "//executorch/runtime/executor:program",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/kernels/portable:generated_lib",
        ],
        external_deps = ["gflags"],
        define_static_target = True,
    )

    runtime.python_library(
        name = "model",
        srcs = ["model.py"],
        visibility = [],  # Private
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.python_library(
        name = "export_model_lib",
        srcs = ["export_model.py"],
        visibility = ["//executorch/extension/training/examples/XOR/..."],
        deps = [
            ":model",
            "//caffe2:torch",
            "//executorch/exir:lib",
        ],
    )

    runtime.python_binary(
        name = "export_model",
        main_module = "executorch.extension.training.examples.XOR.export_model",
        deps = [
            ":export_model_lib",
        ],
    )
