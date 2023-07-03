load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "tensor_shape_dynamism",
        exported_headers = [
            "TensorShapeDynamism.h",
        ],
        visibility = [
            "//executorch/core/kernel_types/...",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        # Depend on this target if your types (Tensor, ArrayRef, etc) should be flexible between ATen and executor
        runtime.cxx_library(
            name = "kernel_types" + aten_suffix,
            exported_headers = ["kernel_types.h"],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            # Visible because clients may want to build ATen-specific versions
            # of their custom operators, to load into local PyTorch using
            # `torch.ops.load_library()`. See codegen.bzl.
            visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
            exported_deps = [":tensor_shape_dynamism"] + ([] if aten_mode else ["//executorch/core/kernel_types/lean:kernel_types"]),
            fbcode_exported_deps = ["//caffe2:torch-cpp"] if aten_mode else [],
            xplat_exported_deps = ["//xplat/caffe2:torch_mobile_core"] if aten_mode else [],
        )
