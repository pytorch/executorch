load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "converters",
        srcs = [
            "__init__.py",
            "activations.py",
            "add.py",
            "addmm.py",
            "batch_norm.py",
            "bmm.py",
            "clamp.py",
            "comparison.py",
            "concat.py",
            "conv2d.py",
            "dim_order_ops.py",
            "div.py",
            "embedding.py",
            "expand.py",
            "getitem.py",
            "layer_norm.py",
            "linear.py",
            "mm.py",
            "mul.py",
            "permute_copy.py",
            "pixel_shuffle.py",
            "pooling.py",
            "reduction.py",
            "relu.py",
            "reshape.py",
            "sdpa.py",
            "slice.py",
            "sub.py",
            "upsample.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/backends/nvidia/tensorrt:converter_registry",
            "//executorch/backends/nvidia/tensorrt:converter_utils",
        ],
    )
