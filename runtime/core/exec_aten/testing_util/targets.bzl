load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "tensor_util" + (aten_suffix),
            srcs = ["tensor_util.cpp"],
            exported_headers = [
                "tensor_util.h",
                "tensor_factory.h",
            ],
            visibility = [
                # Be strict with the visibility so that operator implementations
                # under //executorch/kernels/... can't depend on this test-only
                # target. It's ok to add any //executorch/*/test/... path to this
                # list.
                "//executorch/runtime/core/exec_aten/util/test/...",
                "//executorch/runtime/core/exec_aten/testing_util/test/...",
                "//executorch/runtime/core/portable_type/test/...",
                "//executorch/kernels/prim_ops/test/...",
                "//executorch/kernels/portable/test/...",
                "//executorch/kernels/portable/cpu/util/test/...",
                "//executorch/kernels/quantized/test/...",
                "//executorch/kernels/optimized/test/...",
                "//executorch/kernels/test/...",
                "//executorch/kernels/fb/custom_ops/...",
                "//executorch/runtime/core/test/...",
                "//executorch/test/...",
                "//executorch/backends/fb/qnnpack/test/...",
                "//executorch/extension/kernel_util/test/...",
                "@EXECUTORCH_CLIENTS",
            ],
            compiler_flags = ["-Wno-unneeded-internal-declaration"],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
            exported_external_deps = [
                "gmock" + aten_suffix,
            ] + (["libtorch"] if aten_mode else []),
        )
