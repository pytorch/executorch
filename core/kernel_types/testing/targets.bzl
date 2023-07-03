load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        xplat_deps = ["//xplat/third-party/gmock:gmock"] + (["//xplat/caffe2:torch_mobile_all_ops"] if aten_mode else [])
        fbcode_deps = ["fbsource//third-party/googletest:gtest", "fbsource//third-party/googletest:gmock"] + (["//caffe2:libtorch"] if aten_mode else [])

        runtime.cxx_library(
            name = "tensor_util" + (aten_suffix),
            srcs = ["TensorUtil.cpp"],
            exported_headers = [
                "TensorUtil.h",
                "TensorFactory.h",
            ],
            visibility = [
                # Be strict with the visibility so that operator implementations
                # under //executorch/kernels/... can't depend on this test-only
                # target. It's ok to add any //executorch/*/test/... path to this
                # list.
                "//executorch/core/kernel_types/util/test/...",
                "//executorch/core/values/test/...",
                "//executorch/core/prim_ops/test/...",
                "//executorch/kernels/portable/test/...",
                "//executorch/kernels/portable/cpu/util/test/...",
                "//executorch/kernels/quantized/test/...",
                "//executorch/kernels/optimized/test/...",
                "//executorch/kernels/test/...",
                "//executorch/core/test/...",
                "//executorch/test/...",
                "//executorch/util/...",
                "//executorch/backends/qnnpack/test/...",
                "@EXECUTORCH_CLIENTS",
            ],
            compiler_flags = ["-Wno-unneeded-internal-declaration"],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/core/kernel_types:kernel_types" + aten_suffix,
                "//executorch/core/kernel_types/util:scalar_type_util" + aten_suffix,
                "//executorch/core/kernel_types/util:tensor_util" + aten_suffix,
            ],
            fbcode_exported_deps = fbcode_deps,
            xplat_exported_deps = xplat_deps,
        )

    runtime.cxx_test(
        name = "tensor_util_test",
        srcs = ["test/TensorUtilTest.cpp"],
        deps = [
            ":tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "tensor_factory_test",
        srcs = ["test/TensorFactoryTest.cpp"],
        deps = [
            ":tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "tensor_factory_test_aten",
        srcs = ["test/TensorFactoryTest.cpp"],
        preprocessor_flags = ["-DUSE_ATEN_LIB"],
        deps = [
            ":tensor_util_aten",
        ],
    )
