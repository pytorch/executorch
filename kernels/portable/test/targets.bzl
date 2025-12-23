load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
load("@fbsource//xplat/executorch/kernels/test:util.bzl", "define_supported_features_lib", "op_test")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    if not runtime.is_oss:
        define_supported_features_lib()

        if not is_xplat():
            python_unittest(
                name = "op_upsample_bilinear2d_test",
                srcs = [
                    "op_upsample_bilinear2d_test.py",
                ],
                preload_deps = [
                    ":aot_ops_test_lib",
                ],
                deps = [
                    "//caffe2:torch",
                ],
            )

            python_unittest(
                name = "op_upsample_bilinear2d_aa_test",
                srcs = [
                    "op_upsample_bilinear2d_aa_test.py",
                ],
                preload_deps = [
                    ":aot_ops_test_lib",
                ],
                deps = [
                    "//caffe2:torch",
                ],
            )

            python_unittest(
                name = "op_upsample_nearest2d_test",
                srcs = [
                    "op_upsample_nearest2d_test.py",
                ],
                preload_deps = [
                    ":aot_ops_test_lib",
                ],
                deps = [
                    "//caffe2:torch",
                ],
            )

            python_unittest(
                name = "op_grid_sampler_2d_test",
                srcs = [
                    "op_grid_sampler_2d_test.py",
                ],
                preload_deps = [
                    ":aot_ops_test_lib",
                ],
                deps = [
                    "//caffe2:torch",
                ],
            )

        op_test(name = "op_allclose_test")
        op_test(name = "op_div_test")
        op_test(name = "op_gelu_test")
        op_test(name = "op_mul_test")

    if is_xplat():
        et_operator_library(
            name = "add_float",
            ops_dict = {
                "aten::add.out": ["v1/6;0,1"],  # float
            },
        )

        executorch_generated_lib(
            name = "add_float_lib",
            functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
            kernel_deps = [
                "//executorch/kernels/portable:operators",
            ],
            visibility = ["//executorch/..."],
            deps = [
                ":add_float",
            ],
            dtype_selective_build = True,
        )

        runtime.cxx_test(
            name = "dtype_selective_build_test",
            srcs = ["dtype_selective_build_test.cpp"],
            deps = [
                ":add_float_lib",
                "//executorch/kernels/portable/cpu:scalar_utils",
            ],
        )
