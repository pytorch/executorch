load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/test:util.bzl", "codegen_function_header_wrapper", "op_test")

def _common_op_test(name, kernels):
    """
    Defines test targets in format of <kernel>_op_<op-name>_test
    For ATen kernel testing, let's use portable functions.yaml for tested ops.
    """
    for kernel in kernels:
        deps = [":function_header_wrapper_{}".format(kernel)]
        op_test(name, kernel_name = kernel, use_kernel_prefix = True, deps = deps)

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_kernel in (True, False):
        aten_suffix = "_aten" if aten_kernel else ""
        runtime.cxx_library(
            name = "gtest_utils" + aten_suffix,
            exported_headers=[
                "TestUtil.h",
            ],
            visibility = ["PUBLIC"],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_kernel else [],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/kernel:kernel_includes",
                "//executorch/test/utils:utils" + aten_suffix,
                "//executorch/runtime/platform:pal_interface",
            ],
            fbcode_exported_deps = [
                "//common/gtest:gtest",
            ] if not runtime.is_oss else [],
            xplat_exported_deps = [
                "//third-party/googletest:gtest_main",
            ],
        )
        runtime.cxx_library(
            name = "test_util" + aten_suffix,
            srcs = [
                "BinaryLogicalOpTest.cpp",
                "UnaryUfuncRealHBBF16ToFloatHBF16Test.cpp",
            ],
            exported_headers = [
                "BinaryLogicalOpTest.h",
                "ScalarOverflowTestMacros.h",
                "UnaryUfuncRealHBBF16ToFloatHBF16Test.h",
            ],
            visibility = ["PUBLIC"],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_kernel else [],
            exported_deps = [
                ":supported_features_header",
                ":gtest_utils",
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
                "//executorch/runtime/kernel:kernel_includes",
                "//executorch/test/utils:utils" + aten_suffix,
            ],
            fbcode_exported_deps = [
                "//common/init:init",
                "//common/gtest:gtest",
            ] if not runtime.is_oss else [],
            xplat_exported_deps = [
                "fbsource//xplat/folly/init:init",
                "//third-party/googletest:gtest_main",
            ],
        )

        runtime.cxx_test(
            name = "kernel_runtime_context_test" + aten_suffix,
            srcs = ["kernel_runtime_context_test.cpp"],
            deps = [
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
            ],
        )

    runtime.python_binary(
        name = "gen_supported_features",
        main_module = "executorch.kernels.test.gen_supported_features",
        deps = [
            ":gen_supported_features_lib",
        ],
        visibility = [
            "//executorch/kernels/...",
        ],
    )

    runtime.python_library(
        name = "gen_supported_features_lib",
        srcs = ["gen_supported_features.py"],
        resources = [
            "supported_features_header.ini",
            "supported_features_definition.ini",
        ],
        base_module = "executorch.kernels.test",
        visibility = ["//executorch/kernels/test/..."],
        deps = [
            "fbsource//third-party/pypi/setuptools:setuptools",
            "fbsource//third-party/pypi/pyyaml:pyyaml",
        ],
    )

    runtime.genrule(
        name = "supported_feature_header_gen",
        cmd = "$(exe //executorch/kernels/test:gen_supported_features) ${SRCS} > $OUT/supported_features.h",
        srcs = ["supported_features.yaml"],
        outs = {"supported_features.h": ["supported_features.h"]},
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "supported_features_header",
        srcs = [],
        exported_headers = {"supported_features.h": ":supported_feature_header_gen[supported_features.h]"},
        visibility = [
            "//executorch/kernels/...",
        ],
    )

    runtime.genrule(
        name = "supported_feature_aten_gen",
        cmd = "$(exe //executorch/kernels/test:gen_supported_features) ${SRCS} > $OUT/supported_features_aten.cpp",
        srcs = ["supported_features_def_aten.yaml"],
        outs = {"supported_features_aten.cpp": ["supported_features_aten.cpp"]},
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "supported_features_aten",
        srcs = [":supported_feature_aten_gen[supported_features_aten.cpp]"],
        visibility = [
            "//executorch/kernels/...",
        ],
        exported_deps = [
            "//executorch/kernels/test:supported_features_header",
        ],
    )

    TEST_SRCS = native.glob(["op_*_test.cpp"])

    runtime.filegroup(
        name = "test_srcs",
        srcs = TEST_SRCS,
        visibility = ["PUBLIC"],
    )

    runtime.genrule(
        name = "test_srcs_gen",
        srcs = [":test_srcs"],
        cmd = "cp $(location :test_srcs)/* $OUT",
        outs = {f: [f] for f in TEST_SRCS},
        default_outs = ["."],
        visibility = ["PUBLIC"],
    )

    codegen_function_header_wrapper("executorch/kernels/aten", "aten")
    codegen_function_header_wrapper("executorch/kernels/portable", "portable")
    codegen_function_header_wrapper("executorch/kernels/optimized", "optimized")
    codegen_function_header_wrapper("executorch/kernels/quantized", "quantized")
    codegen_function_header_wrapper("executorch/kernels/test/custom_kernel_example", "custom_kernel_example")

    _common_op_test("op__to_dim_order_copy_test", ["aten", "portable"])
    _common_op_test("op__empty_dim_order_test", ["aten", "portable"])
    _common_op_test("op__clone_dim_order_test", ["aten", "portable"])
    _common_op_test("op__conj_physical_test", ["aten", "portable"])
    _common_op_test("op_abs_test", ["aten", "portable"])
    _common_op_test("op_acos_test", ["aten", "portable"])
    _common_op_test("op_acosh_test", ["aten", "portable"])
    _common_op_test("op_add_test", ["aten", "portable", "optimized"])
    _common_op_test("op_addmm_test", ["aten", "portable"])
    _common_op_test("op_alias_copy_test", ["aten", "portable"])
    _common_op_test("op_amax_test", ["aten", "portable"])
    _common_op_test("op_amin_test", ["aten", "portable"])
    _common_op_test("op_any_test", ["aten", "portable"])
    _common_op_test("op_arange_test", ["aten", "portable"])
    _common_op_test("op_argmax_test", ["aten", "portable"])
    _common_op_test("op_argmin_test", ["aten", "portable"])
    _common_op_test("op_as_strided_copy_test", ["aten", "portable"])
    _common_op_test("op_asin_test", ["aten", "portable"])
    _common_op_test("op_asinh_test", ["aten", "portable"])
    _common_op_test("op_atan_test", ["aten", "portable"])
    _common_op_test("op_atan2_test", ["aten", "portable"])
    _common_op_test("op_atanh_test", ["aten", "portable"])
    _common_op_test("op_avg_pool2d_test", ["aten", "portable"])
    _common_op_test("op_bitwise_and_test", ["aten", "portable"])
    _common_op_test("op_bitwise_left_shift_test", ["portable"])
    _common_op_test("op_bitwise_not_test", ["aten", "portable"])
    _common_op_test("op_bitwise_or_test", ["aten", "portable"])
    _common_op_test("op_bitwise_right_shift_test", ["portable"])
    _common_op_test("op_bitwise_xor_test", ["aten", "portable"])
    _common_op_test("op_bmm_test", ["aten", "portable", "optimized"])
    _common_op_test("op_cat_test", ["aten", "portable"])
    _common_op_test("op_cdist_forward_test", ["aten", "portable"])
    _common_op_test("op_ceil_test", ["aten", "portable"])
    _common_op_test("op_clamp_test", ["aten", "portable"])
    _common_op_test("op_clone_test", ["aten", "portable"])
    _common_op_test("op_constant_pad_nd_test", ["aten", "portable"])
    _common_op_test("op_convolution_test", ["aten", "portable"])
    _common_op_test("op_convolution_backward_test", ["aten", "portable"])
    _common_op_test("op_copy_test", ["aten", "portable"])
    _common_op_test("op_cos_test", ["aten", "portable"])
    _common_op_test("op_cosh_test", ["aten", "portable"])
    _common_op_test("op_cumsum_test", ["aten", "portable"])
    _common_op_test("op_detach_copy_test", ["aten", "portable"])
    _common_op_test("op_diagonal_copy_test", ["aten", "portable"])
    _common_op_test("op_div_test", ["aten", "portable", "optimized"])
    _common_op_test("op_elu_test", ["aten", "portable", "optimized"])
    _common_op_test("op_embedding_test", ["aten", "portable"])
    _common_op_test("op_empty_test", ["aten", "portable"])
    _common_op_test("op_eq_test", ["aten", "portable"])
    _common_op_test("op_erf_test", ["aten", "portable"])
    _common_op_test("op_exp_test", ["aten", "portable", "optimized"])
    _common_op_test("op_expand_copy_test", ["aten", "portable"])
    _common_op_test("op_expm1_test", ["aten", "portable"])
    _common_op_test("op_fft_c2r_test", ["aten", "optimized"])
    _common_op_test("op_fft_r2c_test", ["aten", "optimized"])
    _common_op_test("op_fill_test", ["aten", "portable"])
    _common_op_test("op_flip_test", ["aten", "portable"])
    _common_op_test("op_floor_divide_test", ["aten", "portable"])
    _common_op_test("op_floor_test", ["aten", "portable"])
    _common_op_test("op_fmod_test", ["aten", "portable"])
    _common_op_test("op_full_like_test", ["aten", "portable"])
    _common_op_test("op_full_test", ["aten", "portable"])
    _common_op_test("op_gather_test", ["aten", "portable"])
    _common_op_test("op_ge_test", ["aten", "portable"])
    _common_op_test("op_gelu_test", ["aten", "portable", "optimized"])
    _common_op_test("op_glu_test", ["aten", "portable"])
    _common_op_test("op_gt_test", ["aten", "portable"])
    _common_op_test("op_hardtanh_test", ["aten", "portable"])
    _common_op_test("op_index_put_test", ["aten", "portable"])
    _common_op_test("op_index_select_test", ["aten", "portable"])
    _common_op_test("op_index_test", ["aten", "portable"])
    _common_op_test("op_isinf_test", ["aten", "portable"])
    _common_op_test("op_isnan_test", ["aten", "portable"])
    _common_op_test("op_le_test", ["aten", "portable", "optimized"])
    _common_op_test("op_leaky_relu_test", ["aten", "portable"])
    _common_op_test("op_lift_fresh_copy_test", ["aten", "portable"])
    _common_op_test("op_linear_test", ["aten", "optimized"])
    _common_op_test("op_log_softmax_test", ["aten", "portable", "optimized"])
    _common_op_test("op_log_test", ["aten", "portable"])
    _common_op_test("op_log10_test", ["aten", "portable"])
    _common_op_test("op_log1p_test", ["aten", "portable"])
    _common_op_test("op_log2_test", ["aten", "portable"])
    _common_op_test("op_logical_and_test", ["aten", "portable"])
    _common_op_test("op_logical_not_test", ["aten", "portable"])
    _common_op_test("op_logical_or_test", ["aten", "portable"])
    _common_op_test("op_logical_xor_test", ["aten", "portable"])
    _common_op_test("op_logit_test", ["aten", "portable"])
    _common_op_test("op_lt_test", ["aten", "portable"])
    _common_op_test("op_masked_fill_test", ["aten", "portable"])
    _common_op_test("op_masked_scatter_test", ["aten", "portable"])
    _common_op_test("op_masked_select_test", ["aten", "portable"])
    _common_op_test("op_max_test", ["aten", "portable"])
    _common_op_test("op_max_pool2d_with_indices_test", ["aten", "portable"])
    _common_op_test("op_max_pool2d_with_indices_backward_test", ["aten", "portable"])
    _common_op_test("op_maximum_test", ["aten", "portable"])
    _common_op_test("op_mean_test", ["aten", "portable"])
    _common_op_test("op_min_test", ["aten", "portable"])
    _common_op_test("op_minimum_test", ["aten", "portable"])
    _common_op_test("op_mm_test", ["aten", "portable", "optimized"])
    _common_op_test("op_mul_test", ["aten", "portable", "optimized"])
    _common_op_test("op_narrow_copy_test", ["aten", "portable"])
    _common_op_test("op_native_batch_norm_test", ["aten", "portable"])
    _common_op_test("op_native_dropout_test", ["aten", "portable"])
    _common_op_test("op_native_group_norm_test", ["aten", "portable"])
    _common_op_test("op_native_layer_norm_test", ["aten", "portable", "optimized"])
    _common_op_test("op_ne_test", ["aten", "portable"])
    _common_op_test("op_neg_test", ["aten", "portable"])
    _common_op_test("op_nonzero_test", ["aten", "portable"])
    _common_op_test("op_ones_test", ["aten", "portable"])
    _common_op_test("op_pdist_forward_test", ["aten", "portable"])
    _common_op_test("op_permute_copy_test", ["aten", "portable"])
    _common_op_test("op_pixel_shuffle_test", ["aten", "portable"])
    _common_op_test("op_pixel_unshuffle_test", ["aten", "portable"])
    _common_op_test("op_pow_test", ["aten", "portable"])
    _common_op_test("op_prod_test", ["aten", "portable"])
    _common_op_test("op_rand_test", ["aten", "portable"])
    _common_op_test("op_randn_test", ["aten", "portable"])
    _common_op_test("op_reciprocal_test", ["aten", "portable"])
    _common_op_test("op_relu_test", ["aten", "portable"])
    _common_op_test("op_remainder_test", ["aten", "portable"])
    _common_op_test("op_repeat_test", ["aten", "portable"])
    _common_op_test("op_repeat_interleave_test", ["aten", "portable"])
    _common_op_test("op_reflection_pad1d_test", ["aten", "portable"])
    _common_op_test("op_reflection_pad2d_test", ["aten", "portable"])
    _common_op_test("op_reflection_pad3d_test", ["aten", "portable"])
    _common_op_test("op_replication_pad1d_test", ["aten", "portable"])
    _common_op_test("op_replication_pad2d_test", ["aten", "portable"])
    _common_op_test("op_replication_pad3d_test", ["aten", "portable"])
    _common_op_test("op_roll_test", ["aten", "portable"])
    _common_op_test("op_round_test", ["aten", "portable"])
    _common_op_test("op_rsqrt_test", ["aten", "portable"])
    _common_op_test("op_rsub_test", ["aten", "portable"])
    _common_op_test("op_scalar_tensor_test", ["aten", "portable"])
    _common_op_test("op_scatter_test", ["aten", "portable"])
    _common_op_test("op_scatter_add_test", ["aten", "portable"])
    _common_op_test("op_select_scatter_test", ["aten", "portable"])
    _common_op_test("op_select_copy_test", ["aten", "portable"])
    _common_op_test("op_sigmoid_test", ["aten", "portable"])
    _common_op_test("op_sign_test", ["aten", "portable"])
    _common_op_test("op_sin_test", ["aten", "portable"])
    _common_op_test("op_sinh_test", ["aten", "portable"])
    _common_op_test("op_slice_scatter_test", ["aten", "portable"])
    _common_op_test("op_slice_copy_test", ["aten", "portable"])
    _common_op_test("op_softmax_test", ["aten", "portable"])
    _common_op_test("op_split_copy_test", ["aten", "portable"])
    _common_op_test("op_split_with_sizes_copy_test", ["aten", "portable"])
    _common_op_test("op_sqrt_test", ["aten", "portable"])
    _common_op_test("op_squeeze_copy_test", ["aten", "portable"])
    _common_op_test("op_stack_test", ["aten", "portable"])
    _common_op_test("op_sub_test", ["aten", "portable", "optimized"])
    _common_op_test("op_sum_test", ["aten", "portable"])
    _common_op_test("op_t_copy_test", ["aten", "portable"])
    _common_op_test("op_tan_test", ["aten", "portable"])
    _common_op_test("op_tanh_test", ["aten", "portable"])
    _common_op_test("op_to_copy_test", ["aten", "portable"])
    _common_op_test("op_topk_test", ["aten", "portable"])
    _common_op_test("op_transpose_copy_test", ["aten", "portable"])
    _common_op_test("op_tril_test", ["aten", "portable"])
    _common_op_test("op_trunc_test", ["aten", "portable"])
    _common_op_test("op_unbind_copy_test", ["aten", "portable"])
    _common_op_test("op_unfold_copy_test", ["aten", "portable"])
    _common_op_test("op_unsqueeze_copy_test", ["aten", "portable"])
    _common_op_test("op_upsample_bilinear2d_test", ["aten", "portable"])
    _common_op_test("op_upsample_bilinear2d_aa_test", ["portable"])
    _common_op_test("op_upsample_nearest2d_test", ["aten", "portable"])
    _common_op_test("op_var_test", ["aten", "portable"])
    _common_op_test("op_view_as_real_copy_test", ["aten", "portable"])
    _common_op_test("op_view_copy_test", ["aten", "portable"])
    _common_op_test("op_where_test", ["aten", "portable"])
    _common_op_test("op_zeros_test", ["aten", "portable"])
