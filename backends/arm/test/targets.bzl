# load("//caffe2/test/fb:defs.bzl", "define_tests")
load("@fbcode_macros//build_defs:python_pytest.bzl", "python_pytest")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

_ENABLE_VGF = False  # Disabled: memfd_create blocked by seccomp on Sandcastle causes segfaults before Python pre-flight check can run

def define_arm_tests():
    # TODO [fbonly] Add more tests
    test_files = []

    # Passes
    test_files += native.glob(["passes/test_*.py"])
    # https://github.com/pytorch/executorch/issues/8606
    test_files.remove("passes/test_ioquantization_pass.py")

    # Operators
    test_files += [
        "ops/test_abs.py",
        "ops/test_add.py",
        "ops/test_addmm.py",
        "ops/test_amax.py",
        "ops/test_amin.py",
        "ops/test_avg_pool2d.py",
        # "ops/test_batch_norm.py",  # T000000 FVP reshape fails: dtype mismatch in output parsing (expected [3], got 12 elements)
        "ops/test_bitwise.py",
        "ops/test_bmm.py",
        "ops/test_cat.py",
        "ops/test_ceil.py",
        "ops/test_clamp.py",
        "ops/test_clone.py",
        "ops/test_conv1d.py",
        "ops/test_conv2d.py",
        "ops/test_conv3d.py",
        "ops/test_cos.py",
        # "ops/test_depthwise_conv.py",  # T000000 Cross-file imports from test_conv1d/test_conv2d break Buck target listing
        # "ops/test_div.py",  # 15 failures: NoneType input in bundled_program serialization (pre-existing bug)
        "ops/test_elu.py",
        "ops/test_embedding.py",
        "ops/test_eq.py",
        "ops/test_erf.py",
        "ops/test_exp.py",
        "ops/test_expand.py",
        "ops/test_floor.py",
        "ops/test_full.py",
        "ops/test_ge.py",
        "ops/test_group_norm.py",
        "ops/test_gt.py",
        "ops/test_hardswish.py",
        "ops/test_hardtanh.py",
        # "ops/test_layer_norm.py",  # 1 failure: 16a8w u55 quantization issue
        "ops/test_le.py",
        "ops/test_leaky_relu.py",
        "ops/test_linear.py",
        "ops/test_log10.py",
        "ops/test_log.py",
        "ops/test_logical.py",
        "ops/test_lt.py",
        # matmul: Vela compilation fails with 'Non-passthrough operation'
        # for int16 matmul operations, xfail interacts incorrectly with
        # XfailIfNoCorstone.
        # "ops/test_matmul.py",
        "ops/test_max_pool1d.py",
        "ops/test_max_pool.py",
        "ops/test_mean_dim.py",
        "ops/test_maximum.py",
        "ops/test_minimum.py",
        "ops/test_mm.py",
        "ops/test_mul.py",
        "ops/test_ne.py",
        "ops/test_neg.py",
        "ops/test_ones.py",
        "ops/test_permute.py",
        "ops/test_pixel_shuffling.py",
        "ops/test_pow.py",
        "ops/test_reciprocal.py",
        "ops/test_relu.py",
        "ops/test_remainder.py",
        "ops/test_repeat.py",
        "ops/test_round.py",
        # rshift: U55 FVP output dtype mismatch for rshift operations,
        # xfail interacts incorrectly with XfailIfNoCorstone.
        # "ops/test_rshift.py",
        "ops/test_rsqrt.py",
        "ops/test_rsub.py",
        "ops/test_scalar_tensor.py",
        "ops/test_scalars.py",
        "ops/test_sdpa.py",
        "ops/test_select.py",
        "ops/test_select_scatter.py",
        "ops/test_sigmoid.py",
        "ops/test_sigmoid_32bit.py",
        "ops/test_sign.py",
        # silu: U55 numerical mismatch for inplace SiLU, xfail interacts
        # incorrectly with XfailIfNoCorstone.
        # "ops/test_silu.py",
        "ops/test_sin.py",
        "ops/test_sinh.py",
        "ops/test_slice.py",
        # slice_scatter: U55/U85 FVP failures for int8 and end_none cases,
        # xfail interacts incorrectly with XfailIfNoCorstone.
        # "ops/test_slice_scatter.py",
        "ops/test_softmax.py",
        "ops/test_split.py",
        "ops/test_sqrt.py",
        "ops/test_squeeze.py",
        "ops/test_stack.py",
        "ops/test_sub.py",
        # sum: xfail markers interact incorrectly with XfailIfNoCorstone
        # when test inputs contain None (NoneType not supported in
        # bundled_program serialization).
        # "ops/test_sum.py",
        "ops/test_t_copy.py",
        "ops/test_tan.py",
        "ops/test_tanh.py",
        "ops/test_to_copy.py",
        # transpose_conv2d: xfail markers for per-channel quantization
        # interact incorrectly with XfailIfNoCorstone when tests fail
        # during quantization (before reaching FVP execution).
        # "ops/test_transpose_conv2d.py",
        "ops/test_transpose_copy.py",
        "ops/test_tril.py",
        # "ops/test_unary_combos.py",  # 1 failure: NegAdd u55_INT
        "ops/test_unbind.py",
        "ops/test_unflatten.py",
        "ops/test_unfold_copy.py",
        "ops/test_unsqueeze.py",
        "ops/test_upsample_bilinear2d.py",
        "ops/test_upsample_nearest2d.py",
        "ops/test_var.py",
        "ops/test_view.py",
        "ops/test_where.py",
        "ops/test_while.py",
        "ops/test_zeros.py",
    ]

    # Quantization
    test_files += [
        "quantizer/test_conv_relu_fusing.py",
        "quantizer/test_generic_annotater.py",
        "quantizer/test_partial_quantization.py",
        "quantizer/test_preserve_kwargs.py",
        # "quantizer/test_selective_quantization.py",  # needs torchvision (not in deps)
        "quantizer/test_set_module_name.py",
        "quantizer/test_tosa_quantizer_validate.py",
    ]

    # Misc tests
    test_files += [
        "misc/test_compile_spec.py",
        # "misc/test_evaluate_model.py",
        "misc/test_pass_pipeline_config.py",
        "misc/test_tosa_spec.py",
        "misc/test_bn_relu_folding_qat.py",
        "misc/test_call_operator_submodule.py",
        "misc/test_compile_spec.py",
        "misc/test_const_shape.py",
        "misc/test_conv_relu_residual_add.py",
        "misc/test_count_tosa_ops.py",
        "misc/test_custom_partition.py",
        "misc/test_debug_feats.py",
        "misc/test_debug_hook.py",
        "misc/test_post_quant_device_switch.py",
        # "misc/test_dim_order.py", (TODO - T238390249)
    ]

    # Deprecation tests
    test_files += [
        "misc/test_dw_convs_with_shared_weights.py",
        "misc/test_extract_io_params_tosa.py",
        # "misc/test_int64.py",  # 5 failures: int64 overflow/chain handling issues
        "misc/test_lifted_tensor.py",
        "misc/test_mixed_fp_bf16_partition.py",
        "misc/test_mixed_type_lowering.py",
        # "misc/test_model_evaluator.py",  # needs executorch.backends.arm.util (no BUCK target)
        "misc/test_multiple_delegates.py",
        "misc/test_multiple_outputs.py",
        "misc/test_non_persistent_buffers.py",
        "deprecation/test_arm_compile_spec_deprecation.py",
        "misc/test_partition_decomposed_quantized_ops.py",
        "misc/test_pass_pipeline_config.py",
        "misc/test_pass_required_order.py",
        "misc/test_qat_training_loop.py",
        "misc/test_quant_custom_meta.py",
        # "misc/test_rescale_range.py",  # 3 failures: zero-point and rescale range validation
        # "misc/test_runner_utils.py",  # name collides with runner_utils library in BUCK
        "misc/test_save_exported_model.py",
        # "misc/test_shared_qspecs.py",  # needs executorch.backends.cortex_m.test.tester (no BUCK target)
        "misc/test_tosa_dialect_conv2d.py",
        "misc/test_tosa_dialect_dw_conv2d.py",
        "misc/test_tosa_dialect_shape_ops.py",
        "misc/test_tosa_spec.py",
    ]

    TESTS = {}

    for test_file in test_files:
        test_file_name = paths.basename(test_file)
        test_name = test_file_name.replace("test_", "").replace(".py", "")

        python_pytest(
            name = test_name,
            srcs = [test_file],
            pytest_config = "pytest.ini",
            resources = ["conftest.py"],
            compile = "with-source",
            typing = False,
            skip_on_mode_mac = True,
            env = {} if runtime.is_oss else ({
                "MODEL_CONVERTER_PATH": "$(location fbsource//third-party/pypi/ai-ml-sdk-model-converter/0.8.0:model-converter-bin)",
                "MODEL_CONVERTER_LIB_DIR": "$(location fbsource//third-party/nvidia-nsight-systems:linux-x86_64)/host-linux-x64",
                "LAVAPIPE_LIB_PATH": "$(location fbsource//third-party/mesa:vulkan_lvp)",
                "EMULATION_LAYER_TENSOR_SO": "$(location fbsource//third-party/arm-ml-emulation-layer/v0.9.0/src:libVkLayer_Tensor)",
                "EMULATION_LAYER_GRAPH_SO": "$(location fbsource//third-party/arm-ml-emulation-layer/v0.9.0/src:libVkLayer_Graph)",
                "EMULATION_LAYER_TENSOR_JSON": "$(location fbsource//third-party/arm-ml-emulation-layer/v0.9.0/src:VkLayer_Tensor_json)",
                "EMULATION_LAYER_GRAPH_JSON": "$(location fbsource//third-party/arm-ml-emulation-layer/v0.9.0/src:VkLayer_Graph_json)",
            } if _ENABLE_VGF else {}),
            preload_deps = [] if runtime.is_oss or not _ENABLE_VGF else [
                "//executorch/kernels/quantized:custom_ops_generated_lib",
                "fbsource//third-party/khronos:vulkan",
                "//executorch/backends/arm/runtime:vgf_backend",
            ],
            deps = [
                "//executorch/backends/arm/test:arm_tester" if runtime.is_oss else "//executorch/backends/arm/test/tester/fb:arm_tester_fb",
                "//executorch/backends/arm/test:conftest",
                "//executorch/backends/arm/test/misc:dw_convs_shared_weights_module",
                "//executorch/backends/arm:ethosu",
                "//executorch/backends/arm/tosa:compile_spec",
                "//executorch/backends/arm/tosa:partitioner",
                "//executorch/backends/arm:vgf",
                "//executorch/backends/test:graph_builder",
                "//executorch/backends/test/harness:tester",
                "//executorch/exir:lib",
                "fbsource//third-party/pypi/pytest:pytest",
                "fbsource//third-party/pypi/parameterized:parameterized",
                "fbsource//third-party/tosa_tools:tosa_reference_model",
            ],
        )
