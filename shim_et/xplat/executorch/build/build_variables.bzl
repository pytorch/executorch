# WARNING: the contents of this file must BOTH be valid Starlark (for Buck) as well as
# valid Python (for our cmake build). This means that load() directives are not allowed
# (as they are not recognized by Python). If you want to fix this, figure out how run
# this file from cmake with a proper Starlark interpreter as part of the default OSS
# build process. If you need some nontrivial Starlark features, make a separate bzl
# file. (Remember that bzl files are not exported via ShipIt by default, so you may also
# need to update ExecuTorch's ShipIt config.)

# This file contains srcs lists that are shared between our Buck and CMake build
# systems. We had three choices for listing src files:
# 1) List them in Buck and use buck query to get them in CMake. This was our setup for a
# long time; the problem is that OSS users would prefer not to have to deal with Buck at
# all.
# 2) List them in both Buck targets.bzl files and CMake's CMakeLists.txt files. This is
# unnecessary duplication, and people will invariably forget to update one or the other.
# 3) List them somewhere CMake and Buck can both get at them; that's this file. Buck
# files can load() it, and our CMake build evaluates it with Python. (See
# executorch_append_filelist in build/Codegen.cmake.)
#
# Inconveniently, the Buck target layout is much more granular than the CMake library
# layout, leading to several complications:
# 1) Single-file Buck targets will just list the one src file they contain. Nothing to
# share with CMake in that case, and that src will be in a list in this file that does
# not map directly to that particular Buck target.
# 2) Multi-file Buck targets should have a list below that corresponds exactly to their
# `srcs`. There should then be simple Python code that combines those lists into lists
# that map 1:1 to the CMake library layout.

EXECUTORCH_SRCS = [
    "kernels/prim_ops/et_copy_index.cpp",
    "kernels/prim_ops/et_view.cpp",
    "kernels/prim_ops/register_prim_ops.cpp",
]

PROGRAM_NO_PRIM_OPS_SRCS = [
    "method.cpp",
    "method_meta.cpp",
    "program.cpp",
    "tensor_parser_exec_aten.cpp",
]

PLATFORM_SRCS = [
    "abort.cpp",
    "log.cpp",
    "platform.cpp",
    "profiler.cpp",
    "runtime.cpp",
]

EXECUTORCH_CORE_SRCS = sorted([
    "runtime/backend/interface.cpp",
    "runtime/core/evalue.cpp",
    "runtime/core/exec_aten/util/tensor_shape_to_c_string.cpp",
    "runtime/core/exec_aten/util/tensor_util_portable.cpp",
    "runtime/core/portable_type/tensor_impl.cpp",
    "runtime/core/tag.cpp",
    "runtime/core/tensor_layout.cpp",
    "runtime/executor/tensor_parser_portable.cpp",
    "runtime/executor/pte_data_map.cpp",
    "runtime/kernel/operator_registry.cpp",
    "schema/extended_header.cpp",
] + ["runtime/executor/" + x for x in PROGRAM_NO_PRIM_OPS_SRCS] + ["runtime/platform/" + x for x in PLATFORM_SRCS])

PATTERN_SRCS = [
    "unary_ufunc_realhbbf16_to_bool.cpp",
    "unary_ufunc_realhbbf16_to_floathbf16.cpp",
    "unary_ufunc_realhbf16.cpp",
]

PORTABLE_KERNELS_SRCS = [
    "kernels/portable/cpu/op__clone_dim_order.cpp",
    "kernels/portable/cpu/op__empty_dim_order.cpp",
    "kernels/portable/cpu/op__to_dim_order_copy.cpp",
    "kernels/portable/cpu/op_abs.cpp",
    "kernels/portable/cpu/op_acos.cpp",
    "kernels/portable/cpu/op_acosh.cpp",
    "kernels/portable/cpu/op_add.cpp",
    "kernels/portable/cpu/op_addmm.cpp",
    "kernels/portable/cpu/op_alias_copy.cpp",
    "kernels/portable/cpu/op_allclose.cpp",
    "kernels/portable/cpu/op_amax.cpp",
    "kernels/portable/cpu/op_amin.cpp",
    "kernels/portable/cpu/op_any.cpp",
    "kernels/portable/cpu/op_arange.cpp",
    "kernels/portable/cpu/op_argmax.cpp",
    "kernels/portable/cpu/op_argmin.cpp",
    "kernels/portable/cpu/op_as_strided_copy.cpp",
    "kernels/portable/cpu/op_asin.cpp",
    "kernels/portable/cpu/op_asinh.cpp",
    "kernels/portable/cpu/op_atan.cpp",
    "kernels/portable/cpu/op_atan2.cpp",
    "kernels/portable/cpu/op_atanh.cpp",
    "kernels/portable/cpu/op_avg_pool2d.cpp",
    "kernels/portable/cpu/op_bitwise_and.cpp",
    "kernels/portable/cpu/op_bitwise_not.cpp",
    "kernels/portable/cpu/op_bitwise_or.cpp",
    "kernels/portable/cpu/op_bitwise_xor.cpp",
    "kernels/portable/cpu/op_bmm.cpp",
    "kernels/portable/cpu/op_cat.cpp",
    "kernels/portable/cpu/op_cdist_forward.cpp",
    "kernels/portable/cpu/op_ceil.cpp",
    "kernels/portable/cpu/op_clamp.cpp",
    "kernels/portable/cpu/op_clone.cpp",
    "kernels/portable/cpu/op_constant_pad_nd.cpp",
    "kernels/portable/cpu/op_convolution.cpp",
    "kernels/portable/cpu/op_convolution_backward.cpp",
    "kernels/portable/cpu/op_copy.cpp",
    "kernels/portable/cpu/op_cos.cpp",
    "kernels/portable/cpu/op_cosh.cpp",
    "kernels/portable/cpu/op_cumsum.cpp",
    "kernels/portable/cpu/op_detach_copy.cpp",
    "kernels/portable/cpu/op_diagonal_copy.cpp",
    "kernels/portable/cpu/op_div.cpp",
    "kernels/portable/cpu/op_elu.cpp",
    "kernels/portable/cpu/op_embedding.cpp",
    "kernels/portable/cpu/op_empty.cpp",
    "kernels/portable/cpu/op_eq.cpp",
    "kernels/portable/cpu/op_erf.cpp",
    "kernels/portable/cpu/op_exp.cpp",
    "kernels/portable/cpu/op_expand_copy.cpp",
    "kernels/portable/cpu/op_expm1.cpp",
    "kernels/portable/cpu/op_fill.cpp",
    "kernels/portable/cpu/op_flip.cpp",
    "kernels/portable/cpu/op_floor.cpp",
    "kernels/portable/cpu/op_floor_divide.cpp",
    "kernels/portable/cpu/op_fmod.cpp",
    "kernels/portable/cpu/op_full.cpp",
    "kernels/portable/cpu/op_full_like.cpp",
    "kernels/portable/cpu/op_gather.cpp",
    "kernels/portable/cpu/op_ge.cpp",
    "kernels/portable/cpu/op_gelu.cpp",
    "kernels/portable/cpu/op_glu.cpp",
    "kernels/portable/cpu/op_gt.cpp",
    "kernels/portable/cpu/op_hardtanh.cpp",
    "kernels/portable/cpu/op_index.cpp",
    "kernels/portable/cpu/op_index_put.cpp",
    "kernels/portable/cpu/op_index_select.cpp",
    "kernels/portable/cpu/op_isinf.cpp",
    "kernels/portable/cpu/op_isnan.cpp",
    "kernels/portable/cpu/op_le.cpp",
    "kernels/portable/cpu/op_leaky_relu.cpp",
    "kernels/portable/cpu/op_lift_fresh_copy.cpp",
    "kernels/portable/cpu/op_log.cpp",
    "kernels/portable/cpu/op_log10.cpp",
    "kernels/portable/cpu/op_log1p.cpp",
    "kernels/portable/cpu/op_log2.cpp",
    "kernels/portable/cpu/op_log_softmax.cpp",
    "kernels/portable/cpu/op_logical_and.cpp",
    "kernels/portable/cpu/op_logical_not.cpp",
    "kernels/portable/cpu/op_logical_or.cpp",
    "kernels/portable/cpu/op_logical_xor.cpp",
    "kernels/portable/cpu/op_logit.cpp",
    "kernels/portable/cpu/op_lt.cpp",
    "kernels/portable/cpu/op_masked_fill.cpp",
    "kernels/portable/cpu/op_masked_scatter.cpp",
    "kernels/portable/cpu/op_masked_select.cpp",
    "kernels/portable/cpu/op_max.cpp",
    "kernels/portable/cpu/op_max_pool2d_with_indices.cpp",
    "kernels/portable/cpu/op_max_pool2d_with_indices_backward.cpp",
    "kernels/portable/cpu/op_maximum.cpp",
    "kernels/portable/cpu/op_mean.cpp",
    "kernels/portable/cpu/op_min.cpp",
    "kernels/portable/cpu/op_minimum.cpp",
    "kernels/portable/cpu/op_mm.cpp",
    "kernels/portable/cpu/op_mul.cpp",
    "kernels/portable/cpu/op_narrow_copy.cpp",
    "kernels/portable/cpu/op_native_batch_norm.cpp",
    "kernels/portable/cpu/op_native_dropout.cpp",
    "kernels/portable/cpu/op_native_group_norm.cpp",
    "kernels/portable/cpu/op_native_layer_norm.cpp",
    "kernels/portable/cpu/op_ne.cpp",
    "kernels/portable/cpu/op_neg.cpp",
    "kernels/portable/cpu/op_nonzero.cpp",
    "kernels/portable/cpu/op_ones.cpp",
    "kernels/portable/cpu/op_pdist_forward.cpp",
    "kernels/portable/cpu/op_permute_copy.cpp",
    "kernels/portable/cpu/op_pixel_shuffle.cpp",
    "kernels/portable/cpu/op_pixel_unshuffle.cpp",
    "kernels/portable/cpu/op_pow.cpp",
    "kernels/portable/cpu/op_prod.cpp",
    "kernels/portable/cpu/op_rand.cpp",
    "kernels/portable/cpu/op_randn.cpp",
    "kernels/portable/cpu/op_reciprocal.cpp",
    "kernels/portable/cpu/op_reflection_pad1d.cpp",
    "kernels/portable/cpu/op_reflection_pad2d.cpp",
    "kernels/portable/cpu/op_reflection_pad3d.cpp",
    "kernels/portable/cpu/op_relu.cpp",
    "kernels/portable/cpu/op_remainder.cpp",
    "kernels/portable/cpu/op_repeat.cpp",
    "kernels/portable/cpu/op_repeat_interleave.cpp",
    "kernels/portable/cpu/op_replication_pad1d.cpp",
    "kernels/portable/cpu/op_replication_pad2d.cpp",
    "kernels/portable/cpu/op_replication_pad3d.cpp",
    "kernels/portable/cpu/op_roll.cpp",
    "kernels/portable/cpu/op_round.cpp",
    "kernels/portable/cpu/op_rsqrt.cpp",
    "kernels/portable/cpu/op_rsub.cpp",
    "kernels/portable/cpu/op_scalar_tensor.cpp",
    "kernels/portable/cpu/op_scatter.cpp",
    "kernels/portable/cpu/op_scatter_add.cpp",
    "kernels/portable/cpu/op_select_copy.cpp",
    "kernels/portable/cpu/op_select_scatter.cpp",
    "kernels/portable/cpu/op_sigmoid.cpp",
    "kernels/portable/cpu/op_sign.cpp",
    "kernels/portable/cpu/op_sin.cpp",
    "kernels/portable/cpu/op_sinh.cpp",
    "kernels/portable/cpu/op_slice_copy.cpp",
    "kernels/portable/cpu/op_slice_scatter.cpp",
    "kernels/portable/cpu/op_softmax.cpp",
    "kernels/portable/cpu/op_split_copy.cpp",
    "kernels/portable/cpu/op_split_with_sizes_copy.cpp",
    "kernels/portable/cpu/op_sqrt.cpp",
    "kernels/portable/cpu/op_squeeze_copy.cpp",
    "kernels/portable/cpu/op_stack.cpp",
    "kernels/portable/cpu/op_sub.cpp",
    "kernels/portable/cpu/op_sum.cpp",
    "kernels/portable/cpu/op_t_copy.cpp",
    "kernels/portable/cpu/op_tan.cpp",
    "kernels/portable/cpu/op_tanh.cpp",
    "kernels/portable/cpu/op_to_copy.cpp",
    "kernels/portable/cpu/op_topk.cpp",
    "kernels/portable/cpu/op_transpose_copy.cpp",
    "kernels/portable/cpu/op_tril.cpp",
    "kernels/portable/cpu/op_trunc.cpp",
    "kernels/portable/cpu/op_unbind_copy.cpp",
    "kernels/portable/cpu/op_unfold_copy.cpp",
    "kernels/portable/cpu/op_unsqueeze_copy.cpp",
    "kernels/portable/cpu/op_upsample_bilinear2d.cpp",
    "kernels/portable/cpu/op_upsample_bilinear2d_aa.cpp",
    "kernels/portable/cpu/op_upsample_nearest2d.cpp",
    "kernels/portable/cpu/op_var.cpp",
    "kernels/portable/cpu/op_view_as_real_copy.cpp",
    "kernels/portable/cpu/op_view_copy.cpp",
    "kernels/portable/cpu/op_where.cpp",
    "kernels/portable/cpu/op_zeros.cpp",
] + ["kernels/portable/cpu/pattern/" + x for x in PATTERN_SRCS]

KERNELS_UTIL_ALL_DEPS_SRCS = [
    "kernels/portable/cpu/util/activation_ops_util.cpp",
    "kernels/portable/cpu/util/advanced_index_util.cpp",
    "kernels/portable/cpu/util/arange_util.cpp",
    "kernels/portable/cpu/util/broadcast_util.cpp",
    "kernels/portable/cpu/util/copy_ops_util.cpp",
    "kernels/portable/cpu/util/delinearize_index.cpp",
    "kernels/portable/cpu/util/distance_util.cpp",
    "kernels/portable/cpu/util/dtype_util.cpp",
    "kernels/portable/cpu/util/index_util.cpp",
    "kernels/portable/cpu/util/kernel_ops_util.cpp",
    "kernels/portable/cpu/util/matmul_ops_util.cpp",
    "kernels/portable/cpu/util/normalization_ops_util.cpp",
    "kernels/portable/cpu/util/padding_util.cpp",
    "kernels/portable/cpu/util/reduce_util.cpp",
    "kernels/portable/cpu/util/repeat_util.cpp",
    "kernels/portable/cpu/util/select_copy_util.cpp",
    "kernels/portable/cpu/util/slice_util.cpp",
    "kernels/portable/cpu/util/upsample_util.cpp",
]

OPTIMIZED_KERNELS_SRCS = [
    "kernels/optimized/cpu/binary_ops.cpp",
    "kernels/optimized/cpu/op_add.cpp",
    "kernels/optimized/cpu/op_bmm.cpp",
    "kernels/optimized/cpu/op_div.cpp",
    "kernels/optimized/cpu/op_elu.cpp",
    "kernels/optimized/cpu/op_exp.cpp",
    "kernels/optimized/cpu/op_fft_c2r.cpp",
    "kernels/optimized/cpu/op_fft_r2c.cpp",
    "kernels/optimized/cpu/op_gelu.cpp",
    "kernels/optimized/cpu/op_le.cpp",
    "kernels/optimized/cpu/op_linear.cpp",
    "kernels/optimized/cpu/op_log_softmax.cpp",
    "kernels/optimized/cpu/op_mm.cpp",
    "kernels/optimized/cpu/op_mul.cpp",
    "kernels/optimized/cpu/op_native_layer_norm.cpp",
    "kernels/optimized/cpu/op_sub.cpp",
    "kernels/optimized/cpu/op_where.cpp",
]

QUANTIZED_KERNELS_SRCS = [
    "kernels/quantized/cpu/embeddingxb.cpp",
    "kernels/quantized/cpu/op_add.cpp",
    "kernels/quantized/cpu/op_choose_qparams.cpp",
    "kernels/quantized/cpu/op_dequantize.cpp",
    "kernels/quantized/cpu/op_embedding.cpp",
    "kernels/quantized/cpu/op_embedding2b.cpp",
    "kernels/quantized/cpu/op_embedding4b.cpp",
    "kernels/quantized/cpu/op_mixed_linear.cpp",
    "kernels/quantized/cpu/op_mixed_mm.cpp",
    "kernels/quantized/cpu/op_quantize.cpp",
]

OPTIMIZED_CPUBLAS_SRCS = [
    "kernels/optimized/blas/BlasKernel.cpp",
    "kernels/optimized/blas/CPUBlas.cpp",
]

OPTIMIZED_NATIVE_CPU_OPS_SRCS = [
    "codegen/templates/RegisterCodegenUnboxedKernels.cpp",
    "codegen/templates/RegisterDispatchKeyCustomOps.cpp",
    "codegen/templates/RegisterKernels.cpp",
    "codegen/templates/RegisterSchema.cpp",
    "kernels/optimized/cpu/binary_ops.cpp",
    "kernels/optimized/cpu/op_add.cpp",
    "kernels/optimized/cpu/op_bmm.cpp",
    "kernels/optimized/cpu/op_div.cpp",
    "kernels/optimized/cpu/op_elu.cpp",
    "kernels/optimized/cpu/op_exp.cpp",
    "kernels/optimized/cpu/op_fft_c2r.cpp",
    "kernels/optimized/cpu/op_fft_r2c.cpp",
    "kernels/optimized/cpu/op_gelu.cpp",
    "kernels/optimized/cpu/op_le.cpp",
    "kernels/optimized/cpu/op_linear.cpp",
    "kernels/optimized/cpu/op_log_softmax.cpp",
    "kernels/optimized/cpu/op_mm.cpp",
    "kernels/optimized/cpu/op_mul.cpp",
    "kernels/optimized/cpu/op_native_layer_norm.cpp",
    "kernels/optimized/cpu/op_sub.cpp",
    "kernels/optimized/cpu/op_where.cpp",
]

TEST_BACKEND_COMPILER_LIB_SRCS = [
    "runtime/executor/test/test_backend_compiler_lib.cpp",
]

EXTENSION_DATA_LOADER_SRCS = [
    "extension/data_loader/file_data_loader.cpp",
    "extension/data_loader/mmap_data_loader.cpp",
]

EXTENSION_EVALUE_UTIL_SRCS = [
    "extension/evalue_util/print_evalue.cpp",
]

EXTENSION_FLAT_TENSOR_SRCS = [
    "extension/flat_tensor/flat_tensor_data_map.cpp",
    "extension/flat_tensor/serialize/flat_tensor_header.cpp",
]

EXTENSION_MEMORY_ALLOCATOR_SRCS = [
    "extension/memory_allocator/cpu_caching_malloc_allocator.cpp",
]

EXTENSION_MODULE_SRCS = [
    "extension/module/module.cpp",
]

EXTENSION_NAMED_DATA_MAP_SRCS = [
    "extension/named_data_map/merged_data_map.cpp",
]

EXTENSION_RUNNER_UTIL_SRCS = [
    "extension/runner_util/inputs.cpp",
    "extension/runner_util/inputs_portable.cpp",
]

EXTENSION_LLM_RUNNER_SRCS = [
    "extension/llm/runner/llm_runner_helper.cpp",
    "extension/llm/runner/multimodal_prefiller.cpp",
    "extension/llm/runner/multimodal_runner.cpp",
    "extension/llm/runner/text_decoder_runner.cpp",
    "extension/llm/runner/text_llm_runner.cpp",
    "extension/llm/runner/text_prefiller.cpp",
    "extension/llm/sampler/sampler.cpp",
]

EXTENSION_TENSOR_SRCS = [
    "extension/tensor/tensor_ptr.cpp",
    "extension/tensor/tensor_ptr_maker.cpp",
]

THREADPOOL_SRCS = [
    "thread_parallel.cpp",
    "threadpool.cpp",
    "threadpool_guard.cpp",
]

EXTENSION_THREADPOOL_SRCS = ["extension/threadpool/" + x for x in THREADPOOL_SRCS]

EXTENSION_TRAINING_SRCS = [
    "extension/training/module/training_module.cpp",
    "extension/training/optimizer/sgd.cpp",
]

TRAIN_XOR_SRCS = [
    # REVIEW: removing this breaks the build; where is it supposed to come from?
    "extension/flat_tensor/serialize/serialize.cpp",
    "extension/training/examples/XOR/train.cpp",
]

EXECUTOR_RUNNER_SRCS = [
    "examples/portable/executor_runner/executor_runner.cpp",
    "extension/data_loader/file_data_loader.cpp",
    "runtime/executor/test/test_backend_compiler_lib.cpp",
]

SIZE_TEST_SRCS = [
    "test/size_test.cpp",
]

MPS_EXECUTOR_RUNNER_SRCS = [
    "backends/apple/mps/runtime/MPSBackend.mm",
    "backends/apple/mps/runtime/MPSCompiler.mm",
    "backends/apple/mps/runtime/MPSDelegateHeader.mm",
    "backends/apple/mps/runtime/MPSDevice.mm",
    "backends/apple/mps/runtime/MPSExecutor.mm",
    "backends/apple/mps/runtime/MPSGraphBuilder.mm",
    "backends/apple/mps/runtime/MPSStream.mm",
    "backends/apple/mps/runtime/operations/ActivationOps.mm",
    "backends/apple/mps/runtime/operations/BinaryOps.mm",
    "backends/apple/mps/runtime/operations/ClampOps.mm",
    "backends/apple/mps/runtime/operations/ConstantOps.mm",
    "backends/apple/mps/runtime/operations/ConvolutionOps.mm",
    "backends/apple/mps/runtime/operations/IndexingOps.mm",
    "backends/apple/mps/runtime/operations/LinearAlgebra.mm",
    "backends/apple/mps/runtime/operations/NormalizationOps.mm",
    "backends/apple/mps/runtime/operations/OperationUtils.mm",
    "backends/apple/mps/runtime/operations/PadOps.mm",
    "backends/apple/mps/runtime/operations/PoolingOps.mm",
    "backends/apple/mps/runtime/operations/QuantDequant.mm",
    "backends/apple/mps/runtime/operations/RangeOps.mm",
    "backends/apple/mps/runtime/operations/ReduceOps.mm",
    "backends/apple/mps/runtime/operations/ShapeOps.mm",
    "backends/apple/mps/runtime/operations/UnaryOps.mm",
    "devtools/bundled_program/bundled_program.cpp",
    "devtools/etdump/data_sinks/buffer_data_sink.cpp",
    "devtools/etdump/emitter.cpp",
    "devtools/etdump/etdump_flatcc.cpp",
    "examples/apple/mps/executor_runner/mps_executor_runner.mm",
    "extension/data_loader/file_data_loader.cpp",
]

MPS_BACKEND_BUCK_SRCS = [
    "runtime/MPSBackend.mm",
    "runtime/MPSCompiler.mm",
    "runtime/MPSDelegateHeader.mm",
    "runtime/MPSDevice.mm",
    "runtime/MPSExecutor.mm",
    "runtime/MPSGraphBuilder.mm",
    "runtime/MPSStream.mm",
    "runtime/operations/ActivationOps.mm",
    "runtime/operations/BinaryOps.mm",
    "runtime/operations/ClampOps.mm",
    "runtime/operations/ConstantOps.mm",
    "runtime/operations/ConvolutionOps.mm",
    "runtime/operations/IndexingOps.mm",
    "runtime/operations/LinearAlgebra.mm",
    "runtime/operations/NormalizationOps.mm",
    "runtime/operations/OperationUtils.mm",
    "runtime/operations/PadOps.mm",
    "runtime/operations/PoolingOps.mm",
    "runtime/operations/QuantDequant.mm",
    "runtime/operations/RangeOps.mm",
    "runtime/operations/ReduceOps.mm",
    "runtime/operations/ShapeOps.mm",
    "runtime/operations/UnaryOps.mm",
]

MPS_BACKEND_SRCS = ["backends/apple/mps/" + x for x in MPS_BACKEND_BUCK_SRCS]

MPS_SCHEMA_SRCS = [
    "backends/apple/mps/serialization/schema.fbs",
]

XNN_EXECUTOR_RUNNER_SRCS = [
    "examples/portable/executor_runner/executor_runner.cpp",
    "extension/data_loader/file_data_loader.cpp",
]

XNNPACK_BACKEND_BUCK_SRCS = [
    "runtime/XNNCompiler.cpp",
    "runtime/XNNExecutor.cpp",
    "runtime/XNNHeader.cpp",
    "runtime/XNNPACKBackend.cpp",
    "runtime/XNNWeightsCache.cpp",
    "runtime/XNNWorkspaceManager.cpp",
    "runtime/profiling/XNNProfiler.cpp",
]

XNNPACK_BACKEND_SRCS = ["backends/xnnpack/" + x for x in XNNPACK_BACKEND_BUCK_SRCS]

XNNPACK_SCHEMA_SRCS = [
    "backends/xnnpack/serialization/runtime_schema.fbs",
]

VULKAN_SCHEMA_SRCS = [
    "backends/vulkan/serialization/schema.fbs",
]

EXTENSION_LLM_CUSTOM_OPS_BUCK_SRCS = [
    "op_fallback.cpp",
    "op_fast_hadamard_transform.cpp",
    "op_sdpa.cpp",
    "op_update_cache.cpp",
]

CUSTOM_OPS_SRCS = ["extension/llm/custom_ops/" + x for x in EXTENSION_LLM_CUSTOM_OPS_BUCK_SRCS] + [
    "extension/llm/custom_ops/spinquant/fast_hadamard_transform.cpp",
]

LLAMA_RUNNER_SRCS = [
    "examples/models/llama/runner/runner.cpp",
    "examples/models/llama/tokenizer/llama_tiktoken.cpp",
]
