# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(kernels_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_kernels portable optimized)

foreach(kernel ${_kernels})
  set(_wrapper_dir
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/test"
  )
  set(_wrapper_path "${_wrapper_dir}/FunctionHeaderWrapper.h")
  add_custom_command(
    OUTPUT "${_wrapper_path}"
    COMMAND mkdir -p ${_wrapper_dir}
    COMMAND echo "#include <executorch/kernels/${kernel}/Functions.h>" >
            "${_wrapper_path}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating ${_wrapper_path}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${_wrapper_dir}/supported_features.cpp"
           "${_wrapper_dir}/supported_features.h"
    COMMAND mkdir -p ${_wrapper_dir}
    COMMAND
      python kernels/test/gen_supported_features.py
      kernels/${kernel}/test/supported_features_def.yaml >
      ${_wrapper_dir}/supported_features.cpp
    COMMAND
      python kernels/test/gen_supported_features.py
      kernels/test/supported_features.yaml >
      ${_wrapper_dir}/supported_features.h
    WORKING_DIRECTORY "${EXECUTORCH_ROOT}"
    COMMENT "Generating ${_wrapper_dir}/supported_features.cpp and header"
    VERBATIM
  )
endforeach()

add_custom_target(
  generate_wrapper
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/FunctionHeaderWrapper.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/supported_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/supported_features.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized/executorch/kernels/test/FunctionHeaderWrapper.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized/executorch/kernels/test/supported_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized/executorch/kernels/test/supported_features.cpp"
)

set(all_test_sources
    "op__to_dim_order_copy_test.cpp"
    "op_abs_test.cpp"
    "op_acos_test.cpp"
    "op_acosh_test.cpp"
    "op_add_test.cpp"
    "op_addmm_test.cpp"
    "op_alias_copy_test.cpp"
    "op_amax_test.cpp"
    "op_amin_test.cpp"
    "op_any_test.cpp"
    "op_arange_test.cpp"
    "op_argmax_test.cpp"
    "op_argmin_test.cpp"
    "op_as_strided_copy_test.cpp"
    "op_asin_test.cpp"
    "op_asinh_test.cpp"
    "op_atan_test.cpp"
    "op_atan2_test.cpp"
    "op_atanh_test.cpp"
    "op_avg_pool2d_test.cpp"
    "op_bitwise_and_test.cpp"
    "op_bitwise_not_test.cpp"
    "op_bitwise_or_test.cpp"
    "op_bitwise_xor_test.cpp"
    "op_bmm_test.cpp"
    "op_cat_test.cpp"
    "op_cdist_forward_test.cpp"
    "op_ceil_test.cpp"
    "op_clamp_test.cpp"
    "op_clone_test.cpp"
    "op_constant_pad_nd_test.cpp"
    "op_convolution_test.cpp"
    "op_copy_test.cpp"
    "op_cos_test.cpp"
    "op_cosh_test.cpp"
    "op_cumsum_test.cpp"
    "op_detach_copy_test.cpp"
    "op_diagonal_copy_test.cpp"
    "op_div_test.cpp"
    "op_embedding_test.cpp"
    "op_empty_test.cpp"
    "op_eq_test.cpp"
    "op_erf_test.cpp"
    "op_exp_test.cpp"
    "op_expand_copy_test.cpp"
    "op_expm1_test.cpp"
    "op_fill_test.cpp"
    "op_flip_test.cpp"
    "op_floor_divide_test.cpp"
    "op_floor_test.cpp"
    "op_fmod_test.cpp"
    "op_full_like_test.cpp"
    "op_full_test.cpp"
    "op_ge_test.cpp"
    "op_gelu_test.cpp"
    "op_glu_test.cpp"
    "op_gt_test.cpp"
    "op_hardtanh_test.cpp"
    "op_index_put_test.cpp"
    "op_index_select_test.cpp"
    "op_index_test.cpp"
    "op_isinf_test.cpp"
    "op_isnan_test.cpp"
    "op_le_test.cpp"
    "op_leaky_relu_test.cpp"
    "op_lift_fresh_copy_test.cpp"
    "op_log_softmax_test.cpp"
    "op_log_test.cpp"
    "op_log10_test.cpp"
    "op_log1p_test.cpp"
    "op_log2_test.cpp"
    "op_logical_and_test.cpp"
    "op_logical_not_test.cpp"
    "op_logical_or_test.cpp"
    "op_logical_xor_test.cpp"
    "op_logit_test.cpp"
    "op_lt_test.cpp"
    "op_masked_fill_test.cpp"
    "op_max_test.cpp"
    "op_max_pool2d_with_indices_test.cpp"
    "op_maximum_test.cpp"
    "op_mean_test.cpp"
    "op_min_test.cpp"
    "op_minimum_test.cpp"
    "op_mm_test.cpp"
    "op_mul_test.cpp"
    "op_pow_test.cpp"
    "op_native_batch_norm_test.cpp"
    "op_native_group_norm_test.cpp"
    "op_native_layer_norm_test.cpp"
    "op_ne_test.cpp"
    "op_neg_test.cpp"
    "op_nonzero_test.cpp"
    "op_ones_test.cpp"
    "op_pdist_forward_test.cpp"
    "op_permute_copy_test.cpp"
    "op_pixel_shuffle_test.cpp"
    "op_prod_test.cpp"
    "op_reciprocal_test.cpp"
    "op_relu_test.cpp"
    "op_remainder_test.cpp"
    "op_repeat_test.cpp"
    "op_reflection_pad1d_test.cpp"
    "op_reflection_pad2d_test.cpp"
    "op_reflection_pad3d_test.cpp"
    "op_replication_pad1d_test.cpp"
    "op_replication_pad2d_test.cpp"
    "op_replication_pad3d_test.cpp"
    "op_roll_test.cpp"
    "op_round_test.cpp"
    "op_rsqrt_test.cpp"
    "op_rsub_test.cpp"
    "op_scalar_tensor_test.cpp"
    "op_scatter_add_test.cpp"
    "op_select_scatter_test.cpp"
    "op_select_copy_test.cpp"
    "op_sigmoid_test.cpp"
    "op_sign_test.cpp"
    "op_sin_test.cpp"
    "op_sinh_test.cpp"
    "op_slice_scatter_test.cpp"
    "op_slice_copy_test.cpp"
    "op_softmax_test.cpp"
    "op_split_copy_test.cpp"
    "op_split_with_sizes_copy_test.cpp"
    "op_sqrt_test.cpp"
    "op_squeeze_copy_test.cpp"
    "op_stack_test.cpp"
    "op_sub_test.cpp"
    "op_sum_test.cpp"
    "op_t_copy_test.cpp"
    "op_tan_test.cpp"
    "op_tanh_test.cpp"
    "op_to_copy_test.cpp"
    "op_transpose_copy_test.cpp"
    "op_tril_test.cpp"
    "op_trunc_test.cpp"
    "op_unbind_copy_test.cpp"
    "op_unsqueeze_copy_test.cpp"
    "op_var_test.cpp"
    "op_view_copy_test.cpp"
    "op_where_test.cpp"
    "op_zeros_test.cpp"
)

set(_portable_kernels_test_sources
    ${all_test_sources}
    ${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/supported_features.cpp
)

et_cxx_test(
  portable_kernels_test SOURCES ${_portable_kernels_test_sources} EXTRA_LIBS
  portable_kernels portable_ops_lib
)
add_dependencies(portable_kernels_test generate_wrapper)
target_include_directories(
  portable_kernels_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/portable"
                                "${CMAKE_INSTALL_PREFIX}/include"
)

set(_optimized_kernels_test_sources
    "op_add_test.cpp"
    "op_bmm_test.cpp"
    "op_div_test.cpp"
    "op_exp_test.cpp"
    "op_gelu_test.cpp"
    "op_le_test.cpp"
    "op_log_softmax_test.cpp"
    "op_mul_test.cpp"
    "op_native_layer_norm_test.cpp"
    "op_neg_test.cpp"
    "op_sub_test.cpp"
    ${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/supported_features.cpp
)

# We don't have sleef on OSS so we don't have gelu and log_softmax
list(REMOVE_ITEM _optimized_kernels_test_sources "op_gelu_test.cpp"
     "op_log_softmax_test.cpp"
)

et_cxx_test(
  optimized_kernels_test
  SOURCES
  ${_optimized_kernels_test_sources}
  EXTRA_LIBS
  cpuinfo
  extension_threadpool
  optimized_kernels
  optimized_ops_lib
  portable_kernels
  pthreadpool
  eigen_blas
)
add_dependencies(optimized_kernels_test generate_wrapper)
target_include_directories(
  optimized_kernels_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/optimized"
                                 "${CMAKE_INSTALL_PREFIX}/include"
)
