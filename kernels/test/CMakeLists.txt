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

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_kernels portable optimized_portable optimized quantized)
foreach(kernel ${_kernels})
  set(_wrapper_dir
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/test"
  )
  set(_wrapper_path "${_wrapper_dir}/FunctionHeaderWrapper.h")
  set(_functions_include "#include <executorch/kernels/${kernel}/Functions.h>")
  add_custom_command(
    OUTPUT "${_wrapper_path}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_wrapper_dir}
    COMMAND ${CMAKE_COMMAND} -E echo ${_functions_include} > "${_wrapper_path}"
    DEPENDS
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/Functions.h"
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/NativeFunctions.h"
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/RegisterKernels.h"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating ${_wrapper_path}"
    VERBATIM
  )

  set(_supported_features_kernel ${kernel})
  if(${kernel} STREQUAL "optimized_portable")
    set(_supported_features_kernel "portable")
  endif()
  add_custom_command(
    OUTPUT "${_wrapper_dir}/supported_features.cpp"
           "${_wrapper_dir}/supported_features.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_wrapper_dir}
    COMMAND
      ${PYTHON_EXECUTABLE} kernels/test/gen_supported_features.py
      kernels/${_supported_features_kernel}/test/supported_features_def.yaml >
      ${_wrapper_dir}/supported_features.cpp
    COMMAND
      ${PYTHON_EXECUTABLE} kernels/test/gen_supported_features.py
      kernels/test/supported_features.yaml >
      ${_wrapper_dir}/supported_features.h
    WORKING_DIRECTORY "${EXECUTORCH_ROOT}"
    COMMENT "Generating ${_wrapper_dir}/supported_features.cpp and header"
    VERBATIM
  )
  if(${kernel} STREQUAL "optimized")
    set(_kernel_ops_lib "optimized_native_cpu_ops_lib")
    set(_kernel_ops_lib_path
        "${CMAKE_CURRENT_BINARY_DIR}/../../configurations/optimized_native_cpu_ops_lib"
    )
  elseif(${kernel} STREQUAL "optimized_portable")
    set(_kernel_ops_lib "${kernel}_ops_lib")
    set(_kernel_ops_lib_path
        "${CMAKE_CURRENT_BINARY_DIR}/../../kernels/portable/${kernel}_ops_lib"
    )
  else()
    set(_kernel_ops_lib "${kernel}_ops_lib")
    set(_kernel_ops_lib_path
        "${CMAKE_CURRENT_BINARY_DIR}/../../kernels/${kernel}/${kernel}_ops_lib"
    )
  endif()

  # Copy with glob needs to be handle in a platform-specific manner.
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    # The quoting here is complicated, because there are three levels of
    # interpretation: CMake -> Batch -> Powershell. The invoked (batch) command
    # should look like `powershell -Command "Copy-Item ... -Path \"...\" ...".
    # Powershell sees `Copy-Item -Path "..." ...`.
    set(_copy_headers_cmd
        powershell
        -Command
        "Copy-Item -Path \\\"${_kernel_ops_lib_path}/*.h\\\" -Destination \\\"${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/\\\""
    )
  else()
    set(_copy_headers_cmd
        cp
        "${_kernel_ops_lib_path}/*.h"
        "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/"
    )
  endif()

  add_custom_command(
    OUTPUT
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/Functions.h"
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/NativeFunctions.h"
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/RegisterKernels.h"
    COMMAND
      ${CMAKE_COMMAND} -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/${kernel}/"
    COMMAND ${_copy_headers_cmd}
    DEPENDS ${_kernel_ops_lib}
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
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized_portable/executorch/kernels/test/FunctionHeaderWrapper.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized_portable/executorch/kernels/test/supported_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized_portable/executorch/kernels/test/supported_features.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/include/quantized/executorch/kernels/test/FunctionHeaderWrapper.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/quantized/executorch/kernels/test/supported_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/quantized/executorch/kernels/test/supported_features.cpp"
)

set(all_test_sources
    "BinaryLogicalOpTest.cpp"
    "op__to_dim_order_copy_test.cpp"
    "op__clone_dim_order_test.cpp"
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
    "op_convolution_backward_test.cpp"
    "op_convolution_test.cpp"
    "op_copy_test.cpp"
    "op_cos_test.cpp"
    "op_cosh_test.cpp"
    "op_cumsum_test.cpp"
    "op_detach_copy_test.cpp"
    "op_diagonal_copy_test.cpp"
    "op_div_test.cpp"
    "op_elu_test.cpp"
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
    "op_gather_test.cpp"
    "op_ge_test.cpp"
    "op_gelu_test.cpp"
    "op_glu_test.cpp"
    "op_grid_sampler_2d_test.cpp"
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
    "op_native_dropout_test.cpp"
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
    "op_rand_test.cpp"
    "op_randn_test.cpp"
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
    "op_scatter_test.cpp"
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
    "op_topk_test.cpp"
    "op_transpose_copy_test.cpp"
    "op_tril_test.cpp"
    "op_trunc_test.cpp"
    "op_unbind_copy_test.cpp"
    "op_unsqueeze_copy_test.cpp"
    "op_upsample_bilinear2d_test.cpp"
    "op_upsample_bilinear2d_aa_test.cpp"
    "op_upsample_nearest2d_test.cpp"
    "op_var_test.cpp"
    "op_view_as_real_copy_test.cpp"
    "op_view_copy_test.cpp"
    "op_where_test.cpp"
    "op_zeros_test.cpp"
    "UnaryUfuncRealHBBF16ToFloatHBF16Test.cpp"
)

# These requires rtti and not working so far.
list(REMOVE_ITEM all_test_sources "op_to_copy_test.cpp"
     "op__to_dim_order_copy_test.cpp"
)

set(_portable_kernels_test_sources
    ${all_test_sources}
    ${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/supported_features.cpp
    "${EXECUTORCH_ROOT}/kernels/portable/test/op_div_test.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/test/op_gelu_test.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/test/op_mul_test.cpp"
)

et_cxx_test(
  portable_kernels_test SOURCES ${_portable_kernels_test_sources} EXTRA_LIBS
  portable_kernels portable_ops_lib
)
add_dependencies(portable_kernels_test generate_wrapper)
target_include_directories(
  portable_kernels_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/portable"
)

set(_optimized_kernels_test_sources
    "op_add_test.cpp"
    "op_bmm_test.cpp"
    "op_div_test.cpp"
    "op_elu_test.cpp"
    "op_exp_test.cpp"
    "op_fft_c2r_test.cpp"
    "op_fft_r2c_test.cpp"
    "op_gelu_test.cpp"
    "op_le_test.cpp"
    "op_linear_test.cpp"
    "op_log_softmax_test.cpp"
    "op_mm_test.cpp"
    "op_mul_test.cpp"
    "op_native_layer_norm_test.cpp"
    "op_neg_test.cpp"
    "op_sub_test.cpp"
    "op_where_test.cpp"
    "UnaryUfuncRealHBBF16ToFloatHBF16Test.cpp"
    ${CMAKE_CURRENT_BINARY_DIR}/include/optimized/executorch/kernels/test/supported_features.cpp
)

if(TARGET optimized_portable_kernels)
  list(APPEND _optimized_kernels_test_sources ${all_test_sources})
  list(REMOVE_DUPLICATES _optimized_kernels_test_sources)

  # Make sure that we still test optimized versions of portable kernels even if
  # they would currently be shadowed by specific optimized implementations.
  et_cxx_test(
    optimized_portable_kernels_test
    SOURCES
    ${all_test_sources}
    ${CMAKE_CURRENT_BINARY_DIR}/include/optimized_portable/executorch/kernels/test/supported_features.cpp
    EXTRA_LIBS
    optimized_portable_kernels
  )
  add_dependencies(optimized_portable_kernels_test generate_wrapper)
  target_include_directories(
    optimized_portable_kernels_test
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/optimized_portable"
  )
endif()

et_cxx_test(
  optimized_kernels_test
  SOURCES
  ${_optimized_kernels_test_sources}
  EXTRA_LIBS
  cpuinfo
  extension_threadpool
  optimized_native_cpu_ops_lib
  pthreadpool
  eigen_blas
)
add_dependencies(optimized_kernels_test generate_wrapper)
target_include_directories(
  optimized_kernels_test
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/optimized"
          "${CMAKE_CURRENT_BINARY_DIR}/include/portable"
          "${CMAKE_INSTALL_PREFIX}/include"
)

if(TARGET quantized_kernels)
  set(_quantized_kernels_test_sources
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_add_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_choose_qparams_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_embedding2b_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_embedding4b_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_embedding_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_mixed_linear_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_mixed_mm_test.cpp"
      "${EXECUTORCH_ROOT}/kernels/quantized/test/op_quantize_test.cpp"
  )

  et_cxx_test(
    quantized_kernels_test
    SOURCES
    ${_quantized_kernels_test_sources}
    EXTRA_LIBS
    cpuinfo
    extension_threadpool
    quantized_kernels
    quantized_ops_lib
    portable_kernels
    portable_ops_lib
    pthreadpool
    eigen_blas
  )
  add_dependencies(quantized_kernels_test generate_wrapper)
  target_include_directories(
    quantized_kernels_test
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/quantized"
            "${CMAKE_CURRENT_BINARY_DIR}/include/portable"
  )
endif()
