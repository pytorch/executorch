/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Define export macro for Windows DLL
// When building the aoti_cuda_backend library, EXPORT_AOTI_FUNCTIONS is defined
// by CMake, which causes this macro to export symbols using
// __declspec(dllexport). When consuming the library, the macro imports symbols
// using
// __declspec(dllimport). On non-Windows platforms, the macro is empty and has
// no effect.
#ifdef _WIN32
#ifdef EXPORT_AOTI_FUNCTIONS
#define AOTI_SHIM_EXPORT __declspec(dllexport)
#else
#define AOTI_SHIM_EXPORT __declspec(dllimport)
#endif
#else
#define AOTI_SHIM_EXPORT
#endif

// Keep ExecuTorch's aoti_torch_* shim *definitions* in lockstep with the symbol
// names an AOTInductor blob imports. The blob and these (libtorch-free) shim
// libraries are both compiled with -DAOTI_SHIM_SYMBOL_PREFIX=executorch_ (see
// cuda_backend.py get_aoti_compile_options and the shim targets'
// exported_preprocessor_flags), so this block renames the exported definitions
// to executorch_aoti_torch_*. The blob (whose imports torch's
// aoti_torch/c/macros.h renames identically) then binds to ExecuTorch's
// SlimTensor shims and never to libtorch's at::Tensor aoti_torch_* in a
// coalesced (TensorRT[ATen] + CUDA) process.
//
// This block is kept self-contained (no torch include) to preserve the slim
// layer's libtorch-free property. It MUST stay in sync with the identical list
// in torch's aoti_torch/c/shim_symbol_prefix.h; a mismatch is a loud link error
// (an unresolved executorch_aoti_torch_* symbol), never a silent crash. With
// the define unset (portable builds) it expands to nothing.
#ifdef AOTI_SHIM_SYMBOL_PREFIX
#define AOTI_SHIM_CONCAT2_(a, b) a##b
#define AOTI_SHIM_CONCAT_(a, b) AOTI_SHIM_CONCAT2_(a, b)
#define AOTI_SHIM_RENAME_(name) AOTI_SHIM_CONCAT_(AOTI_SHIM_SYMBOL_PREFIX, name)

#define aoti_torch_assign_tensors_out \
  AOTI_SHIM_RENAME_(aoti_torch_assign_tensors_out)
#define aoti_torch_check AOTI_SHIM_RENAME_(aoti_torch_check)
#define aoti_torch_clone AOTI_SHIM_RENAME_(aoti_torch_clone)
#define aoti_torch_clone_preserve_strides \
  AOTI_SHIM_RENAME_(aoti_torch_clone_preserve_strides)
#define aoti_torch_copy_ AOTI_SHIM_RENAME_(aoti_torch_copy_)
#define aoti_torch_create_cuda_guard \
  AOTI_SHIM_RENAME_(aoti_torch_create_cuda_guard)
#define aoti_torch_create_cuda_stream_guard \
  AOTI_SHIM_RENAME_(aoti_torch_create_cuda_stream_guard)
#define aoti_torch_create_tensor_from_blob \
  AOTI_SHIM_RENAME_(aoti_torch_create_tensor_from_blob)
#define aoti_torch_create_tensor_from_blob_v2 \
  AOTI_SHIM_RENAME_(aoti_torch_create_tensor_from_blob_v2)
#define aoti_torch_cuda_guard_set_index \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_guard_set_index)
#define aoti_torch_cuda_int4_plain_mm \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_int4_plain_mm)
#define aoti_torch_cuda_int5_plain_mm \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_int5_plain_mm)
#define aoti_torch_cuda_int6_plain_mm \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_int6_plain_mm)
#define aoti_torch_cuda_int8_plain_mm \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_int8_plain_mm)
#define aoti_torch_cuda_rand AOTI_SHIM_RENAME_(aoti_torch_cuda_rand)
#define aoti_torch_cuda_randint_low_out \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_randint_low_out)
#define aoti_torch_cuda_sort_stable \
  AOTI_SHIM_RENAME_(aoti_torch_cuda_sort_stable)
#define aoti_torch_cuda__weight_int4pack_mm \
  AOTI_SHIM_RENAME_(aoti_torch_cuda__weight_int4pack_mm)
#define aoti_torch_delete_cuda_guard \
  AOTI_SHIM_RENAME_(aoti_torch_delete_cuda_guard)
#define aoti_torch_delete_cuda_stream_guard \
  AOTI_SHIM_RENAME_(aoti_torch_delete_cuda_stream_guard)
#define aoti_torch_delete_tensor_object \
  AOTI_SHIM_RENAME_(aoti_torch_delete_tensor_object)
#define aoti_torch_device_type_cpu AOTI_SHIM_RENAME_(aoti_torch_device_type_cpu)
#define aoti_torch_device_type_cuda \
  AOTI_SHIM_RENAME_(aoti_torch_device_type_cuda)
#define aoti_torch_dtype_bfloat16 AOTI_SHIM_RENAME_(aoti_torch_dtype_bfloat16)
#define aoti_torch_dtype_bool AOTI_SHIM_RENAME_(aoti_torch_dtype_bool)
#define aoti_torch_dtype_float16 AOTI_SHIM_RENAME_(aoti_torch_dtype_float16)
#define aoti_torch_dtype_float32 AOTI_SHIM_RENAME_(aoti_torch_dtype_float32)
#define aoti_torch_dtype_int16 AOTI_SHIM_RENAME_(aoti_torch_dtype_int16)
#define aoti_torch_dtype_int32 AOTI_SHIM_RENAME_(aoti_torch_dtype_int32)
#define aoti_torch_dtype_int64 AOTI_SHIM_RENAME_(aoti_torch_dtype_int64)
#define aoti_torch_dtype_int8 AOTI_SHIM_RENAME_(aoti_torch_dtype_int8)
#define aoti_torch_dtype_uint8 AOTI_SHIM_RENAME_(aoti_torch_dtype_uint8)
#define aoti_torch_empty_strided AOTI_SHIM_RENAME_(aoti_torch_empty_strided)
#define aoti_torch_empty_strided_pinned \
  AOTI_SHIM_RENAME_(aoti_torch_empty_strided_pinned)
#define aoti_torch_get_current_cuda_stream \
  AOTI_SHIM_RENAME_(aoti_torch_get_current_cuda_stream)
#define aoti_torch_get_data_ptr AOTI_SHIM_RENAME_(aoti_torch_get_data_ptr)
#define aoti_torch_get_device_index \
  AOTI_SHIM_RENAME_(aoti_torch_get_device_index)
#define aoti_torch_get_device_type AOTI_SHIM_RENAME_(aoti_torch_get_device_type)
#define aoti_torch_get_dim AOTI_SHIM_RENAME_(aoti_torch_get_dim)
#define aoti_torch_get_dtype AOTI_SHIM_RENAME_(aoti_torch_get_dtype)
#define aoti_torch_get_numel AOTI_SHIM_RENAME_(aoti_torch_get_numel)
#define aoti_torch_get_sizes AOTI_SHIM_RENAME_(aoti_torch_get_sizes)
#define aoti_torch_get_storage_offset \
  AOTI_SHIM_RENAME_(aoti_torch_get_storage_offset)
#define aoti_torch_get_storage_size \
  AOTI_SHIM_RENAME_(aoti_torch_get_storage_size)
#define aoti_torch_get_strides AOTI_SHIM_RENAME_(aoti_torch_get_strides)
#define aoti_torch_grad_mode_is_enabled \
  AOTI_SHIM_RENAME_(aoti_torch_grad_mode_is_enabled)
#define aoti_torch_grad_mode_set_enabled \
  AOTI_SHIM_RENAME_(aoti_torch_grad_mode_set_enabled)
#define aoti_torch_item_bool AOTI_SHIM_RENAME_(aoti_torch_item_bool)
#define aoti_torch_layout_strided AOTI_SHIM_RENAME_(aoti_torch_layout_strided)
#define aoti_torch_new_tensor_handle \
  AOTI_SHIM_RENAME_(aoti_torch_new_tensor_handle)
#define aoti_torch__reinterpret_tensor \
  AOTI_SHIM_RENAME_(aoti_torch__reinterpret_tensor)
#define aoti_torch_warn AOTI_SHIM_RENAME_(aoti_torch_warn)
#endif // AOTI_SHIM_SYMBOL_PREFIX
