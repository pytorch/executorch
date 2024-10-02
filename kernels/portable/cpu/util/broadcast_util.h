/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

/**
 * Check whether or not the broadcast_from_shape can be broadcasted onto the
 * broadcast_to_shape.
 *
 * @param[in] broadcast_from_shape The tensor shape which we want to broadcast.
 * @param[in] broadcast_to_shape The tensor shape which we want to broadcast to.
 * @returns A bool to indicate whether or not the shape can be broadcasted.
 *
 */
bool tensor_is_broadcastable_to(
    const exec_aten::ArrayRef<Tensor::SizesType> broadcast_from_shape,
    const exec_aten::ArrayRef<Tensor::SizesType> broadcast_to_shape);

/**
 * Check whether or not the broadcast_from tensor should and can be broadcasted
 * onto the broadcast_to tensor. broadcast_tensor should only be called if this
 * returns true.
 *
 * @param[in] broadcast_from The tensor which we want to broadcast from.
 * @param[in] broadcast_to The tensor to which we want to broadcast to.
 * @returns A bool to indicate whether or not the tensor can be broadcasted.
 *
 */
bool tensor_is_broadcastable_to(
    const Tensor& broadcast_from,
    const Tensor& broadcast_to);

/**
 * Returns true if the two tensor shapes can both be broadcasted to a common
 * shape.
 *
 * @param[in] a_shape The sizes of the first tensor going to be test.
 * @param[in] b_shape The sizes of the second tensor going to be test.
 * @returns true if the tensors are broadcastable, false otherwise.
 */
bool tensors_are_broadcastable_between(
    const exec_aten::ArrayRef<Tensor::SizesType> a_shape,
    const exec_aten::ArrayRef<Tensor::SizesType> b_shape);

/**
 * Convenience overload of the above function to accept Tensor inputs.
 *
 * @param[in] a The first tensor going to be test.
 * @param[in] b The second tensor going to be test.
 * @returns true if the tensors are broadcastable, false otherwise.
 */
bool tensors_are_broadcastable_between(const Tensor& a, const Tensor& b);

/**
 * DEPRECATED: Use `delinearize_index()` and `linearize_access_indexes()` for
 * index remapping to avoid memory allocation.
 *
 * The smaller tensor broadcast_from is “broadcast” across the larger tensor
 * broadcast_to so that they have compatible shapes.
 * broadcast_to_shape.size() >= broadcast_from_shape.size() in order for this
 * to work.
 *
 * @param[in] broadcast_from The tensor to which we want to broadcast from.
 * @param[in] broadcast_to The tensor to which we want to broadcast to.
 * @returns A new tensor with the same shape as broadcast_to and the data
 * repeated as appropriate. This tensor contains dynamically allocated memory
 * and must be freed using free_broadcast_tensor.
 */
ET_DEPRECATED exec_aten::Tensor broadcast_tensor(
    const exec_aten::Tensor& broadcast_from,
    const exec_aten::Tensor& broadcast_to);

/**
 * Get the size of the target tensor that two input tensors would be broadcasted
 * to.
 *
 * This function is useful especially for the operator supporting both broadcast
 * and dynamic shape. At that time there may not be a tensor having the size of
 * final output, so we need to calculate it.
 *
 * @param[in] a_size The size of the first tensor going to be broadcasted.
 * @param[in] b_size The size of the second tensor going to be broadcasted.
 * @param[out] out_sizes The memory space storing the size of
 * broadcasted target tensor
 * @param[in] out_sizes_len The largest number of element
 * @param[out] out_dim The dimension of the broadcasted target
 * tensor
 */
ET_NODISCARD Error get_broadcast_target_size(
    const exec_aten::ArrayRef<Tensor::SizesType> a_size,
    const exec_aten::ArrayRef<Tensor::SizesType> b_size,
    Tensor::SizesType* out_sizes,
    const size_t out_sizes_len,
    size_t* out_dim);

/**
 * Convenience overload of the above function to accept Tensor inputs.
 *
 * @param[in] a The first tensor going to be broadcasted.
 * @param[in] b The second tensor going to be broadcasted.
 * @param[out] out_sizes The memory space storing the size of
 * broadcasted target tensor
 * @param[in] out_sizes_len The largest number of element
 * @param[out] out_dim The dimension of the broadcasted target
 * tensor
 */
ET_NODISCARD Error get_broadcast_target_size(
    const Tensor& a,
    const Tensor& b,
    Tensor::SizesType* out_sizes,
    const size_t out_sizes_len,
    size_t* out_dim);

/**
 * Get the size that two input tensors will be broadcasted to, and resize an
 * output tensor to the resulting broadcasted size.
 *
 * @param[in] a The first tensor going to be broadcasted.
 * @param[in] b The second tensor going to be broadcasted.
 * @param[out] out The output tensor that will be resized.
 */
ET_NODISCARD inline Error
resize_to_broadcast_target_size(const Tensor& a, const Tensor& b, Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  size_t expected_output_dim = 0;

  ET_CHECK_OK_OR_RETURN_ERROR(
      get_broadcast_target_size(
          a,
          b,
          expected_output_size,
          kTensorDimensionLimit,
          &expected_output_dim),
      "Failed to get broadcast target size");

  return resize_tensor(out, {expected_output_size, expected_output_dim});
}

/**
 * Get the size that three input tensors will be broadcasted to, and resize an
 * output tensor to the resulting broadcasted size.
 *
 * @param[in] a The first tensor going to be broadcasted.
 * @param[in] b The second tensor going to be broadcasted.
 * @param[in] c The third tensor going to be broadcasted.
 * @param[out] out The output tensor that will be resized.
 */
ET_NODISCARD inline Error resize_to_broadcast_target_size(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    Tensor& out) {
  Tensor::SizesType interim_output_size[kTensorDimensionLimit];
  size_t interim_output_dim = 0;

  // Obtain the broadcast size of the first two input tensors
  ET_CHECK_OK_OR_RETURN_ERROR(
      get_broadcast_target_size(
          a,
          b,
          interim_output_size,
          kTensorDimensionLimit,
          &interim_output_dim),
      "Failed to get broadcast target size");

  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  size_t expected_output_dim = 0;

  // Apply broadcasting to the intermediate broadcast size and the third input
  // tensor
  ET_CHECK_OK_OR_RETURN_ERROR(
      get_broadcast_target_size(
          {interim_output_size, interim_output_dim},
          c.sizes(),
          expected_output_size,
          kTensorDimensionLimit,
          &expected_output_dim),
      "Failed to get broadcast target size");

  return resize_tensor(out, {expected_output_size, expected_output_dim});
}

/**
 * DEPRECATED: Use `delinearize_index()` and `linearize_access_indexes()` for
 * index remapping to avoid memory allocation.
 *
 * Free the dynamically allocated memory in broadcast_tensor. This should only
 * be used on a tensor returned by broadcast_tensor.
 *
 * @param[in] The tensor that was previosuly returned by a call to
 * broadcast_tensor.
 * @returns void
 */
ET_DEPRECATED void free_broadcast_tensor(
    const exec_aten::Tensor& broadcast_tensor);

/**
 * Delinearize a flattened index to per-dimension indexes.
 *
 * @param[in] linear_index The flattened index
 * @param[in] shape The tensor shape
 * @param[out] out_indexes The per-dimension indexes
 * @param[in] out_indexes_len The maximum size of the out_indexes array
 * @returns void
 */
void delinearize_index(
    size_t linear_index,
    exec_aten::ArrayRef<Tensor::SizesType> shape,
    size_t* out_indexes,
    const size_t out_indexes_len);

/**
 * Delinearize a flattened index to per-dimension indexes.
 *
 * @param[in] linear_index The flattened index
 * @param[in] t The tensor object
 * @param[out] out_indexes The per-dimension indexes
 * @param[in] out_indexes_len The maximum size of the out_indexes array
 * @returns void
 */
void delinearize_index(
    size_t linear_index,
    const Tensor& t,
    size_t* out_indexes,
    const size_t out_indexes_len);

/**
 * Return the linear index for broatcast_from tensor, given the indexes and
 * number of dimensions of broadcast_to tensor, and the shape and strides
 * of broadcast_from tensor.
 *
 * @param[in] indexes_broadcast_to The access indexes of broadcast_to tensor.
 * @param[in] broadcast_to_ndim The number of dims of broadcast_to tensor.
 * @param[in] broadcast_from_shape The shape of the broadcasted tensor.
 * @param[in] broadcast_from_strides The strides of the broadcasted tensor.
 * @returns The flattend index for broadcast_from tensor.
 */
size_t linearize_access_indexes(
    ArrayRef<size_t> indexes_broadcast_to,
    ssize_t broadcast_to_ndim,
    exec_aten::ArrayRef<Tensor::SizesType> broadcast_from_shape,
    exec_aten::ArrayRef<Tensor::StridesType> broadcast_from_strides);

/**
 * Return the linear index for broatcast_from tensor, given the indexes of
 * broadcast_to tensor and itself.
 *
 * @param[in] indexes_broadcast_to The access indexes of broadcast_to tensor.
 * @param[in] broadcast_to_ndim The number of dims of broadcast_to tensor.
 * @param[in] broadcast_from The tensor to be broadcasted.
 * @returns The flattend index for broadcast_from tensor.
 */
size_t linearize_access_indexes(
    ArrayRef<size_t> indexes_broadcast_to,
    ssize_t broadcast_to_ndim,
    const Tensor& broadcast_from);

//
// Mapping with broadcasting
//

namespace internal {
template <typename To, typename From>
To load_and_convert(const void* fromPtr) {
  return static_cast<To>(*reinterpret_cast<const From*>(fromPtr));
}

template <typename To, typename From>
void convert_and_store(From f, void* dst) {
  *reinterpret_cast<To*>(dst) = static_cast<To>(f);
}
} // namespace internal

template <typename CTYPE_COMMON>
using load_to_common_fn = CTYPE_COMMON (*)(const void*);

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_realhbbf16(
    const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMMON>
using store_common_to_tensor_fn = void (*)(CTYPE_COMMON, void*);

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_realhbbf16(const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMMON>;
      });
  return result;
}

/**
 * Useful for binary elementwise operators. For each element of the inputs,
 * perform a computation and write to the corresponding element of the output.
 * Tensor broadcasting is applied wherever it is required.
 */
template <typename CTYPE_A, typename CTYPE_B, typename CTYPE_OUT, typename Op>
inline void apply_binary_elementwise_fn(
    const Op& compute_fun,
    const Tensor& a,
    const Tensor& b,
    const Tensor& out) {
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool any_is_broadcasted = (a_is_broadcasted || b_is_broadcasted);

  const CTYPE_A* const data_a = a.const_data_ptr<CTYPE_A>();
  const CTYPE_B* const data_b = b.const_data_ptr<CTYPE_B>();
  CTYPE_OUT* const data_out = out.mutable_data_ptr<CTYPE_OUT>();

  for (size_t i = 0; i < out.numel(); ++i) {
    size_t a_linear_index = i;
    size_t b_linear_index = i;

    if (any_is_broadcasted) {
      size_t out_indexes[kTensorDimensionLimit];
      delinearize_index(i, out, out_indexes, kTensorDimensionLimit);

      if (a_is_broadcasted) {
        a_linear_index = linearize_access_indexes(out_indexes, out.dim(), a);
      }
      if (b_is_broadcasted) {
        b_linear_index = linearize_access_indexes(out_indexes, out.dim(), b);
      }
    }

    data_out[i] = compute_fun(data_a[a_linear_index], data_b[b_linear_index]);
  }
}

/**
 * Useful for ternary elementwise operators. For each element of the inputs,
 * perform a computation and write to the corresponding element of the output.
 * Tensor broadcasting is applied wherever it is required.
 *
 * In order to mitigate build time cost (straightforwardly |CTYPE_A| *
 * |CTYPE_B| * |CTYPE_C| * |CTYPE_OUT|), all arguments to compute_fun
 * are passed as CTYPE_COMMON. We require compute_fun to return
 * CTYPE_COMMON, and we require loading conversion functions from each
 * input type to CTYPE_COMMON and a storing conversion from
 * CTYPE_COMMON to CTYPE_OUT be provided. Each conversion function
 * must take a void* pointing to an element of the corresponding
 * tensor, load that element, and convert it to CTYPE_COMMON. The
 * storing conversion function must have the signature
 * void(CTYPE_COMMON, void*), convert the given element to CTYPE_OUT,
 * and store it to the given location.
 */
template <typename CTYPE_COMMON, typename Op>
inline void apply_ternary_elementwise_fn(
    const Op& compute_fun,
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& out,
    CTYPE_COMMON (*load_a_to_common)(const void*),
    CTYPE_COMMON (*load_b_to_common)(const void*),
    CTYPE_COMMON (*load_c_to_common)(const void*),
    void (*store_common_to_out)(CTYPE_COMMON, void*)) {
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool c_is_broadcasted = !out.sizes().equals(c.sizes());
  const bool any_is_broadcasted =
      (a_is_broadcasted || b_is_broadcasted || c_is_broadcasted);

  const char* const data_a = reinterpret_cast<const char*>(a.const_data_ptr());
  const char* const data_b = reinterpret_cast<const char*>(b.const_data_ptr());
  const char* const data_c = reinterpret_cast<const char*>(c.const_data_ptr());
  const auto a_element_size = a.element_size();
  const auto b_element_size = b.element_size();
  const auto c_element_size = c.element_size();
  const auto out_element_size = out.element_size();
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());

  for (size_t i = 0; i < out.numel(); ++i) {
    size_t a_linear_index = i;
    size_t b_linear_index = i;
    size_t c_linear_index = i;

    if (any_is_broadcasted) {
      size_t out_indexes[kTensorDimensionLimit];
      delinearize_index(i, out, out_indexes, kTensorDimensionLimit);

      if (a_is_broadcasted) {
        a_linear_index = linearize_access_indexes(out_indexes, out.dim(), a);
      }
      if (b_is_broadcasted) {
        b_linear_index = linearize_access_indexes(out_indexes, out.dim(), b);
      }
      if (c_is_broadcasted) {
        c_linear_index = linearize_access_indexes(out_indexes, out.dim(), c);
      }
    }

    auto result = compute_fun(
        load_a_to_common(&data_a[a_linear_index * a_element_size]),
        load_b_to_common(&data_b[b_linear_index * b_element_size]),
        load_c_to_common(&data_c[c_linear_index * c_element_size]));
    store_common_to_out(result, &data_out[i * out_element_size]);
  }
}

} // namespace executor
} // namespace torch
