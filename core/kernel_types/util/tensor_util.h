// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <array> // std::array
#include <cinttypes> // PRId64
#include <cmath>
#include <cstddef> // size_t
#include <limits>

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/util/DimOrderUtils.h>
#include <executorch/core/kernel_types/util/ScalarTypeUtil.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/compiler.h>

/// All assertion messages should begin with this prefix.
#define ET_TENSOR_CHECK_PREFIX__ "Tensors do not match"
#define ET_MIN2(a, b) (std::min(a, b))
#define ET_MIN3(a, b, c) (std::min(a, std::min(b, c)))

#define ET_NORMALIZE_IX(IX, UPPER_BOUND) IX < 0 ? IX + UPPER_BOUND : IX

#define ET_CHECK_VALID_IX(IX, UPPER_BOUND)                  \
  ET_CHECK_MSG(                                             \
      IX >= -static_cast<int64_t>(UPPER_BOUND) &&           \
          IX < static_cast<int64_t>(UPPER_BOUND),           \
      "index %" PRId64 " must be within range [-%zd, %zd)", \
      IX,                                                   \
      UPPER_BOUND,                                          \
      UPPER_BOUND)

#define ET_CHECK_VALID_DIM(DIM, UPPER_BOUND)              \
  ET_CHECK_MSG(                                           \
      DIM >= -static_cast<int64_t>(UPPER_BOUND) &&        \
          DIM < static_cast<int64_t>(UPPER_BOUND),        \
      "dim %" PRId64 " must be within range [-%zd, %zd)", \
      DIM,                                                \
      UPPER_BOUND,                                        \
      UPPER_BOUND)

#define ET_CHECK_NON_ZERO_DIM_SIZE(DIM, T)           \
  const size_t udim = ET_NORMALIZE_IX(DIM, T.dim()); \
  ET_CHECK_MSG(                                      \
      T.size(udim) != 0, "Expected dim %zd to have non-zero size.", udim);

/**
 * Asserts that all tensors have the same shape.
 * This also handles a edge case where there is only one element in all the
 * tensors being compared but the number of dimensions >= 0. In the for loop
 * iterating over the dimensions we make sure that we pick the smallest
 * dimension of all the tensors as the upper bound for the for loop.
 */
#define ET_CHECK_SAME_SHAPE2(a__, b__)                                    \
  ({                                                                      \
    const size_t a_numel__ = (a__).numel();                               \
    const size_t b_numel__ = (b__).numel();                               \
    const size_t a_dim__ = (a__).dim();                                   \
    const size_t b_dim__ = (b__).dim();                                   \
    ET_CHECK_MSG(                                                         \
        a_numel__ == b_numel__ &&                                         \
            ((a_numel__ == 1 && b_numel__ == 1) || (a_dim__ == b_dim__)), \
        ET_TENSOR_CHECK_PREFIX__ ": numel={%zu, %zu}, dim={%zu, %zu}",    \
        a_numel__,                                                        \
        b_numel__,                                                        \
        a_dim__,                                                          \
        b_dim__);                                                         \
    for (size_t dim__ = 0; dim__ < ET_MIN2(a_dim__, b_dim__); ++dim__) {  \
      size_t a_size__ = (a__).size(dim__);                                \
      size_t b_size__ = (b__).size(dim__);                                \
      ET_CHECK_MSG(                                                       \
          a_size__ == b_size__,                                           \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu}",           \
          dim__,                                                          \
          a_size__,                                                       \
          b_size__);                                                      \
    }                                                                     \
  })

#define ET_CHECK_SAME_SHAPE3(a__, b__, c__)                            \
  ({                                                                   \
    const size_t a_numel__ = (a__).numel();                            \
    const size_t b_numel__ = (b__).numel();                            \
    const size_t c_numel__ = (c__).numel();                            \
    const size_t a_dim__ = (a__).dim();                                \
    const size_t b_dim__ = (b__).dim();                                \
    const size_t c_dim__ = (c__).dim();                                \
    ET_CHECK_MSG(                                                      \
        a_numel__ == b_numel__ && b_numel__ == c_numel__ &&            \
            ((a_numel__ == 1 && b_numel__ == 1 && c_numel__ == 1) ||   \
             a_dim__ == b_dim__ && b_dim__ == c_dim__),                \
        ET_TENSOR_CHECK_PREFIX__                                       \
        ": numel={%zu, %zu, %zu}, dim={%zu, %zu, %zu}",                \
        a_numel__,                                                     \
        b_numel__,                                                     \
        c_numel__,                                                     \
        a_dim__,                                                       \
        b_dim__,                                                       \
        c_dim__);                                                      \
    for (size_t dim__ = 0; dim__ < ET_MIN3(a_dim__, b_dim__, c_dim__); \
         ++dim__) {                                                    \
      size_t a_size__ = (a__).size(dim__);                             \
      size_t b_size__ = (b__).size(dim__);                             \
      size_t c_size__ = (c__).size(dim__);                             \
      ET_CHECK_MSG(                                                    \
          a_size__ == b_size__ && b_size__ == c_size__,                \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu, %zu}",   \
          dim__,                                                       \
          a_size__,                                                    \
          b_size__,                                                    \
          c_size__);                                                   \
    }                                                                  \
  })

/// Asserts that all tensors have the same dtype.
#define ET_CHECK_SAME_DTYPE2(a__, b__)                            \
  ({                                                              \
    const ::exec_aten::ScalarType a_type__ = (a__).scalar_type(); \
    const ::exec_aten::ScalarType b_type__ = (b__).scalar_type(); \
    ET_CHECK_MSG(                                                 \
        a_type__ == b_type__,                                     \
        ET_TENSOR_CHECK_PREFIX__ ": dtype={%hhd, %hhd}",          \
        a_type__,                                                 \
        b_type__);                                                \
  })

#define ET_CHECK_SAME_DTYPE3(a__, b__, c__)                       \
  ({                                                              \
    const ::exec_aten::ScalarType a_type__ = (a__).scalar_type(); \
    const ::exec_aten::ScalarType b_type__ = (b__).scalar_type(); \
    const ::exec_aten::ScalarType c_type__ = (c__).scalar_type(); \
    ET_CHECK_MSG(                                                 \
        a_type__ == b_type__ && b_type__ == c_type__,             \
        ET_TENSOR_CHECK_PREFIX__ ": dtype={%hhd, %hhd, %hhd}",    \
        a_type__,                                                 \
        b_type__,                                                 \
        c_type__);                                                \
  })

/**
 * Asserts that all tensors have the same shape and dtype.
 *
 * This macro should produce less code/data than calling the SHAPE and DTYPE
 * macros independently, because it only calls ET_CHECK_MSG once.
 */
#define ET_CHECK_SAME_SHAPE_AND_DTYPE2(a__, b__)                          \
  ({                                                                      \
    const size_t a_numel__ = (a__).numel();                               \
    const size_t b_numel__ = (b__).numel();                               \
    const size_t a_dim__ = (a__).dim();                                   \
    const size_t b_dim__ = (b__).dim();                                   \
    const ::exec_aten::ScalarType a_type__ = (a__).scalar_type();         \
    const ::exec_aten::ScalarType b_type__ = (b__).scalar_type();         \
                                                                          \
    ET_CHECK_MSG(                                                         \
        a_numel__ == b_numel__ &&                                         \
            ((a_numel__ == 1 && b_numel__ == 1) || a_dim__ == b_dim__) && \
            a_type__ == b_type__,                                         \
        ET_TENSOR_CHECK_PREFIX__                                          \
        ": numel={%zu, %zu}, dim={%zu, %zu}, dtype={%hhd, %hhd}",         \
        a_numel__,                                                        \
        b_numel__,                                                        \
        a_dim__,                                                          \
        b_dim__,                                                          \
        a_type__,                                                         \
        b_type__);                                                        \
    for (size_t dim__ = 0; dim__ < ET_MIN2(a_dim__, b_dim__); ++dim__) {  \
      size_t a_size__ = (a__).size(dim__);                                \
      size_t b_size__ = (b__).size(dim__);                                \
      ET_CHECK_MSG(                                                       \
          a_size__ == b_size__,                                           \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu}",           \
          dim__,                                                          \
          a_size__,                                                       \
          b_size__);                                                      \
    }                                                                     \
  })

#define ET_CHECK_SAME_SHAPE_AND_DTYPE3(a__, b__, c__)                  \
  ({                                                                   \
    const size_t a_numel__ = (a__).numel();                            \
    const size_t b_numel__ = (b__).numel();                            \
    const size_t c_numel__ = (c__).numel();                            \
    const size_t a_dim__ = (a__).dim();                                \
    const size_t b_dim__ = (b__).dim();                                \
    const size_t c_dim__ = (c__).dim();                                \
    const ::exec_aten::ScalarType a_type__ = (a__).scalar_type();      \
    const ::exec_aten::ScalarType b_type__ = (b__).scalar_type();      \
    const ::exec_aten::ScalarType c_type__ = (c__).scalar_type();      \
                                                                       \
    ET_CHECK_MSG(                                                      \
        a_numel__ == b_numel__ && b_numel__ == c_numel__ &&            \
            ((a_numel__ == 1 && b_numel__ == 1 && c_numel__ == 1) ||   \
             (a_dim__ == b_dim__ && b_dim__ == c_dim__)) &&            \
            a_type__ == b_type__ && b_type__ == c_type__,              \
        ET_TENSOR_CHECK_PREFIX__                                       \
        ": numel={%zu, %zu, %zu}, dim={%zu, %zu, %zu}, "               \
        "dtype={%hhd, %hhd, %hhd}",                                    \
        a_numel__,                                                     \
        b_numel__,                                                     \
        c_numel__,                                                     \
        a_dim__,                                                       \
        b_dim__,                                                       \
        c_dim__,                                                       \
        a_type__,                                                      \
        b_type__,                                                      \
        c_type__);                                                     \
    for (size_t dim__ = 0; dim__ < ET_MIN3(a_dim__, b_dim__, c_dim__); \
         ++dim__) {                                                    \
      size_t a_size__ = (a__).size(dim__);                             \
      size_t b_size__ = (b__).size(dim__);                             \
      size_t c_size__ = (c__).size(dim__);                             \
      ET_CHECK_MSG(                                                    \
          a_size__ == b_size__ && b_size__ == c_size__,                \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu, %zu}",   \
          dim__,                                                       \
          a_size__,                                                    \
          b_size__,                                                    \
          c_size__);                                                   \
    }                                                                  \
  })

/**
 * Assert that the input tensor is contiguous tensor.
 */
#define ET_CHECK_CONTIGUOUS(a__)                                              \
  ({                                                                          \
    const ::exec_aten::ArrayRef<int32_t> strides = a__.strides();             \
    const ::exec_aten::ArrayRef<int32_t> sizes = a__.sizes();                 \
    ET_CHECK_MSG(                                                             \
        strides[strides.size() - 1] == 1,                                     \
        "The stride of the last dimension shall be 1 for contiguous tensor, " \
        "not %d",                                                             \
        strides[strides.size() - 1]);                                         \
    for (size_t i = strides.size() - 1; i > 0; i--) {                         \
      ET_CHECK_MSG(                                                           \
          strides[i - 1] == strides[i] * sizes[i],                            \
          "The stride of the %zu-th dimension shall equal to "                \
          "strides[%zu] * sizes[%zu], now is %d and %d",                      \
          i - 1,                                                              \
          i,                                                                  \
          i,                                                                  \
          strides[i - 1],                                                     \
          strides[i] * sizes[i]);                                             \
    }                                                                         \
  })

/**
 * Assert the input two tensors share same strides.
 * Noted that this function does not make any check or promise on the contiguity
 * of any input tensors.
 */
#define ET_CHECK_SAME_STRIDES2(a__, b__)                                       \
  ({                                                                           \
    ET_CHECK_MSG(                                                              \
        a__.dim() == b__.dim(),                                                \
        "Two tensors shall have same number of strides, but not %zu and %zu.", \
        a__.dim(),                                                             \
        b__.dim());                                                            \
    const ::exec_aten::ArrayRef<int32_t> a_strides = a__.strides();            \
    const ::exec_aten::ArrayRef<int32_t> b_strides = b__.strides();            \
    for (size_t i = 0; i < a__.dim(); i++) {                                   \
      ET_CHECK_MSG(                                                            \
          a_strides[i] == b_strides[i],                                        \
          "a.strides()[%zu] shall equal to b.strides()[%zu], "                 \
          "but now is %d and %d.",                                             \
          i,                                                                   \
          i,                                                                   \
          a_strides[i],                                                        \
          b_strides[i]);                                                       \
    }                                                                          \
  })

/**
 * Assert the input three tensors share same strides.
 * Noted that this function does not make any check or promise on the contiguity
 * of any input tensors.
 */
#define ET_CHECK_SAME_STRIDES3(a__, b__, c__)                           \
  ({                                                                    \
    ET_CHECK_MSG(                                                       \
        a__.dim() == b__.dim() && b__.dim() == c__.dim(),               \
        "Three tensors shall have same number of strides, "             \
        "but not %zu, %zu and %zu.",                                    \
        a__.dim(),                                                      \
        b__.dim(),                                                      \
        c__.dim());                                                     \
    const ::exec_aten::ArrayRef<int32_t> a_strides = a__.strides();     \
    const ::exec_aten::ArrayRef<int32_t> b_strides = b__.strides();     \
    const ::exec_aten::ArrayRef<int32_t> c_strides = c__.strides();     \
    for (size_t i = 0; i < a__.dim(); i++) {                            \
      ET_CHECK_MSG(                                                     \
          a_strides[i] == b_strides[i] && b_strides[i] == c_strides[i], \
          "a_strides[%zu], b_strides[%zu] and c_strides[%zu] "          \
          "shall share same value, but now is %d, %d and %d",           \
          i,                                                            \
          i,                                                            \
          i,                                                            \
          a_strides[i],                                                 \
          b_strides[i],                                                 \
          c_strides[i]);                                                \
    }                                                                   \
  })

#define ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(t__)           \
  ({                                                             \
    ET_CHECK_MSG(                                                \
        is_default_dim_order(                                    \
            t__.dim_order().data(), t__.dim_order().size()) ||   \
            is_channels_last_dim_order(                          \
                t__.dim_order().data(), t__.dim_order().size()), \
        "Tensor must have default or channels last dim order");  \
  })

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

/**
 * The expected output size may not be the existing size of any inputs and
 * outputs if the operator supports both broadcast and dynamic shape. Therefore
 * such operators needs extra space to store the calculated expected output
 * size. such dynamic allocation is troublesome in executorch so we can just
 * hard code a static value of a relatively small value because users don't
 * create high dimensional tensors.
 */
constexpr size_t kTensorDimensionLimit = 16;

/// Returns the product of dim[0:dim), not including dim.
inline size_t getLeadingDims(const Tensor& tensor, int64_t dim) {
  ET_CHECK_MSG(
      dim >= 0 && dim <= tensor.dim(),
      "Ending dimension %" PRId64
      " should be in the range [0, tensor.dim() %zd].",
      dim,
      ssize_t(tensor.dim()));
  size_t dims = 1;
  for (size_t i = 0; i < dim; ++i) {
    dims *= static_cast<size_t>(tensor.size(i));
  }
  return dims;
}

/// Returns the product of dim[dim+1:].
inline size_t getTrailingDims(const Tensor& tensor, int64_t dim) {
  ET_CHECK_MSG(
      dim >= -1 && dim < tensor.dim(),
      "Starting dimension %" PRId64
      " should be in the range [-1, tensor.dim() -1 %zd).",
      dim,
      ssize_t(tensor.dim()));
  size_t dims = 1;
  for (size_t i = dim + 1; i < tensor.dim(); ++i) {
    dims *= static_cast<size_t>(tensor.size(i));
  }
  return dims;
}

/**
 * Extracts an integer value from a scalar Tensor.
 *
 * @param[in] tensor The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that value.
 *    `false` if a value could not be extracted: either it was not an integer
 *    Scalar Tensor, or the value of that Scalar Tensor could not be represented
 *    by INT_T.
 */
template <
    typename INT_T,
    typename std::enable_if<
        std::is_integral<INT_T>::value && !std::is_same<INT_T, bool>::value,
        bool>::type = true>
bool extract_scalar_tensor(Tensor tensor, INT_T* out_val) {
  if (tensor.numel() != 1) {
    return false;
  }
#define CASE_INT_DTYPE(TENSOR_CTYPE, TENSOR_DTYPE)                     \
  case ScalarType::TENSOR_DTYPE: {                                     \
    const TENSOR_CTYPE val = tensor.const_data_ptr<TENSOR_CTYPE>()[0]; \
    if (val < std::numeric_limits<INT_T>::lowest() ||                  \
        val > std::numeric_limits<INT_T>::max()) {                     \
      return false;                                                    \
    }                                                                  \
    *out_val = static_cast<INT_T>(val);                                \
    return true;                                                       \
  }

  switch (tensor.scalar_type()) {
    ET_FORALL_INT_TYPES(CASE_INT_DTYPE);
    default:
      return false;
  }
#undef CASE_INT_DTYPE
}

/**
 * Extracts a floating point value from a scalar Tensor.
 *
 * @param[in] tensor The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that value.
 *    `false` if a value could not be extracted: either it was not a floating
 *    point Scalar Tensor, or the value of that Scalar Tensor could not be
 *    represented by FLOAT_T.
 */
template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
bool extract_scalar_tensor(Tensor tensor, FLOAT_T* out_val) {
  if (tensor.numel() != 1) {
    return false;
  }
#define CASE_REAL_DTYPE(TENSOR_CTYPE, TENSOR_DTYPE)                    \
  case ScalarType::TENSOR_DTYPE: {                                     \
    /* ET_FORALL_REAL_TYPES guarantees TENSOR_CTYPE is a real type. */ \
    double val =                                                       \
        static_cast<double>(tensor.const_data_ptr<TENSOR_CTYPE>()[0]); \
    if (std::isfinite(val) &&                                          \
        (val < std::numeric_limits<FLOAT_T>::lowest() ||               \
         val > std::numeric_limits<FLOAT_T>::max())) {                 \
      return false;                                                    \
    }                                                                  \
    *out_val = static_cast<FLOAT_T>(val);                              \
    return true;                                                       \
  }

  switch (tensor.scalar_type()) {
    ET_FORALL_REAL_TYPES(CASE_REAL_DTYPE);
    default:
      return false;
  }
#undef CASE_REAL_DTYPE
}

/**
 * Extracts a boolean value from a Scalar.
 *
 * @param[in] scalar The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that
 * value. `false` if a value could not be extracted, i.e. not a boolean
 */
template <
    typename BOOL_T,
    typename std::enable_if<std::is_same<BOOL_T, bool>::value, bool>::type =
        true>
bool extract_scalar_tensor(Tensor tensor, BOOL_T* out_val) {
  if (tensor.scalar_type() != exec_aten::ScalarType::Bool) {
    return false;
  }
  if (tensor.numel() != 1) {
    return false;
  }

  bool val = tensor.const_data_ptr<bool>()[0];

  *out_val = static_cast<BOOL_T>(val);

  return true;
}

/// These APIs should not be used outside of Executor.cpp.
namespace internal {
/**
 * Share t_src's data_ptr with t_dst.
 */
__ET_NODISCARD Error share_tensor_data(
    const exec_aten::Tensor& t_dst,
    const exec_aten::Tensor& t_src);

/**
 * Copy t_src's data_ptr to t_dst.
 */
__ET_NODISCARD Error copy_tensor_data(
    const exec_aten::Tensor& t_dst,
    const exec_aten::Tensor& t_src);

/**
 * Reset tensor's data_ptr, clear all the storage for at::Tensor.
 */
void reset_data_ptr(const exec_aten::Tensor& tensor);

/**
 * Resize tensor impl
 */
__ET_NODISCARD Error resize_tensor_impl(
    exec_aten::TensorImpl* impl,
    exec_aten::ArrayRef<exec_aten::SizesType> new_sizes);
} // namespace internal

/**
 * Resize a tensor to new_sizes, rank must stay the same. Currently does not
 * expand the tensor if new size exceeds the current capacity. Currently
 * fails an ET_CHECK if the tensor cannot be resized.
 *
 * WARNING: Placeholder API until discussion around runtime context is settled,
 * will likely move to be a class method on a TensorResizer object passed in
 * through runtimeContext.
 */
__ET_NODISCARD inline Error resize_tensor(
    exec_aten::Tensor t,
    exec_aten::ArrayRef<exec_aten::SizesType> new_sizes) {
  return internal::resize_tensor_impl(t.unsafeGetTensorImpl(), new_sizes);
}

/**
 * Resize a tensor to new_sizes, rank must stay the same. Currently does not
 * expand the tensor if new size exceeds the current capacity. Currently
 * fails an ET_CHECK if the tensor cannot be resized.
 *
 * WARNING: Placeholder API until discussion around runtime context is settled,
 * will likely move to be a class method on a TensorResizer object passed in
 * through runtimeContext.
 */
template <
    typename T,
    typename std::
        enable_if<!std::is_same<exec_aten::SizesType, T>::value, int>::type = 0>
__ET_NODISCARD inline Error resize_tensor(
    exec_aten::Tensor t,
    exec_aten::ArrayRef<T> new_sizes) {
  // Need to cast the input array to an array of Tensor::SizesType
  std::array<exec_aten::SizesType, kTensorDimensionLimit> new_sizes_casted{};
  size_t new_sizes_ndim = new_sizes.size();
  for (size_t i = 0; i < new_sizes_ndim; ++i) {
    new_sizes_casted[i] = static_cast<exec_aten::SizesType>(new_sizes[i]);
  }

  return internal::resize_tensor_impl(
      t.unsafeGetTensorImpl(), {new_sizes_casted.data(), new_sizes_ndim});
}

/// DEPRECATED: Use `resize_tensor()` instead, which can fail non-fatally.
__ET_DEPRECATED inline void resize(
    exec_aten::Tensor t,
    exec_aten::ArrayRef<exec_aten::SizesType> new_sizes) {
  Error err = resize_tensor(t, new_sizes);
  ET_CHECK_MSG(
      err == Error::Ok, "Could not resize Tensor; see logs for details");
}
/**
 * Get dim_order of a Tensor and write it to out_dim_order.
 * @param tensor The tensor where we want to get dim order from.
 * @param out_dim_order Pointing to an array of DimOrderType where we write dim
 * order into it.
 * @param out_dim_order_size Size of the DimOrderType array.
 */
__ET_NODISCARD Error get_dim_order(
    const exec_aten::Tensor& tensor,
    exec_aten::DimOrderType* out_dim_order,
    size_t out_dim_order_size);

/**
 * Given an n-dimensional coordinate array and an array of tensor strides,
 * calculates the linear index that can be used to retrieve the value at the
 * given coordinates.
 * @param coordinate Pointer to the array of coordinates.
 * @param strides Pointer to the array of strides.
 * @param ndim Number of dimensions in the tensor.
 */
inline size_t calculate_linear_index(
    const exec_aten::SizesType* coordinate,
    const exec_aten::StridesType* strides,
    const size_t ndim) {
  size_t index = 0;
  for (size_t i = 0; i < ndim; i++) {
    index += coordinate[i] * strides[i];
  }
  return index;
}

/// These APIs should not be used outside of Executor.cpp.
namespace internal {
/**
 * Share t_src's data_ptr with t_dst.
 */
__ET_NODISCARD Error share_tensor_data(
    const exec_aten::Tensor& t_dst,
    const exec_aten::Tensor& t_src);

/**
 * Copy t_src's data_ptr to t_dst.
 */
__ET_NODISCARD Error copy_tensor_data(
    const exec_aten::Tensor& t_dst,
    const exec_aten::Tensor& t_src);

/**
 * Reset tensor's data_ptr, clear all the storage for at::Tensor.
 */
void reset_data_ptr(const exec_aten::Tensor& tensor);
} // namespace internal
} // namespace executor
} // namespace torch
