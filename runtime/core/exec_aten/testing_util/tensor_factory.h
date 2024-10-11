// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <cstdint>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/tensor_shape_dynamism.h>
#include <executorch/runtime/platform/assert.h>

#ifdef USE_ATEN_LIB
#include <ATen/ATen.h>
#else // !USE_ATEN_LIB
#include <numeric>
// @nolint PATTERNLINT Ok to use stdlib for this test framework
#include <memory>
// @nolint PATTERNLINT Ok to use stdlib for this test framework
#include <vector>
#endif // !USE_ATEN_LIB

namespace executorch {
namespace runtime {
namespace testing {

namespace internal {

/**
 * Returns the number of elements in the tensor, given the dimension
 * sizes, assuming contiguous data.
 */
inline size_t sizes_to_numel(const std::vector<int32_t>& sizes) {
  size_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

/**
 * Check if given strides is legal under given sizes. In the `make` function,
 * the `strides` shall ensure:
 *  - a. strides.size() == sizes.size()
 *  - b. all strides are positive.
 *  - c. All underlying data be accessed.
 *  - d. All legal indexes can access an underlying data.
 *  - e. No two indexes access a same data.
 *  - f. No out of bounds data can be accessed.
 *
 * @param[in] sizes The sizes of the dimensions of the Tensor.
 * @param[in] strides The desired strides for creating new tensor.
 * @return The strides is legal or not
 */

inline bool check_strides(
    const std::vector<int32_t> sizes,
    const std::vector<executorch::aten::StridesType> strides) {
  if (sizes.size() != strides.size()) {
    // The length of stride vector shall equal to size vector.
    return false;
  }

  if (strides.size() == 0) {
    // Both sizes and strides are empty vector. Legal!
    return true;
  }

  // Check if input non-empty strides is legal. The defination of legal is in
  // the comment above function. To check it, we first reformat the strides into
  // contiguous style, in where the strides should be sorted from high to low.
  // Then rearrange the size based on same transformation. After that, we can
  // check if strides[i] == strides[i + 1] * sizes[i + 1] for all i in
  // [0, sizes.size() - 1) and strides[sizes.size() - 1] == 1

  // Get the mapping between current strides and sorted strides (from high to
  // low, if equal then check if correspond size is 1 or 0 in same dimension)
  // e.g. a = tensor(3, 2, 1).permute(2, 1, 0), a.size() == (1, 2, 3) and
  // a.strides == (1, 1, 2). We want to sort create a mapping to make the
  // sorted_stride as (2, 1, 1) while sorted_size == (3, 2, 1)
  std::vector<std::int32_t> sorted_idx(sizes.size());
  for (size_t i = 0; i < sizes.size(); i++) {
    sorted_idx[i] = i;
  }
  std::sort(
      sorted_idx.begin(),
      sorted_idx.end(),
      [&](const int32_t& a, const int32_t& b) {
        if (strides[a] != strides[b]) {
          return strides[a] > strides[b];
        } else {
          // When strides equal to each other, put the index whose
          // coresponding size equal to 0 or 1 to the right. Update the rule to
          // the following comparsion to circumvent strict weak ordering.
          return (sizes[a] ? sizes[a] : 1) > (sizes[b] ? sizes[b] : 1);
        }
      });

  // Use the mapping to rearrange the sizes and strides
  std::vector<std::int32_t> sorted_sizes(sizes.size());
  std::vector<std::int32_t> sorted_strides(sizes.size());
  for (size_t i = 0; i < sizes.size(); i++) {
    sorted_sizes[i] = sizes[sorted_idx[i]] == 0 ? 1 : sizes[sorted_idx[i]];
    sorted_strides[i] = strides[sorted_idx[i]];
  }

  // All strides should be positive. We have sorted it mainly based on strides,
  // so sorted_strides[-1] has lowest value.
  if (sorted_strides[strides.size() - 1] <= 0) {
    return false;
  }

  // Check if strides is legal
  bool legal = sorted_strides[strides.size() - 1] == 1;
  for (size_t i = 0; i < strides.size() - 1 && legal; i++) {
    legal = legal &&
        (sorted_strides[i] == sorted_strides[i + 1] * sorted_sizes[i + 1]);
  }

  return legal;
}

/**
 * Check that a given dim order array is valid. A dim order array is valid if
 * each value from 0 to sizes.size() - 1 appears exactly once in the dim_order
 * array.
 */
inline bool check_dim_order(
    const std::vector<int32_t>& sizes,
    const std::vector<uint8_t>& dim_order) {
  if (sizes.size() != dim_order.size()) {
    return false;
  }
  size_t gauss_sum = 0;
  std::vector<int> count(dim_order.size(), 0);
  for (int i = 0; i < dim_order.size(); i++) {
    if (dim_order[i] < 0 || dim_order[i] >= sizes.size()) {
      return false;
    }
    gauss_sum += static_cast<size_t>(dim_order[i]) + 1;
  }
  // Use the gaussian sum to verify each dim appears exactly once
  size_t expected_sum = (sizes.size() * (sizes.size() + 1)) / 2;
  if (gauss_sum != expected_sum) {
    return false;
  }

  return true;
}

inline std::vector<executorch::aten::StridesType> strides_from_dim_order(
    const std::vector<int32_t>& sizes,
    const std::vector<uint8_t>& dim_order) {
  bool legal = check_dim_order(sizes, dim_order);
  ET_CHECK_MSG(legal, "The input dim_order variable is illegal.");

  size_t ndim = sizes.size();
  std::vector<executorch::aten::StridesType> strides(ndim);
  strides[dim_order[ndim - 1]] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    uint8_t cur_dim = dim_order[i];
    uint8_t next_dim = dim_order[i + 1];
    strides[cur_dim] = (!sizes[next_dim]) ? strides[next_dim]
                                          : strides[next_dim] * sizes[next_dim];
  }
  return strides;
}

inline std::vector<uint8_t> channels_last_dim_order(size_t dims) {
  ET_CHECK_MSG(
      dims >= 4 && dims <= 5,
      "Channels last dim order only valid for 4-dim and 5-dim tensors!");

  std::vector<uint8_t> dim_order(dims);
  // Channels is always assigned to dim 1
  dim_order[dims - 1] = 1;

  dim_order[0] = 0;
  int d = 1;
  while (d < dims - 1) {
    dim_order[d] = d + 1;
    d++;
  }
  return dim_order;
}

} // namespace internal

#ifdef USE_ATEN_LIB

// Note that this USE_ATEN_LIB section uses ATen-specific namespaces instead of
// exec_aten because we know that we're working with ATen, and many of these
// names aren't mapped into executorch::aten::.

namespace internal {

// This wrapper lets us override the C type associated with some ScalarType
// values while using the defaults for everything else.
template <c10::ScalarType DTYPE>
struct ScalarTypeToCppTypeWrapper {
  using ctype = typename c10::impl::ScalarTypeToCPPTypeT<DTYPE>;
};

// Use a C type of `uint8_t` instead of `bool`. The C type will be used to
// declare a `std::vector<CTYPE>`, and `std::vector<bool>` is often optimized to
// store a single bit per entry instead of using an array of separate `bool`
// elements. Since the tensor data will point into the vector, it needs to use
// one byte per element.
template <>
struct ScalarTypeToCppTypeWrapper<c10::ScalarType::Bool> {
  using ctype = uint8_t;
};

} // namespace internal

template <at::ScalarType DTYPE>
class TensorFactory {
 public:
  /*
   * The C types that backs the associated DTYPE. E.g., `float` for
   * `ScalarType::Float`.
   */

  /**
   * Used for the vector provided to the factory functions. May differ
   * from the type usually associate with the ScalarType.
   *
   * Used for the vector<> parameters passed to the factory functions.
   */
  using ctype = typename internal::ScalarTypeToCppTypeWrapper<DTYPE>::ctype;

  /**
   * The official C type for the scalar type. Used when accessing elements
   * of a constructed Tensor.
   */
  using true_ctype = typename c10::impl::ScalarTypeToCPPTypeT<DTYPE>;

  TensorFactory() = default;

  /**
   * Returns a new Tensor with the specified shape, data and stride.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data that the Tensor should be initialized with. The
   *     size of this vector must be equal to the product of the elements of
   *     `sizes`.
   * @param[in] strides The strides for each dimensions of the Tensor. If empty
   *     or not specificed, the function will return a contiguous tensor based
   *     on data and size. If not, the strides shall follow the rules:
   *            - a. strides.size() == sizes.size().
   *            - b. all strides are positive.
   *            - c. All underlying data be accessed.
   *            - d. All legal indexes can access an underlying data.
   *            - e. No two indexes access a same data.
   *            - f. No out of bounds data can be accessed.
   *
   * @return A new Tensor with the specified shape and data.
   */
  at::Tensor make(
      const std::vector<int32_t>& sizes,
      const std::vector<ctype>& data,
      const std::vector<executorch::aten::StridesType> strides = {},
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto expected_numel = internal::sizes_to_numel(sizes);
    ET_CHECK_MSG(
        expected_numel == data.size(),
        "Number of data elements %zd "
        "does not match expected number of elements %zd",
        data.size(),
        expected_numel);

    at::Tensor t;
    if (strides.empty()) {
      t = zeros(sizes);
    } else {
      bool legal = internal::check_strides(sizes, strides);
      ET_CHECK_MSG(legal, "The input strides variable is illegal.");

      t = empty_strided(sizes, strides);
    }
    if (t.nbytes() > 0) {
      memcpy(t.template data<true_ctype>(), data.data(), t.nbytes());
    }
    return t;
  }

  /**
   * Returns a new Tensor with the specified shape, data and dim order.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data that the Tensor should be initialized with. The
   *     size of this vector must be equal to the product of the elements of
   *     `sizes`.
   * @param[in] dim_order The dim order describing how tensor memory is laid
   * out. If empty or not specificed, the function will use a contiguous dim
   * order of {0, 1, 2, 3, ...}
   *
   * @return A new Tensor with the specified shape and data.
   */
  at::Tensor make_with_dimorder(
      const std::vector<int32_t>& sizes,
      const std::vector<ctype>& data,
      const std::vector<uint8_t> dim_order = {},
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto expected_numel = internal::sizes_to_numel(sizes);
    ET_CHECK_MSG(
        expected_numel == data.size(),
        "Number of data elements %zd "
        "does not match expected number of elements %zd",
        data.size(),
        expected_numel);

    at::Tensor t;
    if (dim_order.empty()) {
      t = zeros(sizes);
    } else {
      auto strides = internal::strides_from_dim_order(sizes, dim_order);
      t = empty_strided(sizes, strides);
    }
    if (t.nbytes() > 0) {
      memcpy(t.template data<true_ctype>(), data.data(), t.nbytes());
    }
    return t;
  }

  /**
   * Returns a new Tensor with the specified shape and data in channels last
   * memory layout.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data that the Tensor should be initialized with. The
   *     size of this vector must be equal to the product of the elements of
   *     `sizes`.
   *
   * @return A new Tensor with the specified shape and data.
   */
  at::Tensor make_channels_last(
      const std::vector<int32_t>& sizes,
      const std::vector<ctype>& data,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    return make_with_dimorder(
        sizes, data, internal::channels_last_dim_order(sizes.size()), dynamism);
  }

  /**
   * Given data in contiguous memory format, returns a new Tensor with the
   * specified shape and the same data but in channels last memory format.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data in contiguous memory format that the Tensor should
   * be initialized with. The size of this vector must be equal to the product
   * of the elements of `sizes`.
   *
   * @return A new Tensor with the specified shape and data in channls last
   * memory format.
   */
  at::Tensor channels_last_like(
      const at::Tensor& input,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    ET_CHECK_MSG(
        input.sizes().size() == 4, "Only 4D tensors can be channels last");

    const std::vector<int32_t> sizes(
        input.sizes().begin(), input.sizes().end());

    std::vector<uint8_t> contiguous_dim_order(sizes.size());
    for (uint8_t i = 0; i < sizes.size(); i++) {
      contiguous_dim_order[i] = i;
    }
    std::vector<executorch::aten::StridesType> contiguous_strides =
        internal::strides_from_dim_order(sizes, contiguous_dim_order);

    for (int32_t i = 0; i < input.dim(); i++) {
      ET_CHECK_MSG(
          input.strides()[i] == contiguous_strides[i],
          "Input tensor is not contiguous");
    }

    int32_t N = sizes[0];
    int32_t C = sizes[1];
    int32_t H = sizes[2];
    int32_t W = sizes[3];

    std::vector<ctype> contiguous_data(
        input.data_ptr<ctype>(), input.data_ptr<ctype>() + input.numel());
    std::vector<ctype> channels_last_data(
        N * C * H * W); // Create a new blob with the same total size to contain
                        // channels_last data
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t c = 0; c < C; ++c) {
        for (int32_t h = 0; h < H; ++h) {
          for (int32_t w = 0; w < W; ++w) {
            // Calculate the index in the original blob
            int32_t old_index = ((n * C + c) * H + h) * W + w;
            // Calculate the index in the new blob
            int32_t new_index = ((n * H + h) * W + w) * C + c;
            // Copy the data
            channels_last_data[new_index] = contiguous_data[old_index];
          }
        }
      }
    }

    return make_with_dimorder(
        sizes,
        channels_last_data,
        internal::channels_last_dim_order(sizes.size()),
        dynamism);
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous
   * data will all elements set to `value`.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] value The value of all elements of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  at::Tensor full(
      const std::vector<int32_t>& sizes,
      ctype value,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto sizes64 = vec_32_to_64(sizes);
    return at::full(at::IntArrayRef(sizes64), value, at::dtype(DTYPE));
  }

  /**
   * Returns a new Tensor with the specified shape, containing channels-last
   * contiguous data will all elements set to `value`.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] value The value of all elements of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  at::Tensor full_channels_last(
      const std::vector<int32_t>& sizes,
      ctype value,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto sizes64 = vec_32_to_64(sizes);
    return at::full(at::IntArrayRef(sizes64), value, at::dtype(DTYPE))
        .to(at::MemoryFormat::ChannelsLast);
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous data
   * with all `0` elements.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  at::Tensor zeros(
      const std::vector<int32_t>& sizes,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto sizes64 = vec_32_to_64(sizes);
    return at::zeros(at::IntArrayRef(sizes64), at::dtype(DTYPE));
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous data
   * with all `1` elements.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  at::Tensor ones(
      const std::vector<int32_t>& sizes,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto sizes64 = vec_32_to_64(sizes);
    return at::ones(at::IntArrayRef(sizes64), at::dtype(DTYPE));
  }

  /**
   * Returns a new Tensor with the same shape as the input tensor, containing
   * contiguous data with all `0` elements.
   *
   * @param[in] input The tensor that supplies the shape of the new Tensor.
   * @return A new Tensor with the specified shape.
   */
  at::Tensor zeros_like(
      const at::Tensor& input,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    std::vector<int64_t> sizes64 = {input.sizes().begin(), input.sizes().end()};
    return at::full(at::IntArrayRef(sizes64), 0, at::dtype(DTYPE));
  }

  /**
   * Returns a new Tensor with the same shape as the input tensor, containing
   * contiguous data with all `1` elements.
   *
   * @param[in] input The tensor that supplies the shape of the new Tensor.
   * @return A new Tensor with the specified shape.
   */
  at::Tensor ones_like(
      const at::Tensor& input,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    std::vector<int64_t> sizes64 = {input.sizes().begin(), input.sizes().end()};
    return at::full(at::IntArrayRef(sizes64), 1, at::dtype(DTYPE));
  }

 private:
  /// Copies an int32_t vector into a new int64_t vector.
  static std::vector<int64_t> vec_32_to_64(const std::vector<int32_t>& in) {
    std::vector<int64_t> out{};
    out.reserve(in.size());
    for (auto i : in) {
      out.push_back(i);
    }
    return out;
  }

  /**
   * Returns a new Tensor with the specified shape and stride.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] strides The strides for each dimensions of the Tensor
   * @return A new Tensor with the specified shape and strides.
   */
  at::Tensor empty_strided(
      const std::vector<int32_t>& sizes,
      const std::vector<executorch::aten::StridesType>& strides,
      ET_UNUSED TensorShapeDynamism dynamism =
          TensorShapeDynamism::DYNAMIC_UNBOUND) {
    auto sizes64 = vec_32_to_64(sizes);
    return at::empty_strided(
        sizes64,
        strides,
        DTYPE,
        /*layout_opt=*/at::Layout::Strided,
        /*device_opt=*/at::Device(at::DeviceType::CPU),
        /*pin_memory_opt=*/false);
  }
};

#else // !USE_ATEN_LIB

namespace {
/*
 * Dimension order represents how dimensions are laid out in memory,
 * starting from the inner-most to the outer-most dimension.
 * Thus, the conversion from strides is done by sorting the strides
 * from larger to smaller since the dimension with the largest stride
 * is the outer-most and the dimension with the smallest stride is the
 inner-most.
 * For example, tensor with sizes = (3, 5, 2) and strides = (5, 1, 15), implies
 * dimension order of (2, 0, 1), because 2nd dimension has the biggest stride of
 15,
 * followed by 0th dimension with stride of 5 and then innermost dimension being
 the 1st
 * dimension with size of 1. This order of (2, 0, 1) can be obtained
 * by sorting strides from large to smaller.

 * When strides do not convey dimension order unambiguously, dimension order
 * returned is dependent on stability of sort. We employ stable sort to preserve
 * original order. Thus when strides = (4, 3, 1, 1) returned value is (0, 1, 2,
 3)
 * Another example is: sizes = (1, 3, 1, 1) with strides = (3, 1, 3, 3),
 returned
 * value is (0, 2, 3, 1)
*/
// copied from
// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
// TODO: Add assert for strides[i] != 0 because strides of 0 is really used,
// by pytorch/aten, to convey broadcasting dim.

inline std::vector<uint8_t> dim_order_from_stride(
    const std::vector<int32_t>& v) {
  std::vector<uint8_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  stable_sort(indices.begin(), indices.end(), [&v](size_t i1, size_t i2) {
    return v[i1] > v[i2];
  });
  return indices;
}

inline void validate_strides(
    const std::vector<int32_t>& sizes,
    const std::vector<int32_t>& strides) {
  if (sizes.size() != strides.size()) {
    ET_CHECK_MSG(false, "Stride and sizes are not equal in length");
  }
  for (const auto& s : strides) {
    if (s == 0) {
      ET_CHECK_MSG(false, "Stride value of 0 is not supported");
    }
  }
  // No two dimensions can have same stride value
  for (int32_t i = 0; i < strides.size(); ++i) {
    for (int32_t j = i + 1; j < strides.size(); ++j) {
      if ((sizes[i] == 0) || (sizes[j] == 0) ||
          ((sizes[i] == 1) || (sizes[j] == 1))) {
        continue;
      }
      if ((strides[i] == strides[j])) {
        ET_CHECK_MSG(
            false,
            "Stride value and size dont comply at index %d."
            " strides[%d]: %d, strides[%d] = %d, sizes[%d] = %d, sizes[%d] = %d",
            i,
            i,
            strides[i],
            j,
            strides[j],
            i,
            sizes[i],
            j,
            sizes[j]);
      }
    }
  }
}

} // namespace

// Note that this !USE_ATEN_LIB section uses ExecuTorch-specific namespaces
// instead of exec_aten to make it clear that we're dealing with ETensor, and
// because many of these names aren't mapped into executorch::aten::.

namespace internal {

// This wrapper lets us override the C type associated with some ScalarType
// values while using the defaults for everything else.
template <torch::executor::ScalarType DTYPE>
struct ScalarTypeToCppTypeWrapper {
  using ctype =
      typename ::executorch::runtime::ScalarTypeToCppType<DTYPE>::type;
};

// Use a C type of `uint8_t` instead of `bool`. The C type will be used to
// declare a `std::vector<CTYPE>`, and `std::vector<bool>` is often optimized to
// store a single bit per entry instead of using an array of separate `bool`
// elements. Since the tensor data will point into the vector, it needs to use
// one byte per element.
template <>
struct ScalarTypeToCppTypeWrapper<torch::executor::ScalarType::Bool> {
  using ctype = uint8_t;
};

// To allow implicit conversion between simple types to `ctype`
#define SPECIALIZE_ScalarTypeToCppTypeWrapper(CTYPE, DTYPE)               \
  template <>                                                             \
  struct ScalarTypeToCppTypeWrapper<torch::executor::ScalarType::DTYPE> { \
    using ctype = typename CTYPE::underlying;                             \
  };

ET_FORALL_QINT_TYPES(SPECIALIZE_ScalarTypeToCppTypeWrapper)

#undef SPECIALIZE_ScalarTypeToCppTypeWrapper

} // namespace internal

/**
 * A helper class for creating Tensors, simplifying memory management.
 *
 * NOTE: A given TensorFactory instance owns the memory pointed to by all
 * Tensors that it creates, and must live longer than those Tensors.
 *
 * Example:
 * @code{.cpp}
 * // A factory instance will create Tensors of a single dtype.
 * TensorFactory<ScalarType::Int> tf;
 *
 * // You can create more factories if you need tensors of multiple
 * // dtypes.
 * TensorFactory<ScalarType::Float> tf_float;
 *
 * // The factory will copy the vectors provided to it, letting callers provide
 * // inline literals.
 * Tensor t1 = tf.make(
 *     {2, 2}, // sizes
 *     {1, 2, 3, 4}); // data
 *
 * // There are helpers for creating Tensors with all 1 or 0 elements.
 * Tensor z = tf.zeros({2, 2});
 * Tensor o = tf_float.ones({2, 2});
 *
 * // Sometimes it's helpful to share parameters.
 * std::vector<int32_t> sizes = {2, 2};
 * Tensor t3 = tf.make(sizes, {1, 2, 3, 4});
 * Tensor t4 = tf.ones(sizes);
 *
 * // But remember that the inputs are copied, so providing the same data vector
 * // to two Tensors will not share the same underlying data.
 * std::vector<int> data = {1, 2, 3, 4};
 * Tensor t5 = tf.make(sizes, data);
 * Tensor t6 = tf.make(sizes, data);
 * t5.mutable_data_ptr<int>()[0] = 99;
 * EXPECT_NE(t5, t6);
 * @endcode
 *
 * @tparam DTYPE The dtype of Tensors created by this factory, as a ScalarType
 *     value like `ScalarType::Int`.
 */
template <torch::executor::ScalarType DTYPE>
class TensorFactory {
 public:
  /**
   * The C type that backs the associated DTYPE. E.g., `float` for
   * `ScalarType::Float`.
   */
  using ctype = typename internal::ScalarTypeToCppTypeWrapper<DTYPE>::ctype;

  TensorFactory() = default;

  /**
   * Returns a new Tensor with the specified shape, data and stride.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data that the Tensor should be initialized with. The
   *     size of this vector must be equal to the product of the elements of
   *     `sizes`.
   * @param[in] strides The strides for each dimensions of the Tensor. If empty
   *     or not specificed, the function will return a contiguous tensor based
   *     on data and size. If not, the strides shall follow the rules:
   *            - a. strides.size() == sizes.size().
   *            - b. all strides are positive.
   *            - c. All underlying data be accessed.
   *            - d. All legal indexes can access an underlying data.
   *            - e. No two indexes access a same data.
   *            - f. No out of bounds data can be accessed.
   *
   * @return A new Tensor with the specified shape and data.
   */
  torch::executor::Tensor make(
      const std::vector<int32_t>& sizes,
      const std::vector<ctype>& data,
      const std::vector<executorch::aten::StridesType> strides = {},
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    std::vector<int32_t> default_strides;
    // Generate strides from the tensor dimensions, assuming contiguous data if
    // given strides is empty.
    if (!sizes.empty() && strides.empty()) {
      default_strides.resize(sizes.size(), 1);
      for (size_t i = sizes.size() - 1; i > 0; --i) {
        // For sizes[i] == 0, treat it as 1 to be consistent with core Pytorch
        auto sizes_i = sizes[i] ? sizes[i] : 1;
        default_strides[i - 1] = default_strides[i] * sizes_i;
      }
    }
    auto& actual_strides = default_strides.empty() ? strides : default_strides;
    validate_strides(sizes, actual_strides);
    auto dim_order = dim_order_from_stride(actual_strides);

    auto expected_numel = internal::sizes_to_numel(sizes);
    ET_CHECK_MSG(
        expected_numel == data.size(),
        "Number of data elements %zd "
        "does not match expected number of elements %zd",
        data.size(),
        expected_numel);

    bool legal = internal::check_strides(sizes, actual_strides);
    ET_CHECK_MSG(legal, "The input strides variable is illegal.");

    memory_.emplace_back(std::make_unique<TensorMemory>(
        sizes, data, dim_order, actual_strides, dynamism));
    return torch::executor::Tensor(&memory_.back()->impl_);
  }

  /**
   * Returns a new Tensor with the specified shape, data and dim order.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data that the Tensor should be initialized with. The
   *     size of this vector must be equal to the product of the elements of
   *     `sizes`.
   * @param[in] dim_order The dim order describing how tensor memory is laid
   * out. If empty or not specificed, the function will use a contiguous dim
   * order of {0, 1, 2, 3, ...}
   *
   * @return A new Tensor with the specified shape and data.
   */
  torch::executor::Tensor make_with_dimorder(
      const std::vector<int32_t>& sizes,
      const std::vector<ctype>& data,
      const std::vector<uint8_t> dim_order = {},
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    std::vector<uint8_t> default_dim_order;
    // Generate strides from the tensor dimensions, assuming contiguous data if
    // given strides is empty.
    if (!sizes.empty() && dim_order.empty()) {
      default_dim_order.resize(sizes.size(), 1);
      for (size_t i = 0; i < sizes.size(); ++i) {
        default_dim_order[i] = i;
      }
    }
    auto& actual_dim_order =
        default_dim_order.empty() ? dim_order : default_dim_order;

    auto strides = internal::strides_from_dim_order(sizes, actual_dim_order);

    auto expected_numel = internal::sizes_to_numel(sizes);
    ET_CHECK_MSG(
        expected_numel == data.size(),
        "Number of data elements %zd "
        "does not match expected number of elements %zd",
        data.size(),
        expected_numel);

    memory_.emplace_back(std::make_unique<TensorMemory>(
        sizes, data, actual_dim_order, strides, dynamism));
    return torch::executor::Tensor(&memory_.back()->impl_);
  }

  /**
   * Returns a new Tensor with the specified shape and data in channels last
   * memory format.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data that the Tensor should be initialized with. The
   *     size of this vector must be equal to the product of the elements of
   *     `sizes`.
   *
   * @return A new Tensor with the specified shape and data.
   */
  torch::executor::Tensor make_channels_last(
      const std::vector<int32_t>& sizes,
      const std::vector<ctype>& data,
      const std::vector<uint8_t> dim_order = {},
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    return make_with_dimorder(
        sizes, data, internal::channels_last_dim_order(sizes.size()), dynamism);
  }

  /**
   * Given data in contiguous memory format, returns a new Tensor with the
   * specified shape and the same data but in channels last memory format.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] data The data in contiguous memory format that the Tensor should
   * be initialized with. The size of this vector must be equal to the product
   * of the elements of `sizes`.
   *
   * @return A new Tensor with the specified shape and data in channls last
   * memory format.
   */
  torch::executor::Tensor channels_last_like(
      const torch::executor::Tensor& input,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    const std::vector<int32_t> sizes(
        input.sizes().begin(), input.sizes().end());

    ET_CHECK_MSG(sizes.size() == 4, "Only 4D tensors can be channels last");
    ET_CHECK_MSG(
        is_contiguous_dim_order(input.dim_order().data(), input.dim()) == true,
        "Input tensor is not contiguous");
    int32_t N = sizes[0];
    int32_t C = sizes[1];
    int32_t H = sizes[2];
    int32_t W = sizes[3];

    std::vector<ctype> contiguous_data(
        input.data_ptr<ctype>(), input.data_ptr<ctype>() + input.numel());
    std::vector<ctype> channels_last_data(
        N * C * H * W); // Create a new blob with the same total size to contain
                        // channels_last data
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t c = 0; c < C; ++c) {
        for (int32_t h = 0; h < H; ++h) {
          for (int32_t w = 0; w < W; ++w) {
            // Calculate the index in the original blob
            int32_t old_index = ((n * C + c) * H + h) * W + w;
            // Calculate the index in the new blob
            int32_t new_index = ((n * H + h) * W + w) * C + c;
            // Copy the data
            channels_last_data[new_index] = contiguous_data[old_index];
          }
        }
      }
    }

    return make_with_dimorder(
        sizes,
        channels_last_data,
        internal::channels_last_dim_order(sizes.size()),
        dynamism);
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous data
   * will all elements set to `value`.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] value The value of all elements of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor full(
      const std::vector<int32_t>& sizes,
      ctype value,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    std::vector<ctype> data(internal::sizes_to_numel(sizes), value);
    return make(sizes, data, /* empty strides */ {}, dynamism);
  }

  /**
   * Returns a new Tensor with the specified shape, containing channels last
   * contiguous data will all elements set to `value`.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @param[in] value The value of all elements of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor full_channels_last(
      const std::vector<int32_t>& sizes,
      ctype value,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    std::vector<ctype> data(internal::sizes_to_numel(sizes), value);
    return make_with_dimorder(
        sizes, data, internal::channels_last_dim_order(sizes.size()), dynamism);
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous data
   * in channels last memory format with all `0` elements.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor zeros_channels_last(
      const std::vector<int32_t>& sizes,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    return full_channels_last(sizes, 0, dynamism);
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous data
   * in contiguous memory format with all `0` elements.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor zeros(
      const std::vector<int32_t>& sizes,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    return full(sizes, 0, dynamism);
  }

  /**
   * Returns a new Tensor with the specified shape, containing contiguous data
   * with all `1` elements.
   *
   * @param[in] sizes The sizes of the dimensions of the Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor ones(
      const std::vector<int32_t>& sizes,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    return full(sizes, 1, dynamism);
  }

  /**
   * Returns a new Tensor with the same shape as the input tensor, containing
   * contiguous data with all `0` elements.
   *
   * @param[in] input The tensor that supplies the shape of the new Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor zeros_like(
      const torch::executor::Tensor& input,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    std::vector<int32_t> sizes = {input.sizes().begin(), input.sizes().end()};
    return full(sizes, 0, dynamism);
  }

  /**
   * Returns a new Tensor with the same shape as the input tensor, containing
   * contiguous data with all `1` elements.
   *
   * @param[in] input The tensor that supplies the shape of the new Tensor.
   * @return A new Tensor with the specified shape.
   */
  torch::executor::Tensor ones_like(
      const torch::executor::Tensor& input,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    std::vector<int32_t> sizes = {input.sizes().begin(), input.sizes().end()};
    return full(sizes, 1, dynamism);
  }

 private:
  /**
   * Owns all backing memory for a single Tensor.
   */
  struct TensorMemory {
    TensorMemory(
        const std::vector<int32_t>& sizes,
        const std::vector<ctype>& data,
        const std::vector<uint8_t>& dim_order,
        const std::vector<int32_t>& strides,
        TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC)
        : sizes_(sizes),
          data_(data),
          dim_order_(dim_order),
          strides_(strides),
          impl_(
              DTYPE,
              /*dim=*/sizes_.size(),
              sizes_.data(),
              data_.data(),
              dim_order_.data(),
              strides_.data(),
              dynamism) {}

    std::vector<int32_t> sizes_;
    std::vector<ctype> data_;
    std::vector<uint8_t> dim_order_;
    std::vector<executorch::aten::StridesType> strides_;
    torch::executor::TensorImpl impl_;
  };

  /**
   * The memory pointed to by Tensors created by this factory. This is a vector
   * of pointers so that the TensorMemory objects won't move if the vector needs
   * to resize/realloc.
   */
  std::vector<std::unique_ptr<TensorMemory>> memory_;
};

#endif // !USE_ATEN_LIB

/**
 * A helper class for creating TensorLists, simplifying memory management.
 *
 * NOTE: A given TensorListFactory owns the memory pointed to by all TensorLists
 * (and Tensors they contain), and must live longer than those TensorLists and
 * Tensors.
 */
template <executorch::aten::ScalarType DTYPE>
class TensorListFactory final {
 public:
  TensorListFactory() = default;
  ~TensorListFactory() = default;

  /**
   * Returns a TensorList containing Tensors with the same shapes as the
   * provided Tensors, but filled with zero elements. The dtypes of the template
   * entries are ignored.
   */
  executorch::aten::TensorList zeros_like(
      const std::vector<executorch::aten::Tensor>& templates) {
    memory_.emplace_back(
        std::make_unique<std::vector<executorch::aten::Tensor>>());
    auto& vec = memory_.back();
    std::for_each(
        templates.begin(),
        templates.end(),
        [&](const executorch::aten::Tensor& t) {
          vec->push_back(tf_.zeros_like(t));
        });
    return executorch::aten::TensorList(vec->data(), vec->size());
  }

 private:
  TensorFactory<DTYPE> tf_;
  /**
   * The memory pointed to by TensorLists created by this factory. This is a
   * vector of pointers so that the elements won't move if the vector needs to
   * resize/realloc.
   */
  std::vector<std::unique_ptr<std::vector<executorch::aten::Tensor>>> memory_;
};

} // namespace testing
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
namespace testing {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::testing::TensorFactory;
using ::executorch::runtime::testing::TensorListFactory;
} // namespace testing
} // namespace executor
} // namespace torch
