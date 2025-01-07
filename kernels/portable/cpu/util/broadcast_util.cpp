/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/repeat_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <string.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

void free_broadcast_tensor(const Tensor& broadcast_tensor) {
  free((void*)broadcast_tensor.const_data_ptr());
  free((void*)broadcast_tensor.sizes().data());
  free((void*)broadcast_tensor.dim_order().data());
  free((void*)broadcast_tensor.strides().data());
  free(broadcast_tensor.unsafeGetTensorImpl());
}

namespace {

Tensor make_tensor(
    const ArrayRef<Tensor::SizesType>& sizes,
    const ArrayRef<Tensor::DimOrderType>& dim_order,
    const ArrayRef<Tensor::StridesType>& strides,
    const ScalarType& dtype) {
  int dim = sizes.size();
  int size_nbytes = dim * sizeof(Tensor::SizesType);
  void* size_data_ptr = malloc(size_nbytes);
  ET_CHECK_MSG(size_data_ptr != nullptr, "Failed to malloc for size bytes");
  memcpy(size_data_ptr, sizes.data(), size_nbytes);

  // TODO(T145322324): can we remove the static cast once size is unsigned?
  size_t dim_order_nbytes =
      static_cast<size_t>(dim) * sizeof(Tensor::DimOrderType);
  // This is leaking memory?
  // TODO(T147221312)
  void* dim_order_data_ptr = malloc(dim_order_nbytes);
  ET_CHECK_MSG(
      dim_order_data_ptr != nullptr, "Failed to malloc for dim order bytes");
  memcpy(dim_order_data_ptr, dim_order.data(), dim_order_nbytes);

  int strides_nbytes = dim * sizeof(Tensor::StridesType);
  void* strides_data_ptr = malloc(strides_nbytes);
  ET_CHECK_MSG(
      strides_data_ptr != nullptr, "Failed to malloc for strides bytes");
  memcpy(strides_data_ptr, strides.data(), strides_nbytes);

  auto tensor_impl = static_cast<TensorImpl*>(malloc(sizeof(TensorImpl)));
  ET_CHECK_MSG(tensor_impl != nullptr, "Failed to malloc for data TensorImpl");

  new (tensor_impl) TensorImpl(
      dtype,
      dim,
      reinterpret_cast<Tensor::SizesType*>(size_data_ptr),
      nullptr,
      reinterpret_cast<Tensor::DimOrderType*>(dim_order_data_ptr),
      reinterpret_cast<Tensor::StridesType*>(strides_data_ptr));

  void* data_ptr = malloc(tensor_impl->nbytes());
  ET_CHECK_MSG(data_ptr != nullptr, "Failed to malloc for data buffer");
  tensor_impl->set_data(data_ptr);

  return Tensor{tensor_impl};
}

} // namespace

bool tensor_is_broadcastable_to(
    const exec_aten::ArrayRef<Tensor::SizesType> broadcast_from_shape,
    const exec_aten::ArrayRef<Tensor::SizesType> broadcast_to_shape) {
  bool feasible_bcast = true;

  if (broadcast_to_shape.size() < broadcast_from_shape.size()) {
    return false;
  }

  for (int i = broadcast_to_shape.size() - 1,
           j = broadcast_from_shape.size() - 1;
       j >= 0;
       --i, --j) {
    auto broadcast_to_s = broadcast_to_shape[i],
         broadcast_from_s = broadcast_from_shape[j];
    feasible_bcast &=
        broadcast_to_s == broadcast_from_s || broadcast_from_s == 1;
    if (!feasible_bcast) {
      return false;
    }
  }

  return feasible_bcast;
}

bool tensor_is_broadcastable_to(
    const Tensor& broadcast_from,
    const Tensor& broadcast_to) {
  return tensor_is_broadcastable_to(
      broadcast_from.sizes(), broadcast_to.sizes());
}

bool tensors_are_broadcastable_between(
    const exec_aten::ArrayRef<Tensor::SizesType> a_shape,
    const exec_aten::ArrayRef<Tensor::SizesType> b_shape) {
  auto a_dim = a_shape.size();
  auto b_dim = b_shape.size();

  // Although the documentation (https://fburl.com/n9wl4d0o) says that tensor
  // with 0-dim can not be broadcasted, experiment shows that actually it can
  // (https://www.internalfb.com/intern/px/p/2pMT0). So here we do not test the
  // dimension.

  for (int a_index = a_dim - 1, b_index = b_dim - 1;
       a_index >= 0 && b_index >= 0;
       a_index--, b_index--) {
    if (a_shape[a_index] == b_shape[b_index] || a_shape[a_index] == 1 ||
        b_shape[b_index] == 1) {
      continue;
    }
    return false;
  }

  return true;
}

bool tensors_are_broadcastable_between(const Tensor& a, const Tensor& b) {
  return tensors_are_broadcastable_between(a.sizes(), b.sizes());
}

// Broadcast tensor broadcast_from to match broadcast_to's shape, and return the
// broadcasted tensor.
Tensor broadcast_tensor(
    const Tensor& broadcast_from,
    const Tensor& broadcast_to) {
  auto broadcast_to_shape = broadcast_to.sizes();
  auto broadcast_from_shape = broadcast_from.sizes();
  auto broadcast_to_dim_order = broadcast_to.dim_order();
  auto broadcast_to_strides = broadcast_to.strides();

  // First check if broadcast_from is broadcastable to broadcast_to.
  // Essentially, we can broadcast broadcast_from if it meets three conditions
  // along any dimension i: (1) broadcast_to[i] = broadcast_from[i]; (2)
  // broadcast_from[i] = 1; or (3) broadcast_from[i] does not exist.
  // for torch.tensor(11), the dim is 0 so we can't use *.sizes().empty() to
  // check.
  ET_CHECK_MSG(
      broadcast_from.numel() != 0 || !(broadcast_from).sizes().empty(),
      "Input tensor must be non-empty");
  // there would never be a broadcast_to with only 1 element, so we are checking
  // dim here.
  ET_CHECK_MSG(
      !(broadcast_to).sizes().empty(), "Input tensor must be non-empty");
  ET_CHECK_MSG(
      broadcast_to_shape.size() >= broadcast_from_shape.size(),
      "For broadcast, tensor broadcast_to must be higher dimensional than tensor broadcast_from");

  bool feasible_bcast =
      tensor_is_broadcastable_to(broadcast_from, broadcast_to);

  ET_CHECK_MSG(
      feasible_bcast,
      "Cannot broadcast tensor broadcast_from into tensor broadcast_to along some dimensions");

  // Once we have discovered that broadcast_from can be broadcasted into
  // broadcast_to, use repeat() to do the broadcast.
  Tensor out = make_tensor(
      broadcast_to_shape,
      broadcast_to_dim_order,
      broadcast_to_strides,
      broadcast_from.scalar_type());

  // We need to pass IntArrayRef (i.e. ArrayRef<int64_t>) to cpu::repeat() but
  // .sizes() is ArrayRef<int32_t>
  using T = IntArrayRef::value_type;
  auto ndim = broadcast_to.dim();

  // repeat is int64_t* but broadcast_to_shape is at::ArrayRef<int32_t>
  T* repeats = static_cast<T*>(malloc((ndim) * sizeof(T)));
  for (int i = 0; i < ndim; ++i) {
    repeats[i] = broadcast_to_shape[i];
  }

  // Compute the repeat factor along each dimension
  for (int i = broadcast_to_shape.size() - 1,
           j = broadcast_from_shape.size() - 1;
       j >= 0;
       --i, --j) {
    if (broadcast_to_shape[i] == broadcast_from_shape[j]) {
      repeats[i] = 1;
    }
  }

  ET_CHECK(
      repeat_tensor(broadcast_from, makeArrayRef(repeats, ndim), out) ==
      Error::Ok);

  free(repeats);

  return out;
}

ET_NODISCARD Error get_broadcast_target_size(
    const exec_aten::ArrayRef<Tensor::SizesType> a_size,
    const exec_aten::ArrayRef<Tensor::SizesType> b_size,
    Tensor::SizesType* out_sizes,
    const size_t out_sizes_len,
    size_t* out_dim) {
  ET_CHECK_OR_RETURN_ERROR(
      tensors_are_broadcastable_between(a_size, b_size),
      InvalidArgument,
      "Two input tensors should be broadcastable.\n");

  auto a_dim = a_size.size();
  auto b_dim = b_size.size();

  ET_CHECK_OR_RETURN_ERROR(
      a_dim <= out_sizes_len && b_dim <= out_sizes_len,
      InvalidArgument,
      "Dim of input tensors should be smaller than the limitation, but find %zu, %zu and %zu.",
      a_dim,
      b_dim,
      out_sizes_len);

  *out_dim = a_dim > b_dim ? a_dim : b_dim;

  for (int a_idx = a_dim - 1,
           b_idx = b_dim - 1,
           expected_target_idx = *out_dim - 1;
       expected_target_idx >= 0;
       a_idx--, b_idx--, expected_target_idx--) {
    if (a_idx >= 0 && b_idx >= 0) {
      out_sizes[expected_target_idx] =
          b_size[b_idx] == 1 ? a_size[a_idx] : b_size[b_idx];
    } else {
      out_sizes[expected_target_idx] =
          a_idx >= 0 ? a_size[a_idx] : b_size[b_idx];
    }
  }

  return Error::Ok;
}

ET_NODISCARD Error get_broadcast_target_size(
    const Tensor& a,
    const Tensor& b,
    Tensor::SizesType* out_sizes,
    const size_t out_sizes_len,
    size_t* out_dim) {
  return get_broadcast_target_size(
      a.sizes(), b.sizes(), out_sizes, out_sizes_len, out_dim);
}

void delinearize_index(
    size_t linear_index,
    exec_aten::ArrayRef<Tensor::SizesType> shape,
    size_t* out_indexes,
    const size_t out_indexes_len) {
  ET_CHECK(shape.size() <= out_indexes_len);
  for (auto i = 0; i < shape.size(); ++i) {
    auto dim = shape.size() - 1 - i;
    auto dim_size = shape[dim];
    out_indexes[dim] = linear_index % dim_size;
    linear_index /= dim_size;
  }
}

void delinearize_index(
    size_t linear_index,
    const Tensor& t,
    size_t* out_indexes,
    const size_t out_indexes_len) {
  delinearize_index(linear_index, t.sizes(), out_indexes, out_indexes_len);
}

size_t linearize_access_indexes(
    ArrayRef<size_t> indexes_broadcast_to,
    ssize_t broadcast_to_ndim,
    exec_aten::ArrayRef<Tensor::SizesType> broadcast_from_shape,
    exec_aten::ArrayRef<Tensor::StridesType> broadcast_from_strides) {
  size_t num_skip_dims = broadcast_to_ndim - broadcast_from_shape.size();
  ArrayRef<size_t> indexes_broadcast_from = indexes_broadcast_to.slice(
      num_skip_dims, broadcast_to_ndim - num_skip_dims);

  ET_CHECK(indexes_broadcast_from.size() == broadcast_from_shape.size());

  size_t linear_index = 0;
  for (size_t i = 0; i < indexes_broadcast_from.size(); ++i) {
    // If this dimension is broadcasted, add zero to the linear address.
    if (indexes_broadcast_from[i] >= broadcast_from_shape[i]) {
      ET_CHECK_MSG(
          broadcast_from_shape[i] == 1,
          "Expected dim size == 1 if broadcasted, but actual dim size is %zu",
          static_cast<size_t>(broadcast_from_shape[i]));
      continue;
    }
    linear_index += indexes_broadcast_from[i] * broadcast_from_strides[i];
  }
  return linear_index;
}

size_t linearize_access_indexes(
    ArrayRef<size_t> indexes_broadcast_to,
    ssize_t broadcast_to_ndim,
    const Tensor& broadcast_from) {
  return linearize_access_indexes(
      indexes_broadcast_to,
      broadcast_to_ndim,
      broadcast_from.sizes(),
      broadcast_from.strides());
}

} // namespace executor
} // namespace torch
