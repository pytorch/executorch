/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <ATen/Tensor.h> // @manual
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {
/**
 * Implementation for ATen tensor util, should only be included in
 * `<target>_aten` target and only be used in ATen mode. Explicitly taking
 * at::Tensor (instead of exec_aten::Tensor) to make sure it fails at compile
 * time if built incorrectly.
 */
Error get_dim_order(
    const at::Tensor& tensor,
    exec_aten::DimOrderType* out_dim_order,
    size_t out_dim_order_size) {
  ET_CHECK_OR_RETURN_ERROR(
      out_dim_order_size == tensor.dim(),
      InvalidArgument,
      "out_dim_order_size needs to be equal to the number of dimensions of the tensor. out_dim_order_size %zu, tensor.dim() %" PRId64,
      out_dim_order_size,
      tensor.dim());
  return stride_to_dim_order(
      tensor.strides().data(), tensor.dim(), out_dim_order);
}

bool tensor_has_valid_dim_order(at::Tensor t) {
  exec_aten::DimOrderType dim_order[kTensorDimensionLimit];
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      get_dim_order(t, dim_order, t.dim()) == Error::Ok,
      "Failed to retrieve dim order from tensor!");

  if (!validate_dim_order(dim_order, t.dim())) {
    ET_LOG(Error, "Tensor dim order is not valid:");
    for (size_t d = 0; d < t.dim(); ++d) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          static_cast<size_t>(d),
          static_cast<size_t>(dim_order[d]));
    }
    return false;
  }
  return true;
}

inline bool tensor_is_default_or_channels_last_dim_order(at::Tensor t) {
  exec_aten::DimOrderType dim_order[kTensorDimensionLimit];
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      get_dim_order(t, dim_order, t.dim()) == Error::Ok,
      "Failed to retrieve dim order from tensor!");

  bool ret_val = is_contiguous_dim_order(dim_order, t.dim()) ||
      is_channels_last_dim_order(dim_order, t.dim());

  if (!ret_val) {
    ET_LOG(
        Error,
        "Expected tensor to have default or channels last dim order, but got");
    for (size_t d = 0; d < t.dim(); ++d) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          static_cast<size_t>(d),
          static_cast<size_t>(dim_order[d]));
    }
  }
  return ret_val;
}

namespace internal {

Error share_tensor_data(const at::Tensor& t_dst, const at::Tensor& t_src) {
  at::StorageImpl* storage =
      t_dst.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();

  ET_CHECK_OR_RETURN_ERROR(
      t_dst.nbytes() == t_src.nbytes(),
      InvalidArgument,
      "t_dst.nbytes() %lu != t_src.nbytes(). %lu",
      t_dst.nbytes(),
      t_src.nbytes());

  ET_CHECK_OR_RETURN_ERROR(
      t_src.mutable_data_ptr() != nullptr,
      InvalidArgument,
      "Source tensor should have data_ptr not being nullptr.");
  // Assign the dataptr as the input tensor dataptr
  storage->set_data_ptr(
      at::DataPtr(t_src.mutable_data_ptr(), at::DeviceType::CPU));
  storage->set_nbytes(t_src.nbytes());

  return Error::Ok;
}

Error copy_tensor_data(const at::Tensor& t_dst, const at::Tensor& t_src) {
  void* dst_data_ptr = t_dst.unsafeGetTensorImpl()
                           ->unsafe_storage()
                           .unsafeGetStorageImpl()
                           ->data_ptr()
                           .get();

  // Currently even 0 sized tensors receive a dataptr in pre_allocated
  // memory planning so we can do this check.
  // TODO(jakeszwe, shunting, gasoonjia): this should be clear in design if
  // other people make their own memory plans
  ET_CHECK_OR_RETURN_ERROR(
      dst_data_ptr != nullptr,
      InvalidArgument,
      "Destination tensor data pointer must not be null.");

  // Sources with a size 0 dimension can be nullptr
  if (t_src.const_data_ptr() != nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        t_dst.nbytes() == t_src.nbytes(),
        InvalidArgument,
        "t_dst.nbytes() %lu != t_src.nbytes(). %lu",
        t_dst.nbytes(),
        t_src.nbytes());
    // Copy the source data to the preallocated memory of the destination, which
    // must be the same size as the source.
    std::memcpy(dst_data_ptr, t_src.const_data_ptr(), t_src.nbytes());
  }

  return Error::Ok;
}

__ET_NODISCARD Error
set_tensor_data(const at::Tensor& t, void* buffer, size_t buffer_size) {
  ET_CHECK_OR_RETURN_ERROR(
      buffer_size >= t.nbytes(),
      InvalidArgument,
      "buffer_size %zu is smaller than smaller than tensor nbytes %zu",
      buffer_size,
      t.nbytes());
  t.unsafeGetTensorImpl()->unsafe_storage().set_data_ptr(
      at::DataPtr(buffer, at::DeviceType::CPU));
  return Error::Ok;
}

void reset_data_ptr(const at::Tensor& tensor) {
  auto impl = tensor.unsafeGetTensorImpl();
  impl->set_sizes_contiguous(0);
  impl->unsafe_storage().unsafeGetStorageImpl()->reset();
}

/// Most callers should use resize_tensor() instead.
Error resize_tensor_impl(
    c10::TensorImpl* impl,
    c10::ArrayRef<exec_aten::SizesType> new_sizes) {
  // The lean-mode Tensor will perform this check, but at::Tensor won't.
  // Although at::Tensor can be resized in this case, it's not allowed by the
  // higher-level constraints of the runtime.
  if (impl->dim() != new_sizes.size()) {
    ET_LOG(
        Error,
        "Tensor rank is not mutable: old dim: %" PRId64 " new dim: %zu",
        impl->dim(),
        new_sizes.size());
    return torch::executor::Error::NotSupported;
  }
  // Will panic on failure.
  impl->set_sizes_contiguous(new_sizes);
  return torch::executor::Error::Ok;
}

} // namespace internal

} // namespace runtime
} // namespace executorch
