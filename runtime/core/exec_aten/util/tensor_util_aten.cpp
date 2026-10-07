/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <ATen/Tensor.h> // @manual
#include <c10/core/MemoryFormat.h>
#include <c10/util/irange.h>
#include <c10/util/strides.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace {

Error get_dim_order_from_sizes_and_strides(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    executorch::aten::DimOrderType* out_dim_order) {
  if (strides == c10::contiguous_strides(sizes)) {
    for (const auto i : c10::irange(sizes.size())) {
      out_dim_order[i] = static_cast<executorch::aten::DimOrderType>(i);
    }
    return Error::Ok;
  }
  if (sizes.size() == 4 &&
      strides == c10::get_channels_last_strides_2d(sizes)) {
    constexpr executorch::aten::DimOrderType channels_last[] = {0, 2, 3, 1};
    std::copy(
        std::begin(channels_last), std::end(channels_last), out_dim_order);
    return Error::Ok;
  }
  if (sizes.size() == 5 &&
      strides == c10::get_channels_last_strides_3d(sizes)) {
    constexpr executorch::aten::DimOrderType channels_last_3d[] = {
        0, 2, 3, 4, 1};
    std::copy(
        std::begin(channels_last_3d),
        std::end(channels_last_3d),
        out_dim_order);
    return Error::Ok;
  }
  return stride_to_dim_order(strides.data(), strides.size(), out_dim_order);
}

} // namespace

/**
 * Implementation for ATen tensor util, should only be included in
 * `<target>_aten` target and only be used in ATen mode. Explicitly taking
 * at::Tensor (instead of executorch::aten::Tensor) to make sure it fails at
 * compile time if built incorrectly.
 */
Error get_dim_order(
    const at::Tensor& tensor,
    executorch::aten::DimOrderType* out_dim_order,
    size_t out_dim_order_size) {
  ET_CHECK_OR_RETURN_ERROR(
      out_dim_order_size == tensor.dim(),
      InvalidArgument,
      "out_dim_order_size needs to be equal to the number of dimensions of the tensor. out_dim_order_size %zu, tensor.dim() %" PRId64,
      out_dim_order_size,
      tensor.dim());
  return get_dim_order_from_sizes_and_strides(
      tensor.sizes(), tensor.strides(), out_dim_order);
}

bool tensor_has_valid_dim_order(at::Tensor t) {
  executorch::aten::DimOrderType dim_order[kTensorDimensionLimit];
  ET_CHECK_OR_RETURN_FALSE(
      get_dim_order(t, dim_order, t.dim()) == Error::Ok,
      "Failed to retrieve dim order from tensor!");

  if (!validate_dim_order(dim_order, t.dim())) {
    ET_LOG(Error, "Tensor dim order is not valid:");
    for (const auto d : c10::irange(t.dim())) {
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
  executorch::aten::DimOrderType dim_order[kTensorDimensionLimit];
  ET_CHECK_OR_RETURN_FALSE(
      get_dim_order(t, dim_order, t.dim()) == Error::Ok,
      "Failed to retrieve dim order from tensor!");

  bool ret_val = is_contiguous_dim_order(dim_order, t.dim()) ||
      is_channels_last_dim_order(dim_order, t.dim());

  if (!ret_val) {
    ET_LOG(
        Error,
        "Expected tensor to have default or channels last dim order, but got");
    for (const auto d : c10::irange(t.dim())) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          static_cast<size_t>(d),
          static_cast<size_t>(dim_order[d]));
    }
  }
  return ret_val;
}

bool tensors_have_same_dim_order(
    const executorch::aten::ArrayRef<executorch::aten::Tensor> tensor_list) {
  if (tensor_list.size() < 2) {
    return true;
  }

  executorch::aten::DimOrderType first_dim_order[kTensorDimensionLimit];
  executorch::aten::DimOrderType other_dim_order[kTensorDimensionLimit];

  ET_CHECK_OR_RETURN_FALSE(
      get_dim_order(tensor_list[0], first_dim_order, tensor_list[0].dim()) ==
          Error::Ok,
      "Failed to retrieve dim order from 1st input tensor!");

  bool all_contiguous =
      is_contiguous_dim_order(first_dim_order, tensor_list[0].dim());
  bool all_channels_last =
      is_channels_last_dim_order(first_dim_order, tensor_list[0].dim());

  for (const auto i : c10::irange(1, tensor_list.size())) {
    ET_CHECK_OR_RETURN_FALSE(
        get_dim_order(tensor_list[i], other_dim_order, tensor_list[i].dim()) ==
            Error::Ok,
        "Failed to retrieve dim order from %zd-th input tensor!",
        i);

    all_contiguous = all_contiguous &&
        is_contiguous_dim_order(other_dim_order, tensor_list[i].dim());
    all_channels_last = all_channels_last &&
        is_channels_last_dim_order(other_dim_order, tensor_list[i].dim());
  }

  ET_CHECK_OR_RETURN_FALSE(
      all_contiguous || all_channels_last,
      "%zd input tensors have different dim orders",
      tensor_list.size());

  return all_contiguous || all_channels_last;
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
  // The destination keeps its own TensorImpl but adopts the source storage, so
  // the devices must agree or TensorImpl and DataPtr would disagree.
  ET_CHECK_OR_RETURN_ERROR(
      t_dst.device() == t_src.device(),
      InvalidArgument,
      "Destination device %s does not match source device %s",
      c10::str(t_dst.device()).c_str(),
      c10::str(t_src.device()).c_str());
  // Preserve the source device; hardcoding CPU would mis-tag a device input's
  // storage as host and the backend would later reject it.
  storage->set_data_ptr(at::DataPtr(t_src.mutable_data_ptr(), t_src.device()));
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
    //
    // Both sides have to be host memory. Reaching here with device memory means
    // a program planned a buffer for a tensor that lives on an accelerator, and
    // a host memcpy into it is undefined. Reported because the alternative is a
    // crash with no message.
    ET_CHECK_OR_RETURN_ERROR(
        t_dst.device().is_cpu() && t_src.device().is_cpu(),
        NotSupported,
        // Kept under the 256-char log buffer (runtime/platform/log.cpp) so the
        // MemoryPlanningPass hint, which is the actionable part, is not
        // truncated away.
        "Planned-buffer copy needs host memory on both sides: dst device %s, src device %s. "
        "Export with MemoryPlanningPass(alloc_graph_input=False) to share the caller's memory.",
        c10::DeviceTypeName(t_dst.device().type()).c_str(),
        c10::DeviceTypeName(t_src.device().type()).c_str());
    std::memcpy(dst_data_ptr, t_src.const_data_ptr(), t_src.nbytes());
  }

  return Error::Ok;
}

ET_NODISCARD Error
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
    c10::ArrayRef<executorch::aten::SizesType> new_sizes) {
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
  if (impl->sizes() == new_sizes) {
    return torch::executor::Error::Ok;
  }

  std::array<executorch::aten::DimOrderType, kTensorDimensionLimit> dim_order;
  Error error = get_dim_order_from_sizes_and_strides(
      impl->sizes(), impl->strides(), dim_order.data());
  if (error != Error::Ok) {
    return error;
  }
  std::array<executorch::aten::StridesType, kTensorDimensionLimit> new_strides;
  error = dim_order_to_stride(
      new_sizes.data(), dim_order.data(), new_sizes.size(), new_strides.data());
  if (error != Error::Ok) {
    return error;
  }
  impl->set_sizes_and_strides(
      new_sizes, {new_strides.data(), new_sizes.size()});
  return torch::executor::Error::Ok;
}

} // namespace internal
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
