/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <c10/util/irange.h>
#include <cstring>

#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {
/**
 * Implementation for ExecuTorch tensor util, should only be included in
 * an target with ATen mode turned off. Explicitly taking
 * torch::executor::Tensor (instead of executorch::aten::Tensor) to make sure it
 * fails at compile time if built incorrectly.
 */
Error get_dim_order(
    const torch::executor::Tensor& tensor,
    executorch::aten::DimOrderType* out_dim_order,
    size_t out_dim_order_size) {
  ET_CHECK_OR_RETURN_ERROR(
      out_dim_order_size == tensor.dim_order().size(),
      InvalidArgument,
      "Size needs to be equal to the number of dimensions of the tensor size %zu, tensor.dim() %zu",
      out_dim_order_size,
      tensor.dim_order().size());
  std::memcpy(
      out_dim_order,
      tensor.dim_order().data(),
      tensor.dim_order().size() * sizeof(executorch::aten::DimOrderType));
  return Error::Ok;
}

bool tensor_has_valid_dim_order(torch::executor::Tensor t) {
  if (!validate_dim_order(t.dim_order().data(), t.dim_order().size())) {
    ET_LOG(Error, "Tensor dim order is not valid:");
    for ([[maybe_unused]] const auto d : c10::irange(t.dim())) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          d,
          static_cast<size_t>(t.dim_order()[d]));
    }
    return false;
  }
  return true;
}

bool tensor_is_default_or_channels_last_dim_order(torch::executor::Tensor t) {
  bool ret_val =
      is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size()) ||
      is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size());

  if (!ret_val) {
    ET_LOG(
        Error,
        "Expected tensor to have default or channels last dim order, but got");
    for ([[maybe_unused]] const auto d : c10::irange(t.dim())) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          d,
          static_cast<size_t>(t.dim_order()[d]));
    }
  }
  return ret_val;
}

bool tensor_is_default_dim_order(torch::executor::Tensor t) {
  bool ret_val =
      is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size());

  if (!ret_val) {
    ET_LOG(Error, "Expected tensor to have default dim order, but got");
    for ([[maybe_unused]] const auto d : c10::irange(t.dim())) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          d,
          static_cast<size_t>(t.dim_order()[d]));
    }
  }
  return ret_val;
}

bool tensor_is_channels_last_dim_order(torch::executor::Tensor t) {
  bool ret_val =
      is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size());

  if (!ret_val) {
    ET_LOG(Error, "Expected tensor to have channels last dim order, but got");
    for ([[maybe_unused]] const auto d : c10::irange(t.dim())) {
      ET_LOG(
          Error,
          "    dim_order(%zu): %zu",
          d,
          static_cast<size_t>(t.dim_order()[d]));
    }
  }
  return ret_val;
}

bool tensors_have_same_dim_order(
    const executorch::aten::ArrayRef<executorch::aten::Tensor> tensor_list) {
  if (tensor_list.size() < 2) {
    return true;
  }
  bool all_contiguous = true;
  bool all_channels_last = true;
  for (const auto i : c10::irange(tensor_list.size())) {
    all_contiguous = all_contiguous &&
        is_contiguous_dim_order(
                         tensor_list[i].dim_order().data(),
                         tensor_list[i].dim_order().size());
    all_channels_last = all_channels_last &&
        is_channels_last_dim_order(
                            tensor_list[i].dim_order().data(),
                            tensor_list[i].dim_order().size());
  }

  ET_CHECK_OR_RETURN_FALSE(
      all_contiguous || all_channels_last,
      "%zd input tensors have different dim orders",
      tensor_list.size());

  return true;
}

namespace internal {

Error share_tensor_data(
    const torch::executor::Tensor& t_dst,
    const torch::executor::Tensor& t_src) {
  ET_CHECK_OR_RETURN_ERROR(
      t_dst.nbytes() == t_src.nbytes(),
      InvalidArgument,
      "t_dst.nbytes() %zu != t_src.nbytes(). %zu",
      t_dst.nbytes(),
      t_src.nbytes());

  // Either the t_src is empty or contains valid data.
  ET_CHECK_OR_RETURN_ERROR(
      t_src.mutable_data_ptr() != nullptr || t_src.nbytes() == 0,
      InvalidArgument,
      "Source tensor should have data_ptr not being nullptr.");

  // Setting data_ptr to nullptr explicitly when t_src is empty.
  void* t_src_data_ptr =
      t_src.numel() == 0 ? nullptr : t_src.mutable_data_ptr();
  // Assign internal data_ptr as the one in forwarded tensor
  t_dst.unsafeGetTensorImpl()->set_data(t_src_data_ptr);

  return Error::Ok;
}

Error copy_tensor_data(
    const torch::executor::Tensor& t_dst,
    const torch::executor::Tensor& t_src) {
  ET_CHECK_OR_RETURN_ERROR(
      t_dst.const_data_ptr() != nullptr ||
          (t_dst.nbytes() == 0 && t_src.nbytes() == 0),
      InvalidArgument,
      "ExecutionPlan input supposed to preallocated but has nullptr for data");
  // inputs with a size 0 dimension can be nullptr
  if (t_src.const_data_ptr() != nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        t_dst.nbytes() == t_src.nbytes(),
        InvalidArgument,
        "t_dst.nbytes() %zu != t_src.nbytes(). %zu",
        t_dst.nbytes(),
        t_src.nbytes());
    std::memcpy(
        t_dst.mutable_data_ptr(), t_src.const_data_ptr(), t_src.nbytes());
  }
  return Error::Ok;
}

ET_NODISCARD Error set_tensor_data(
    const torch::executor::Tensor& t,
    void* buffer,
    size_t buffer_size) {
  ET_CHECK_OR_RETURN_ERROR(
      buffer_size >= t.nbytes(),
      InvalidArgument,
      "buffer_size %zu is smaller than smaller than tensor nbytes %zu",
      buffer_size,
      t.nbytes());
  t.unsafeGetTensorImpl()->set_data(buffer);
  return Error::Ok;
}

void reset_data_ptr(const torch::executor::Tensor& tensor) {
  // Lean mode doesn't deallocate the tensor data_ptr in the allocator
  tensor.unsafeGetTensorImpl()->set_data(nullptr);
}

class TensorResizerFriend final {
 public:
  ET_NODISCARD static Error resize_tensor_impl(
      executorch::aten::TensorImpl* impl,
      executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes) {
    return impl->internal_resize_contiguous(new_sizes);
  }
};

Error resize_tensor_impl(
    torch::executor::TensorImpl* impl,
    torch::executor::ArrayRef<executorch::aten::SizesType> new_sizes) {
  return TensorResizerFriend::resize_tensor_impl(impl, new_sizes);
}
} // namespace internal

} // namespace runtime
} // namespace executorch
