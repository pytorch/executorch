/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/kernels/portable/cpu/util/allocate_tensor_util.h"


namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor allocate_tensor(
    KernelRuntimeContext& ctx,
    const ArrayRef<Tensor::SizesType>& sizes,
    const ArrayRef<Tensor::DimOrderType>& dim_order,
    const ArrayRef<Tensor::StridesType>& strides,
    const ScalarType& dtype) {
  int dim = sizes.size();
  int size_nbytes = dim * sizeof(Tensor::SizesType);
  Result<void*> temp_mem_res_size = ctx.allocate_temp(size_nbytes);
  void* size_data_ptr =
      temp_mem_res_size.ok() ? temp_mem_res_size.get() : nullptr;
  ET_CHECK_MSG(size_data_ptr != nullptr, "Failed to malloc for size bytes");
  memcpy(size_data_ptr, sizes.data(), size_nbytes);

  // TODO(T145322324): can we remove the static cast once size is unsigned?
  size_t dim_order_nbytes =
      static_cast<size_t>(dim) * sizeof(Tensor::DimOrderType);
  Result<void*> temp_mem_res_dim_order = ctx.allocate_temp(dim_order_nbytes);
  void* dim_order_data_ptr =
      temp_mem_res_dim_order.ok() ? temp_mem_res_dim_order.get() : nullptr;
  ET_CHECK_MSG(
      dim_order_data_ptr != nullptr, "Failed to malloc for dim order bytes");
  memcpy(dim_order_data_ptr, dim_order.data(), dim_order_nbytes);

  int strides_nbytes = dim * sizeof(Tensor::StridesType);
  Result<void*> temp_mem_res_strides = ctx.allocate_temp(strides_nbytes);
  void* strides_data_ptr =
      temp_mem_res_strides.ok() ? temp_mem_res_strides.get() : nullptr;
  printf("strides_data_ptr: %p\n", strides_data_ptr);
  fflush(stdout);
  ET_CHECK_MSG(
      strides_data_ptr != nullptr, "Failed to malloc for strides bytes");
  memcpy(strides_data_ptr, strides.data(), strides_nbytes);

  Result<void*> temp_mem_res_tensor = ctx.allocate_temp(sizeof(TensorImpl));
  auto tensor_impl = static_cast<TensorImpl*>(
      temp_mem_res_tensor.ok() ? temp_mem_res_tensor.get() : nullptr);
  ET_CHECK_MSG(tensor_impl != nullptr, "Failed to malloc for data TensorImpl");

  new (tensor_impl) TensorImpl(
      dtype,
      dim,
      reinterpret_cast<Tensor::SizesType*>(size_data_ptr),
      nullptr,
      reinterpret_cast<Tensor::DimOrderType*>(dim_order_data_ptr),
      reinterpret_cast<Tensor::StridesType*>(strides_data_ptr));

  Result<void*> temp_mem_res_data = ctx.allocate_temp(tensor_impl->nbytes());
  void* data_ptr = temp_mem_res_data.ok() ? temp_mem_res_data.get() : nullptr;
  ET_CHECK_MSG(data_ptr != nullptr, "Failed to malloc for data buffer");
  tensor_impl->set_data(data_ptr);

  return Tensor{tensor_impl};
}

} // namespace executor
} // namespace torch
