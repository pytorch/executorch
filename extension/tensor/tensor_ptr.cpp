/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch {
namespace extension {

TensorPtr clone_tensor_ptr(const exec_aten::Tensor& tensor) {
  std::vector<exec_aten::SizesType> sizes(
      tensor.sizes().begin(), tensor.sizes().end());
  std::vector<exec_aten::DimOrderType> dim_order{
#ifndef USE_ATEN_LIB
      tensor.dim_order().begin(), tensor.dim_order().end()
#endif // USE_ATEN_LIB
  };
  std::vector<exec_aten::StridesType> strides(
      tensor.strides().begin(), tensor.strides().end());
  auto dynamism = exec_aten::TensorShapeDynamism::DYNAMIC_BOUND;
#ifndef USE_ATEN_LIB
  dynamism = tensor.shape_dynamism();
#endif // USE_ATEN_LIB
  return tensor.const_data_ptr()
      ? make_tensor_ptr(
            std::move(sizes),
            std::vector<uint8_t>(
                (uint8_t*)tensor.const_data_ptr(),
                (uint8_t*)tensor.const_data_ptr() + tensor.nbytes()),
            std::move(dim_order),
            std::move(strides),
            tensor.scalar_type(),
            dynamism)
      : make_tensor_ptr(
            std::move(sizes),
            nullptr,
            std::move(dim_order),
            std::move(strides),
            tensor.scalar_type(),
            dynamism);
}

runtime::Error resize_tensor_ptr(
    TensorPtr& tensor,
    const std::vector<exec_aten::SizesType>& sizes) {
  return runtime::resize_tensor(
      *tensor,
      exec_aten::ArrayRef<exec_aten::SizesType>(sizes.data(), sizes.size()));
}

} // namespace extension
} // namespace executorch
