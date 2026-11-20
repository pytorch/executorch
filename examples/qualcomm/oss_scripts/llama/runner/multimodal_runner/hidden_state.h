/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>
#include <vector>

namespace example {

/**
 * Bind a decoder's final hidden-state output (output index 1, right after the
 * logits; quantized to uint16) for models that also return it.
 */
inline void bind_hidden_output(
    IMemAlloc* buffer_manager,
    const executorch::runtime::MethodMeta& method_meta,
    TensorStructRaw& hidden,
    std::vector<executorch::aten::Tensor>& output_tensors) {
  auto info = method_meta.output_tensor_meta(1);
  ET_CHECK_MSG(
      info->scalar_type() == executorch::aten::ScalarType::UInt16 &&
          info->nbytes() == hidden.size,
      "Unexpected hidden-state output");
  hidden.dtype = info->scalar_type();
  hidden.data = buffer_manager->allocate(hidden.size);
  hidden.tensor = std::make_unique<executorch::aten::TensorImpl>(
      info->scalar_type(),
      info->sizes().size(),
      const_cast<executorch::aten::TensorImpl::SizesType*>(
          info->sizes().data()),
      hidden.data,
      const_cast<executorch::aten::TensorImpl::DimOrderType*>(
          info->dim_order().data()));
  output_tensors.emplace_back(hidden.tensor.get());
  buffer_manager->add_memory_info(hidden.data, hidden.size, info.get());
}

} // namespace example
