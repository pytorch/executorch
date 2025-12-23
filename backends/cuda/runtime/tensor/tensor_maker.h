/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch::backends::cuda {

/**
 * A smart pointer type for managing the lifecycle of a Tensor.
 * This is compatible with executorch::extension::TensorPtr.
 */
using TensorPtr = std::shared_ptr<executorch::aten::Tensor>;

/**
 * Creates a TensorPtr for AOTI backends that skips stride calculation and
 * incontiguous tensor checks. This is specifically designed for AOTI-CUDA
 * which handles both contiguous and incontiguous tensors.
 *
 * This function is similar to executorch::extension::make_tensor_ptr but
 * bypasses the stride validation that assumes contiguous tensors, making it
 * suitable for AOTI backends that support arbitrary strides.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of the tensor.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter will be called when the managed Tensor
 * object is destroyed.
 * @return A TensorPtr that manages the newly created Tensor.
 */
TensorPtr make_tensor(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr);

} // namespace executorch::backends::cuda
