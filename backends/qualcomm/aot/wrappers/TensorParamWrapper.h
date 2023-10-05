/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/aot/wrappers/ParamWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/runtime/core/error.h>

#include <memory>

namespace torch {
namespace executor {
namespace qnn {
class TensorParamWrapper final : public ParamWrapper {
 public:
  explicit TensorParamWrapper(
      std::string name,
      std::shared_ptr<TensorWrapper> static_tensor)
      : ParamWrapper(QNN_PARAMTYPE_TENSOR, std::move(name)),
        static_tensor_wrapper_(std::move(static_tensor)) {}
  // Populate Qnn tensorParam with tensor wrapper
  Error PopulateQnnParam() override {
    // Error out if underlying tensor is not static:
    if (!static_tensor_wrapper_->IsTensorStatic())
      return Error::Internal;
    qnn_param_.tensorParam = static_tensor_wrapper_->CloneTensorStruct();
    return Error::Ok;
  }

  // Accessor functions:
  const void* GetData() const {
    return static_tensor_wrapper_->GetStaticTensorData();
  }

  std::shared_ptr<TensorWrapper> GetTensorWrapper() {
    return static_tensor_wrapper_;
  }

 private:
  std::shared_ptr<TensorWrapper> static_tensor_wrapper_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
