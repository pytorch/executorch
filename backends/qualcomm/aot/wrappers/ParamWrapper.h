/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>

#include <string>
#include <utility>

#include "QnnTypes.h"
namespace torch {
namespace executor {
namespace qnn {
class ParamWrapper {
 public:
  // Populate Qnn_Param_t. Return an error code Error::Ok if succeeded,
  // Error::Internal if failed
  virtual Error PopulateQnnParam() = 0;
  virtual ~ParamWrapper() = default;

  ParamWrapper(const ParamWrapper& rhs) = default;
  ParamWrapper(ParamWrapper&& rhs) = default;
  ParamWrapper& operator=(const ParamWrapper& rhs) = default;
  ParamWrapper& operator=(ParamWrapper&& rhs) = default;

  // Return value to internal Qnn param struct that will be
  // populated with the parameter information
  Qnn_Param_t GetQnnParam() {
    return qnn_param_;
  }

  const std::string& GetName() const {
    return name_;
  }

 protected:
  explicit ParamWrapper(Qnn_ParamType_t type, std::string name)
      : name_(std::move(name)) {
    qnn_param_.paramType = type;
    qnn_param_.name = name_.c_str();
  }
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;

 private:
  std::string name_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
