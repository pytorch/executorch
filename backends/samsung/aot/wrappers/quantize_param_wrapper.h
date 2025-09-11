/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

#include <include/common-types.h>
namespace torch {
namespace executor {
namespace enn {

class EnnQuantizeParamWrapper {
 public:
  EnnQuantizeParamWrapper(
      std::string quantize_dtype,
      const std::vector<float>& scales,
      const std::vector<int32_t>& zero_points)
      : quantize_type_(std::move(quantize_dtype)),
        scales_(scales),
        zero_points_(zero_points) {}

  const std::string& GetQuantizeType() const {
    return quantize_type_;
  }

  ParamWrapper GetScales() const {
    ParamWrapper param;
    param.data = const_cast<float*>(scales_.data());
    param.size = scales_.size();
    param.is_scalar = false;
    param.type = ScalarType::FLOAT32;

    return param;
  }

  ParamWrapper GetZeroPoints() const {
    ParamWrapper param;
    param.data = const_cast<int32_t*>(zero_points_.data());
    param.size = zero_points_.size();
    param.is_scalar = false;
    param.type = ScalarType::INT32;

    return param;
  }

 private:
  std::string quantize_type_;
  std::vector<float> scales_;
  std::vector<int32_t> zero_points_;
};

} // namespace enn
} // namespace executor
} // namespace torch
