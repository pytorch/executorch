// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/TensorInfo.h>

namespace ptn {

class MethodMeta {
 public:
  MethodMeta() = default;
  MethodMeta(
      std::string name,
      std::vector<TensorInfo> inputs,
      std::vector<TensorInfo> outputs)
      : name_(std::move(name)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)) {}

  static MethodMeta from_method(const Method& method);

  const std::string& name() const {
    return name_;
  }

  const std::vector<TensorInfo>& inputs() const {
    return inputs_;
  }

  const std::vector<TensorInfo>& outputs() const {
    return outputs_;
  }

 private:
  std::string name_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;
};

} // namespace ptn
