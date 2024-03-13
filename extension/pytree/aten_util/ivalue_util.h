/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <memory>
#include <utility>

// patternlint-disable executorch-cpp-nostdinc
#include <vector>

#include <executorch/extension/pytree/pytree.h>

namespace torch {
namespace executor {
namespace util {

using Empty = torch::executor::pytree::Empty;

std::pair<
    std::vector<at::Tensor>,
    std::unique_ptr<torch::executor::pytree::TreeSpec<Empty>>>
flatten(const c10::IValue& data);

c10::IValue unflatten(
    const std::vector<at::Tensor>& tensors,
    const std::unique_ptr<torch::executor::pytree::TreeSpec<Empty>>& tree_spec);

bool is_same(
    const std::vector<at::Tensor>& a,
    const std::vector<at::Tensor>& b);

bool is_same(const c10::IValue& lhs, const c10::IValue& rhs);

} // namespace util
} // namespace executor
} // namespace torch
