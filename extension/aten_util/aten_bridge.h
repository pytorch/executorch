/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <ATen/Functions.h> // @manual=//caffe2/aten:ATen-cpu
#include <ATen/Tensor.h> // @manual=//caffe2/aten:ATen-core
#include <ATen/core/functional.h> // @manual=//caffe2/aten:ATen-core
#include <c10/core/ScalarTypeToTypeMeta.h> // @manual=//caffe2/c10:c10

#include <memory>
#include <vector>

namespace executorch {
namespace extension {

torch::executor::ScalarType torch_to_executorch_scalar_type(
    caffe2::TypeMeta type);

c10::ScalarType executorch_to_torch_scalar_type(
    torch::executor::ScalarType type);

/*
 * @param[in] aten_tensor Input at::Tensor
 * @param[in,out] mutable_et ETensor whose underlying memory now will alias to
 * aten_tensor
 */
void alias_etensor_to_attensor(at::Tensor& at, torch::executor::Tensor& et);

/*
 * @param[in] et ETensor whose underlying memory now will alias to returned
 * output tensor
 * @param[ret] aten_tensor output at::Tensor
 * Notes:
 * It is owned by the caller of alias_attensor_to_etensor.
 * Lifetime of tensor meta must be >= to that of the returned tensor since
 * this function uses at::from_blob API that constructs non-owning tensor
 * along with non-owning metadata, e.g. sizes.
 * If such lifetime guarantees cannot be provided, returned tensor should be
 * cloned.
 */
at::Tensor alias_attensor_to_etensor(const torch::executor::Tensor& et);

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::alias_attensor_to_etensor;
using ::executorch::extension::alias_etensor_to_attensor;
inline torch::executor::ScalarType torchToExecuTorchScalarType(
    caffe2::TypeMeta type) {
  return ::executorch::extension::torch_to_executorch_scalar_type(type);
}
inline c10::ScalarType execuTorchtoTorchScalarType(
    torch::executor::ScalarType type) {
  return ::executorch::extension::executorch_to_torch_scalar_type(type);
}
} // namespace util
} // namespace executor
} // namespace torch

// Some users refer to these as `torch::util::`.
namespace torch {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::torch::executor::util::alias_attensor_to_etensor;
using ::torch::executor::util::alias_etensor_to_attensor;
using ::torch::executor::util::execuTorchtoTorchScalarType;
using ::torch::executor::util::torchToExecuTorchScalarType;
} // namespace util
} // namespace torch
