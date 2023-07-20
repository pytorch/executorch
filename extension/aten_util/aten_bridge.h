#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <ATen/Functions.h> // @manual=//caffe2/aten:ATen-cpu
#include <ATen/Tensor.h> // @manual=//caffe2/aten:ATen-core
#include <ATen/core/functional.h> // @manual=//caffe2/aten:ATen-core
#include <c10/core/ScalarTypeToTypeMeta.h> // @manual=//caffe2/c10:c10

#include <memory>
#include <vector>

namespace torch {
namespace util {

using sizes32_t = std::vector<int32_t>;
using sizes64_t = std::vector<int64_t>;

struct KeepAliveSizes {
  std::vector<sizes32_t> sizes32;
  std::vector<sizes64_t> sizes64;
};

// TODO: we should really remove this as
__ET_DEPRECATED std::unique_ptr<torch::executor::TensorImpl>
eTensorFromAtTensor(const at::Tensor& tensor, KeepAliveSizes& keep_alive);

__ET_DEPRECATED at::Tensor atTensorFromETensor(
    torch::executor::TensorImpl* etensor,
    KeepAliveSizes& keep_alive);

torch::executor::ScalarType torchToExecuTorchScalarType(caffe2::TypeMeta type);

c10::ScalarType execuTorchtoTorchScalarType(torch::executor::ScalarType type);

/*
 * @param[in] aten_tensor: Input at::Tensor
 * @param[in/out] mutable_et: ETensor whose underlying memory now will alias to
 * aten_tensor
 */
void alias_etensor_to_attensor(at::Tensor& at, torch::executor::Tensor& et);

/*
 * @param[in] et: ETensor whose underlying memory now will alias to returned
 * output tensor
 * @param[ret] aten_tensor: output at::Tensor
 * Notes:
 * It is owned by the caller of alias_attensor_to_etensor.
 * Lifetime of tensor meta must be >= to that of the returned tensor since
 * this function uses at::from_blob API that constructs non-owning tensor
 * along with non-owning metadata, e.g. sizes.
 * If such lifetime guarantees cannot be provided, returned tensor should be
 * cloned.
 */
at::Tensor alias_attensor_to_etensor(const torch::executor::Tensor& et);
} // namespace util
} // namespace torch
