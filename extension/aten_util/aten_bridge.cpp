/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/aten_bridge.h>

#include <executorch/runtime/platform/assert.h>
#include <cstring>

namespace torch {
namespace util {

namespace {
void check_tensor_meta(const at::Tensor& a, const exec_aten::Tensor& b) {
  // Check sizes/strides pointers
  ET_CHECK_MSG(
      b.sizes().data() != nullptr, "ETensor must have valid sizes array");
  ET_CHECK_MSG(
      b.strides().data() != nullptr, "ETensor must have valid strides array");
  // Check disabled because in ASR model we get 1 element tensor with different
  // rank.
  /*
ET_CHECK_MSG(
  a.dim() == b.dim(),
  "at::Tensor and ETensor must have same rank."
  " at::Tensor rank %zd, ETensor rank %zd.",
  a.dim(),
  b.dim());
  */
  // check sizes
  for (size_t i = 0, dims = a.dim(); i < dims; ++i) {
    ET_CHECK_MSG(
        a.size(i) == b.size(i),
        "Sizes dont match at index:%zd, a size %zd != b size %zd",
        i,
        ssize_t(a.size(i)),
        ssize_t(b.size(i)));
  }
  // check strides
  for (size_t i = 0, dims = a.dim(); i < dims; ++i) {
    // Dont match strides if the size is 1.
    // Why? Because tensor is non-contig only if
    // strides dont match product(sizes[i:]) when size(i) > 1
    // Strong assumption that must be tested and validated.
    ET_CHECK_MSG(
        (a.size(i) == 1 || (a.stride(i) == b.strides()[i])),
        "Strides dont match at index:%zd, a stride %zd != b stride %zd",
        i,
        ssize_t(a.stride(i)),
        ssize_t(b.strides()[i]));
  }
  // check dtype
  ET_CHECK_MSG(
      b.scalar_type() == torchToExecuTorchScalarType(a.options().dtype()),
      "dtypes dont match a %hhd vs. b %hhd",
      torchToExecuTorchScalarType(a.options().dtype()),
      b.scalar_type());
}
} // namespace

torch::executor::ScalarType torchToExecuTorchScalarType(caffe2::TypeMeta type) {
  switch (c10::typeMetaToScalarType(type)) {
    case c10::ScalarType::Byte:
      return torch::executor::ScalarType::Byte;
    case c10::ScalarType::Char:
      return torch::executor::ScalarType::Char;
    case c10::ScalarType::Int:
      return torch::executor::ScalarType::Int;
    case c10::ScalarType::Float:
      return torch::executor::ScalarType::Float;
    case c10::ScalarType::Long:
      return torch::executor::ScalarType::Long;
    case c10::ScalarType::Double:
      return torch::executor::ScalarType::Double;
    case c10::ScalarType::Bool:
      return torch::executor::ScalarType::Bool;
    case c10::ScalarType::QInt8:
      return torch::executor::ScalarType::QInt8;
    case c10::ScalarType::QUInt8:
      return torch::executor::ScalarType::QUInt8;
    default:
      ET_ASSERT_UNREACHABLE();
  }
}

c10::ScalarType execuTorchtoTorchScalarType(torch::executor::ScalarType type) {
  switch (type) {
    case torch::executor::ScalarType::Byte:
      return c10::ScalarType::Byte;
    case torch::executor::ScalarType::Char:
      return c10::ScalarType::Char;
    case torch::executor::ScalarType::Int:
      return c10::ScalarType::Int;
    case torch::executor::ScalarType::Float:
      return c10::ScalarType::Float;
    case torch::executor::ScalarType::Long:
      return c10::ScalarType::Long;
    case torch::executor::ScalarType::Double:
      return c10::ScalarType::Double;
    case torch::executor::ScalarType::Bool:
      return c10::ScalarType::Bool;
    case torch::executor::ScalarType::QInt8:
      return c10::ScalarType::QInt8;
    case torch::executor::ScalarType::QUInt8:
      return c10::ScalarType::QUInt8;
    default:
      ET_ASSERT_UNREACHABLE();
  }
}

/*
 * Following makes two assumptions:
 * 1. aten_tensor's lifetime is longer than the liftime within which mutable_et
 * is consumed
 * 2. memory previously allocated to mutable_et, is leaked. However under the
 * assumption , a strong one, that, such memory is arena allocated whose
 * lifetime is tied to model's lifetime, we assume that memory is not leaked as
 * it is freed when arean is freed.
 * @param[in] aten_tensor: Input at::Tensor
 * @param[in/out] mutable_et: ETensor whose underlying memory now will alias to
 * aten_tensor
 */
void alias_etensor_to_attensor(
    at::Tensor& aten_tensor,
    torch::executor::Tensor& mutable_et) {
  // TODO(kimishpatel): contiguous according to memformat
  // Right now we assume everything is channels first contiguous
  // Note that input tensor must be contiguous for us to alias.
  // Mixing aliasing and copying is dangerous since if we aliased
  // the instance of mutatble_et to aten_tensor in the previous call,
  // then in the next call copying will not be the correct behavior.
  ET_CHECK_MSG(aten_tensor.is_contiguous(), "Input tensor must be contiguous");
  check_tensor_meta(aten_tensor, mutable_et);
  mutable_et.unsafeGetTensorImpl()->set_data(aten_tensor.data_ptr());
}

at::Tensor alias_attensor_to_etensor(const torch::executor::Tensor& etensor) {
  c10::ScalarType dtype = execuTorchtoTorchScalarType(etensor.scalar_type());
  std::vector<int64_t> at_tensor_sizes(
      etensor.sizes().begin(), etensor.sizes().end());
  at::Tensor t = at::from_blob(
      etensor.data_ptr(), at_tensor_sizes, at::TensorOptions(dtype));
  check_tensor_meta(t, etensor);
  return t;
}

std::unique_ptr<torch::executor::TensorImpl> eTensorFromAtTensor(
    const at::Tensor& tensor,
    KeepAliveSizes& keep_alive) {
  auto sizes = tensor.sizes();
  auto options = tensor.options();
  keep_alive.sizes32.emplace_back(sizes.size());
  auto& sizes32 = keep_alive.sizes32.back();
  for (size_t i = 0; i < sizes.size(); ++i) {
    // NOLINTNEXTLINE
    sizes32[i] = sizes[i];
  }

  const torch::executor::ScalarType edtype =
      torchToExecuTorchScalarType(options.dtype());

  return std::make_unique<torch::executor::TensorImpl>(
      edtype, sizes32.size(), sizes32.data(), tensor.data_ptr());
}

at::Tensor atTensorFromETensor(
    torch::executor::TensorImpl* etensor,
    KeepAliveSizes& keep_alive) {
  c10::ScalarType dtype = execuTorchtoTorchScalarType(etensor->scalar_type());
  keep_alive.sizes64.emplace_back(etensor->sizes().size());
  auto& sizes64 = keep_alive.sizes64.back();
  for (size_t i = 0; i < etensor->sizes().size(); ++i) {
    // NOLINTNEXTLINE
    sizes64[i] = etensor->sizes()[i];
  }
  return at::from_blob(
      etensor->mutable_data(), sizes64, at::TensorOptions(dtype));
}

} // namespace util
} // namespace torch
