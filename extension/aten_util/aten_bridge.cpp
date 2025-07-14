/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/aten_bridge.h>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/platform/assert.h>
#include <cstring>

namespace executorch {
namespace extension {

namespace {
void check_tensor_meta(const at::Tensor& a, const executorch::aten::Tensor& b) {
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
  // check strides and dim order
  std::array<exec_aten::StridesType, executorch::runtime::kTensorDimensionLimit>
      expected_strides{};
  runtime::dim_order_to_stride_nocheck(
      b.sizes().data(), b.dim_order().data(), b.dim(), expected_strides.data());

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
    ET_CHECK_MSG(
        (b.size(i) == 1 || (b.strides()[i] == expected_strides[i])),
        "Strides don't match dim order at index:%zd, stride: %zd != expected %zd",
        i,
        ssize_t(a.stride(i)),
        ssize_t(expected_strides[i]));
  }
  // check dtype
  ET_CHECK_MSG(
      b.scalar_type() == torch_to_executorch_scalar_type(a.options().dtype()),
      "dtypes dont match a %hhd vs. b %hhd",
      static_cast<int8_t>(torch_to_executorch_scalar_type(a.options().dtype())),
      static_cast<int8_t>(b.scalar_type()));
}
} // namespace

executorch::runtime::etensor::ScalarType torch_to_executorch_scalar_type(
    caffe2::TypeMeta type) {
  const auto intermediate =
      static_cast<std::underlying_type<c10::ScalarType>::type>(
          c10::typeMetaToScalarType(type));

  ET_CHECK_MSG(
      intermediate >= 0 &&
          intermediate <= static_cast<std::underlying_type<
                              executorch::runtime::etensor::ScalarType>::type>(
                              executorch::runtime::etensor::ScalarType::UInt64),
      "ScalarType %d unsupported in Executorch",
      intermediate);
  return static_cast<executorch::runtime::etensor::ScalarType>(intermediate);
}

c10::ScalarType executorch_to_torch_scalar_type(
    torch::executor::ScalarType type) {
  const auto intermediate = static_cast<
      std::underlying_type<executorch::runtime::etensor::ScalarType>::type>(
      type);

  ET_CHECK_MSG(
      intermediate >= 0 &&
          intermediate <= static_cast<std::underlying_type<
                              executorch::runtime::etensor::ScalarType>::type>(
                              executorch::runtime::etensor::ScalarType::UInt64),
      "ScalarType %d unsupported in Executorch",
      intermediate);
  return static_cast<c10::ScalarType>(intermediate);
}

/*
 * Following makes two assumptions:
 * 1. aten_tensor's lifetime is longer than the liftime within which mutable_et
 * is consumed
 * 2. memory previously allocated to mutable_et, is leaked. However under the
 * assumption , a strong one, that, such memory is arena allocated whose
 * lifetime is tied to model's lifetime, we assume that memory is not leaked as
 * it is freed when arean is freed.
 * @param[in] aten_tensor Input at::Tensor
 * @param[in/out] mutable_et ETensor whose underlying memory now will alias to
 * aten_tensor
 */
void alias_etensor_to_attensor(
    at::Tensor& aten_tensor,
    torch::executor::Tensor& mutable_et) {
  ET_CHECK_MSG(
      aten_tensor.is_contiguous() ||
          aten_tensor.is_contiguous(at::MemoryFormat::ChannelsLast),
      "Input tensor must have contiguous or channels last memory format");

  check_tensor_meta(aten_tensor, mutable_et);
  mutable_et.unsafeGetTensorImpl()->set_data(aten_tensor.mutable_data_ptr());
}

at::Tensor alias_attensor_to_etensor(const torch::executor::Tensor& etensor) {
  c10::ScalarType dtype =
      executorch_to_torch_scalar_type(etensor.scalar_type());
  std::vector<int64_t> at_tensor_sizes(
      etensor.sizes().begin(), etensor.sizes().end());
  std::vector<int64_t> at_tensor_strides(
      etensor.strides().begin(), etensor.strides().end());

  at::Tensor t = at::from_blob(
      etensor.mutable_data_ptr(),
      at_tensor_sizes,
      at_tensor_strides,
      at::TensorOptions(dtype));

  check_tensor_meta(t, etensor);
  return t;
}

TensorPtr alias_tensor_ptr_to_attensor(at::Tensor& t) {
  return make_tensor_ptr(
      {t.sizes().begin(), t.sizes().end()},
      t.mutable_data_ptr(),
      torch::executor::ScalarType(t.scalar_type()));
}

} // namespace extension
} // namespace executorch
