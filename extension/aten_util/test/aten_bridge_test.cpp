/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mutex>
#include <numeric>
#include <random>

#include <executorch/extension/aten_util/aten_bridge.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace torch::executor;
using namespace torch::executor::util;

namespace {
at::Tensor generate_at_tensor() {
  return at::empty({4, 5, 6});
}
std::vector<Tensor::DimOrderType> get_default_dim_order(const at::Tensor& t) {
  std::vector<Tensor::DimOrderType> dim_order(t.dim());
  std::iota(dim_order.begin(), dim_order.end(), 0);
  return dim_order;
}
} // namespace

TEST(ATenBridgeTest, AliasETensorToATenTensor) {
  auto at_tensor = generate_at_tensor();
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  auto dim_order = get_default_dim_order(at_tensor);
  std::vector<Tensor::StridesType> strides(
      at_tensor.strides().begin(), at_tensor.strides().end());
  auto dtype = torchToExecuTorchScalarType(at_tensor.options().dtype());
  torch::executor::TensorImpl tensor_impl(
      dtype,
      at_tensor.dim(),
      sizes.data(),
      nullptr,
      dim_order.data(),
      strides.data());
  torch::executor::Tensor etensor(&tensor_impl);
  alias_etensor_to_attensor(at_tensor, etensor);
  EXPECT_EQ(at_tensor.const_data_ptr(), etensor.const_data_ptr());
}

TEST(ATenBridgeTest, AliasETensorToATenTensorFail) {
  auto at_tensor = generate_at_tensor();
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  auto dim_order = get_default_dim_order(at_tensor);
  std::vector<Tensor::StridesType> strides(
      at_tensor.strides().begin(), at_tensor.strides().end());
  auto dtype = torchToExecuTorchScalarType(at_tensor.options().dtype());
  std::unique_ptr<torch::executor::TensorImpl> tensor_impl =
      std::make_unique<TensorImpl>(
          dtype, 1, sizes.data(), nullptr, dim_order.data(), strides.data());
  torch::executor::Tensor etensor(tensor_impl.get());
  // Empty sizes on etensor
  ET_EXPECT_DEATH(alias_etensor_to_attensor(at_tensor, etensor), "");

  strides = std::vector<Tensor::StridesType>();
  tensor_impl = std::make_unique<torch::executor::TensorImpl>(
      dtype,
      at_tensor.dim(),
      sizes.data(),
      nullptr,
      dim_order.data(),
      strides.data());
  etensor = torch::executor::Tensor(tensor_impl.get());
  // Empty strides on etensor
  ET_EXPECT_DEATH(alias_etensor_to_attensor(at_tensor, etensor), "");
}

TEST(ATenBridgeTest, AliasETensorToATenTensorNonContiguous) {
  auto at_tensor = generate_at_tensor();
  auto sliced_tensor = at_tensor.slice(1, 0, 2);
  auto sliced_tensor_contig = sliced_tensor.contiguous();
  std::vector<Tensor::SizesType> sizes(
      sliced_tensor.sizes().begin(), sliced_tensor.sizes().end());
  auto dim_order = get_default_dim_order(at_tensor);
  std::vector<Tensor::StridesType> strides(
      sliced_tensor_contig.strides().begin(),
      sliced_tensor_contig.strides().end());
  auto dtype = torchToExecuTorchScalarType(sliced_tensor.options().dtype());
  std::vector<uint8_t> etensor_data(sliced_tensor_contig.nbytes());
  torch::executor::TensorImpl tensor_impl(
      dtype,
      sliced_tensor.dim(),
      sizes.data(),
      etensor_data.data(),
      dim_order.data(),
      strides.data());
  torch::executor::Tensor etensor(&tensor_impl);
  alias_etensor_to_attensor(sliced_tensor_contig, etensor);
  EXPECT_EQ(sliced_tensor_contig.const_data_ptr(), etensor.const_data_ptr());
  EXPECT_NE(sliced_tensor.const_data_ptr(), etensor.const_data_ptr());
}

TEST(ATenBridgeTest, AliasETensorToATenTensorNonContiguousFail) {
  auto at_tensor = generate_at_tensor();
  auto sliced_tensor = at_tensor.slice(1, 0, 2);
  auto sliced_tensor_contig = sliced_tensor.contiguous();
  std::vector<Tensor::SizesType> sizes(
      sliced_tensor.sizes().begin(), sliced_tensor.sizes().end());
  std::vector<Tensor::StridesType> strides(
      sliced_tensor_contig.strides().begin(),
      sliced_tensor_contig.strides().end());
  auto dtype = torchToExecuTorchScalarType(sliced_tensor.options().dtype());
  std::vector<uint8_t> etensor_data(sliced_tensor_contig.nbytes());
  auto dim_order = get_default_dim_order(at_tensor);
  torch::executor::TensorImpl tensor_impl(
      dtype,
      sliced_tensor.dim(),
      sizes.data(),
      etensor_data.data(),
      dim_order.data(),
      strides.data());
  torch::executor::Tensor etensor(&tensor_impl);
  ET_EXPECT_DEATH(alias_etensor_to_attensor(sliced_tensor, etensor), "");
}

TEST(ATenBridgeTest, AliasATTensorToETensor) {
  auto at_tensor = generate_at_tensor();
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  auto dim_order = get_default_dim_order(at_tensor);
  std::vector<Tensor::StridesType> strides(
      at_tensor.strides().begin(), at_tensor.strides().end());
  auto dtype = torchToExecuTorchScalarType(at_tensor.options().dtype());
  std::vector<uint8_t> etensor_data(at_tensor.nbytes());
  torch::executor::TensorImpl tensor_impl(
      dtype,
      at_tensor.dim(),
      sizes.data(),
      etensor_data.data(),
      dim_order.data(),
      strides.data());
  torch::executor::Tensor etensor(&tensor_impl);
  auto aliased_at_tensor = alias_attensor_to_etensor(etensor);
  EXPECT_EQ(aliased_at_tensor.const_data_ptr(), etensor_data.data());
}
