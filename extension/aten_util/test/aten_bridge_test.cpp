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
using namespace executorch::extension;

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

TEST(ATenBridgeTest, AliasTensorPtrToATenTensor) {
  auto at_tensor = generate_at_tensor();
  const auto& et_tensor_ptr = alias_tensor_ptr_to_attensor(at_tensor);
  alias_etensor_to_attensor(at_tensor, *et_tensor_ptr);
  EXPECT_EQ(at_tensor.const_data_ptr(), et_tensor_ptr->const_data_ptr());
}

// 0-dim (scalar) tensors legitimately have empty sizes/strides arrays whose
// `.data()` may return nullptr. Regression test for T270603238: ensure
// check_tensor_meta does not abort on valid 0-dim tensors. We pass nullptr
// explicitly for sizes/dim_order/strides because std::vector::data() on an
// empty vector is implementation-defined (libstdc++/libc++ may return a
// non-null sentinel) — using nullptr makes the regression deterministic
// across STL implementations.
TEST(ATenBridgeTest, AliasETensorToATenTensorZeroDim) {
  auto at_tensor = at::scalar_tensor(42.0f);
  ASSERT_EQ(at_tensor.dim(), 0);
  auto dtype = torchToExecuTorchScalarType(at_tensor.options().dtype());
  torch::executor::TensorImpl tensor_impl(
      dtype,
      /*dim=*/0,
      /*sizes=*/nullptr,
      /*data=*/nullptr,
      /*dim_order=*/nullptr,
      /*strides=*/nullptr);
  torch::executor::Tensor etensor(&tensor_impl);
  alias_etensor_to_attensor(at_tensor, etensor);
  EXPECT_EQ(at_tensor.const_data_ptr(), etensor.const_data_ptr());
}

TEST(ATenBridgeTest, AliasATTensorToETensorZeroDim) {
  auto at_tensor = at::scalar_tensor(7);
  ASSERT_EQ(at_tensor.dim(), 0);
  auto dtype = torchToExecuTorchScalarType(at_tensor.options().dtype());
  std::vector<uint8_t> etensor_data(at_tensor.nbytes());
  torch::executor::TensorImpl tensor_impl(
      dtype,
      /*dim=*/0,
      /*sizes=*/nullptr,
      etensor_data.data(),
      /*dim_order=*/nullptr,
      /*strides=*/nullptr);
  torch::executor::Tensor etensor(&tensor_impl);
  auto aliased_at_tensor = alias_attensor_to_etensor(etensor);
  EXPECT_EQ(aliased_at_tensor.dim(), 0);
  EXPECT_EQ(aliased_at_tensor.const_data_ptr(), etensor_data.data());
}

TEST(ATenBridgeTest, AliasATTensorToETensorChannelsLast) {
  auto at_tensor = at::randn({2, 3, 4, 5}).to(at::MemoryFormat::ChannelsLast);
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  std::vector<Tensor::DimOrderType> dim_order = {0, 2, 3, 1};
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

TEST(ATenBridgeTest, AliasATTensorToETensorFailDimOrder) {
  auto at_tensor = at::randn({2, 3, 4, 5}).to(at::MemoryFormat::ChannelsLast);
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  std::vector<Tensor::DimOrderType> dim_order = {0, 1, 2, 3};
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
  ET_EXPECT_DEATH(
      alias_attensor_to_etensor(etensor), "Strides don't match dim order");
}

TEST(ATenBridgeTest, AliasETensorToATenTensorChannelsLast) {
  auto at_tensor = at::randn({2, 3, 4, 5}).to(at::MemoryFormat::ChannelsLast);
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  std::vector<Tensor::DimOrderType> dim_order = {0, 2, 3, 1};
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

TEST(ATenBridgeTest, AliasETensorToATenTensorFailDimOrder) {
  auto at_tensor = at::randn({2, 3, 4, 5}).to(at::MemoryFormat::ChannelsLast);
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  std::vector<Tensor::DimOrderType> dim_order = {0, 1, 2, 3};
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
  ET_EXPECT_DEATH(alias_etensor_to_attensor(at_tensor, etensor), "");
}

TEST(ATenBridgeTest, AliasETensorToATenTensorFailUnsupportedDimOrder) {
  auto at_tensor =
      at::randn({1, 2, 3, 4, 5}).to(at::MemoryFormat::ChannelsLast3d);
  std::vector<Tensor::SizesType> sizes(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  std::vector<Tensor::DimOrderType> dim_order = {0, 2, 3, 4, 1};
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
  ET_EXPECT_DEATH(alias_etensor_to_attensor(at_tensor, etensor), "");
}

TEST(ATenBridgeTest, DeviceMapping) {
  // Needs no accelerator: only the mapping is under test.
  using executorch::runtime::etensor::Device;
  using executorch::runtime::etensor::DeviceType;

  EXPECT_EQ(
      executorch_to_torch_device(Device(DeviceType::CPU)).type(),
      c10::DeviceType::CPU);
  EXPECT_EQ(
      executorch_to_torch_device(Device(DeviceType::CUDA, 1)).type(),
      c10::DeviceType::CUDA);
  EXPECT_EQ(executorch_to_torch_device(Device(DeviceType::CUDA, 1)).index(), 1);
}

TEST(ATenBridgeTest, DeviceMappingAbortsOnUnknownType) {
  using executorch::runtime::etensor::Device;
  using executorch::runtime::etensor::DeviceType;
  ET_EXPECT_DEATH(
      (void)executorch_to_torch_device(Device(static_cast<DeviceType>(99))),
      "");
}

TEST(ATenBridgeTest, ReverseDeviceMapping) {
  // Needs no accelerator: only the mapping is under test.
  using executorch::runtime::etensor::DeviceType;

  const auto cpu =
      torch_to_executorch_device(c10::Device(c10::DeviceType::CPU));
  ASSERT_TRUE(cpu.has_value());
  EXPECT_EQ(cpu->type(), DeviceType::CPU);

  const auto cuda =
      torch_to_executorch_device(c10::Device(c10::DeviceType::CUDA, 7));
  ASSERT_TRUE(cuda.has_value());
  EXPECT_EQ(cuda->type(), DeviceType::CUDA);
  EXPECT_EQ(cuda->index(), 7);
}

TEST(ATenBridgeTest, ReverseDeviceMappingReportsAnUnrepresentableDevice) {
  // Reported rather than fatal, unlike the other direction: a caller can hand
  // in any device PyTorch supports, and the caller is the one holding the
  // context worth putting in the error.
  EXPECT_FALSE(torch_to_executorch_device(c10::Device(c10::DeviceType::Meta))
                   .has_value());
}

TEST(ATenBridgeTest, AliasATTensorToETensorHandlesAnEmptyTensor) {
  // An empty tensor has a null data pointer, so this is the only case that
  // distinguishes dropping the device, passing it through options alone (which
  // raises), and passing target_device. Needs no accelerator: nothing
  // allocates.
  auto at_tensor = at::empty({0});
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
      /*data=*/nullptr,
      dim_order.data(),
      strides.data(),
      executorch::runtime::TensorShapeDynamism::STATIC,
      executorch::runtime::etensor::DeviceType::CUDA,
      0);
  torch::executor::Tensor etensor(&tensor_impl);

  auto aliased = alias_attensor_to_etensor(etensor);
  EXPECT_EQ(aliased.numel(), 0);
  EXPECT_EQ(aliased.device().type(), c10::DeviceType::CUDA);
  EXPECT_EQ(aliased.device().index(), 0);
  // The dispatch key too, not just the label. The key comes from the options
  // while the label can come from target_device, so a call site that passes the
  // device only as target_device still satisfies the checks above and produces
  // a tensor that dispatches to CPU kernels.
  EXPECT_TRUE(aliased.is_cuda());
  EXPECT_TRUE(aliased.key_set().has(c10::DispatchKey::CUDA));
}
