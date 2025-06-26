/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/times.h>

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <on_device_ai/Assistant/Jarvis/lsps/memory.h>

namespace cadence {
namespace impl {
namespace G3 {
namespace native {
namespace {

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::Result;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

#define ASSERT_OK_AND_UNWRAP(result__) \
  ({                                   \
    auto et_result__ = (result__);     \
    ASSERT_TRUE(et_result__.ok());     \
    std::move(*et_result__);           \
  })

Result<Tensor> getAllocatedTensorLike(
    const Tensor& out,
    MemoryAllocator& allocator,
    const int alignment = MemoryAllocator::kDefaultAlignment) {
  void* data = allocator.allocate(out.nbytes(), alignment);
  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr, MemoryAllocationFailed, "allocation failed");
  std::memcpy(data, out.const_data_ptr(), out.nbytes());

  auto* impl = allocator.allocateInstance<TensorImpl>(alignment);
  ET_CHECK_OR_RETURN_ERROR(
      impl != nullptr, MemoryAllocationFailed, "allocation failed");
  auto* sizes =
      allocator.allocateList<TensorImpl::SizesType>(out.dim(), alignment);
  std::memcpy(
      sizes, out.sizes().data(), out.dim() * sizeof(TensorImpl::SizesType));
  auto* dimOrder =
      allocator.allocateList<TensorImpl::DimOrderType>(out.dim(), alignment);
  std::memcpy(
      dimOrder,
      out.sizes().data(),
      out.dim() * sizeof(TensorImpl::DimOrderType));
  auto* strides =
      allocator.allocateList<TensorImpl::StridesType>(out.dim(), alignment);
  std::memcpy(
      strides,
      out.strides().data(),
      out.dim() * sizeof(TensorImpl::StridesType));

  new (impl) TensorImpl(
      out.dtype(),
      out.dim(),
      sizes,
      data,
      dimOrder,
      strides,
      out.shape_dynamism());
  return Tensor(impl);
}

class FusionG3OperatorTest : public OperatorTest {
 public:
 protected:
  Tensor& mul_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return cadence::impl::G3::native::mul_out(context_, a, b, out);
  }
};

TEST_F(FusionG3OperatorTest, MulDTCMMemAllocTest) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> size_a{1, 48, 8, 32}, size_b{1, 48, 1, 32};

  constexpr int kDtcmBank0Index = 0;
  MemoryAllocator allocator(
      cadence::memory::Memory::sizes[kDtcmBank0Index],
      cadence::memory::Memory::addrs[kDtcmBank0Index]);
  constexpr int kAlignment = 16;
  Tensor tensor_a = ASSERT_OK_AND_UNWRAP(
      getAllocatedTensorLike(tf.ones(size_a), allocator, kAlignment));
  Tensor tensor_b = ASSERT_OK_AND_UNWRAP(
      getAllocatedTensorLike(tf.ones(size_b), allocator, kAlignment));
  Tensor out = tf.zeros(size_a);

  mul_out(tensor_a, tensor_b, out);
  EXPECT_TENSOR_EQ(out, tf.ones(size_a));
}

} // namespace
} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence
