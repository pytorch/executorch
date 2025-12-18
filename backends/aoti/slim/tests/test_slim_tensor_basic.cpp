/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/factory/Empty.h>
#include <executorch/backends/aoti/slim/factory/Factory.h>
#include <executorch/backends/aoti/slim/factory/FromBlob.h>
#include <executorch/backends/aoti/slim/factory/FromScalar.h>

namespace standalone::slim {
namespace {

TEST(SlimTensorBasicTest, EmptyTensorCreation) {
  auto tensor =
      empty({2, 3, 4}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(2), 4);
  EXPECT_EQ(tensor.numel(), 24);
  EXPECT_EQ(tensor.dtype(), standalone::c10::ScalarType::Float);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(SlimTensorBasicTest, EmptyTensorContiguousStrides) {
  auto tensor =
      empty({2, 3, 4}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  EXPECT_EQ(tensor.stride(0), 12);
  EXPECT_EQ(tensor.stride(1), 4);
  EXPECT_EQ(tensor.stride(2), 1);
}

TEST(SlimTensorBasicTest, ZerosTensorCreation) {
  auto tensor = zeros({3, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  EXPECT_EQ(tensor.numel(), 9);
  float* data = static_cast<float*>(tensor.data_ptr());
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(data[i], 0.0f);
  }
}

TEST(SlimTensorBasicTest, OnesTensorCreation) {
  auto tensor = ones({2, 2}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  EXPECT_EQ(tensor.numel(), 4);
  float* data = static_cast<float*>(tensor.data_ptr());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(data[i], 1.0f);
  }
}

TEST(SlimTensorBasicTest, FillTensor) {
  auto tensor = empty({2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  tensor.fill_(5.0f);
  float* data = static_cast<float*>(tensor.data_ptr());
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], 5.0f);
  }
}

TEST(SlimTensorBasicTest, FromBlobNonOwning) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor = from_blob(
      data.data(), {2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.numel(), 6);
  EXPECT_EQ(tensor.data_ptr(), data.data());
}

TEST(SlimTensorBasicTest, Clone) {
  auto tensor = empty({2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  tensor.fill_(3.14f);

  auto cloned = tensor.clone();
  EXPECT_NE(cloned.data_ptr(), tensor.data_ptr());
  EXPECT_EQ(cloned.sizes(), tensor.sizes());
  EXPECT_EQ(cloned.strides(), tensor.strides());

  float* cloned_data = static_cast<float*>(cloned.data_ptr());
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(cloned_data[i], 3.14f);
  }
}

TEST(SlimTensorBasicTest, CopyFrom) {
  auto src = empty({2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  src.fill_(2.5f);

  auto dst = empty({2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  dst.copy_(src);

  float* dst_data = static_cast<float*>(dst.data_ptr());
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(dst_data[i], 2.5f);
  }
}

TEST(SlimTensorBasicTest, Reshape) {
  auto tensor = empty({2, 6}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  tensor.fill_(1.0f);

  auto reshaped = tensor.reshape({3, 4});
  EXPECT_EQ(reshaped.dim(), 2);
  EXPECT_EQ(reshaped.size(0), 3);
  EXPECT_EQ(reshaped.size(1), 4);
  EXPECT_EQ(reshaped.numel(), 12);
}

TEST(SlimTensorBasicTest, Transpose) {
  auto tensor = empty({2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  auto transposed = tensor.transpose(0, 1);
  EXPECT_EQ(transposed.size(0), 3);
  EXPECT_EQ(transposed.size(1), 2);
}

TEST(SlimTensorBasicTest, Permute) {
  auto tensor =
      empty({2, 3, 4}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  auto permuted = tensor.permute({2, 0, 1});
  EXPECT_EQ(permuted.size(0), 4);
  EXPECT_EQ(permuted.size(1), 2);
  EXPECT_EQ(permuted.size(2), 3);
}

TEST(SlimTensorBasicTest, Narrow) {
  auto tensor = empty({10}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  for (int i = 0; i < 10; ++i) {
    static_cast<float*>(tensor.data_ptr())[i] = static_cast<float>(i);
  }

  auto narrowed = tensor.narrow(0, 2, 5);
  EXPECT_EQ(narrowed.dim(), 1);
  EXPECT_EQ(narrowed.size(0), 5);

  float* narrowed_data = static_cast<float*>(narrowed.data_ptr());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(narrowed_data[i], static_cast<float>(i + 2));
  }
}

TEST(SlimTensorBasicTest, EmptyLike) {
  auto tensor =
      empty({2, 3, 4}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  auto empty_like_tensor = empty_like(tensor);
  EXPECT_EQ(empty_like_tensor.sizes(), tensor.sizes());
  EXPECT_EQ(empty_like_tensor.dtype(), tensor.dtype());
  EXPECT_EQ(empty_like_tensor.device(), tensor.device());
}

TEST(SlimTensorBasicTest, ZerosLike) {
  auto tensor = empty({2, 3}, standalone::c10::ScalarType::Float, CPU_DEVICE);
  auto zeros_tensor = zeros_like(tensor);
  EXPECT_EQ(zeros_tensor.sizes(), tensor.sizes());

  float* data = static_cast<float*>(zeros_tensor.data_ptr());
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], 0.0f);
  }
}

} // namespace
} // namespace standalone::slim
