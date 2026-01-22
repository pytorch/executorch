/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace executorch::backends::aoti::slim {

// =============================================================================
// Test Data Structures for Parameterized Tests
// =============================================================================

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<int8_t> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::Char;
  static constexpr const char* name = "Char";
  static int8_t test_value(size_t i) {
    return static_cast<int8_t>(i % 127);
  }
};

template <>
struct DTypeTraits<int16_t> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::Short;
  static constexpr const char* name = "Short";
  static int16_t test_value(size_t i) {
    return static_cast<int16_t>(i * 10);
  }
};

template <>
struct DTypeTraits<int32_t> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::Int;
  static constexpr const char* name = "Int";
  static int32_t test_value(size_t i) {
    return static_cast<int32_t>(i * 100);
  }
};

template <>
struct DTypeTraits<int64_t> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::Long;
  static constexpr const char* name = "Long";
  static int64_t test_value(size_t i) {
    return static_cast<int64_t>(i * 1000);
  }
};

template <>
struct DTypeTraits<float> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::Float;
  static constexpr const char* name = "Float";
  static float test_value(size_t i) {
    return static_cast<float>(i) * 1.5f;
  }
};

template <>
struct DTypeTraits<bool> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::Bool;
  static constexpr const char* name = "Bool";
  static bool test_value(size_t i) {
    return (i % 2) == 0;
  }
};

template <>
struct DTypeTraits<c10::BFloat16> {
  static constexpr c10::ScalarType dtype = c10::ScalarType::BFloat16;
  static constexpr const char* name = "BFloat16";
  static c10::BFloat16 test_value(size_t i) {
    return c10::BFloat16(static_cast<float>(i) * 0.5f);
  }
};

// =============================================================================
// Typed Test Fixture
// =============================================================================

template <typename T>
class SlimTensorDTypeTest : public ::testing::Test {
 protected:
  static constexpr c10::ScalarType kDType = DTypeTraits<T>::dtype;
  static constexpr size_t kNumel = 24;
  static constexpr std::array<int64_t, 3> kSizes = {2, 3, 4};

  SlimTensor create_tensor() {
    return empty({2, 3, 4}, kDType);
  }

  void fill_tensor(SlimTensor& tensor) {
    T* data = static_cast<T*>(tensor.data_ptr());
    for (size_t i = 0; i < tensor.numel(); ++i) {
      data[i] = DTypeTraits<T>::test_value(i);
    }
  }

  void verify_tensor_values(const SlimTensor& tensor) {
    const T* data = static_cast<const T*>(tensor.data_ptr());
    for (size_t i = 0; i < tensor.numel(); ++i) {
      T expected = DTypeTraits<T>::test_value(i);
      if constexpr (std::is_same_v<T, float>) {
        EXPECT_FLOAT_EQ(data[i], expected) << "Mismatch at index " << i;
      } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
        EXPECT_FLOAT_EQ(
            static_cast<float>(data[i]), static_cast<float>(expected))
            << "Mismatch at index " << i;
      } else {
        EXPECT_EQ(data[i], expected) << "Mismatch at index " << i;
      }
    }
  }
};

// Define the types to test
using DTypeTestTypes = ::testing::
    Types<int8_t, int16_t, int32_t, int64_t, float, bool, c10::BFloat16>;

TYPED_TEST_SUITE(SlimTensorDTypeTest, DTypeTestTypes);

// =============================================================================
// Core Tensor Creation Tests
// =============================================================================

TYPED_TEST(SlimTensorDTypeTest, CreateEmptyTensor) {
  SlimTensor tensor = this->create_tensor();

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dtype(), this->kDType);
  EXPECT_EQ(tensor.dim(), 3u);
  EXPECT_EQ(tensor.numel(), this->kNumel);
  EXPECT_TRUE(tensor.is_cpu());
  EXPECT_TRUE(tensor.is_contiguous());
}

TYPED_TEST(SlimTensorDTypeTest, CorrectElementSize) {
  SlimTensor tensor = this->create_tensor();
  EXPECT_EQ(tensor.itemsize(), sizeof(TypeParam));
}

TYPED_TEST(SlimTensorDTypeTest, CorrectNbytes) {
  SlimTensor tensor = this->create_tensor();
  EXPECT_EQ(tensor.nbytes(), this->kNumel * sizeof(TypeParam));
}

TYPED_TEST(SlimTensorDTypeTest, DataPtrIsValid) {
  SlimTensor tensor = this->create_tensor();
  EXPECT_NE(tensor.data_ptr(), nullptr);
}

// =============================================================================
// Data Read/Write Tests
// =============================================================================

TYPED_TEST(SlimTensorDTypeTest, WriteAndReadData) {
  SlimTensor tensor = this->create_tensor();
  this->fill_tensor(tensor);
  this->verify_tensor_values(tensor);
}

TYPED_TEST(SlimTensorDTypeTest, ZeroInitialize) {
  SlimTensor tensor = this->create_tensor();
  std::memset(tensor.data_ptr(), 0, tensor.nbytes());

  const TypeParam* data = static_cast<const TypeParam*>(tensor.data_ptr());
  for (size_t i = 0; i < tensor.numel(); ++i) {
    if constexpr (std::is_same_v<TypeParam, bool>) {
      EXPECT_FALSE(data[i]) << "Non-zero at index " << i;
    } else if constexpr (std::is_same_v<TypeParam, float>) {
      EXPECT_FLOAT_EQ(data[i], 0.0f) << "Non-zero at index " << i;
    } else if constexpr (std::is_same_v<TypeParam, c10::BFloat16>) {
      EXPECT_FLOAT_EQ(static_cast<float>(data[i]), 0.0f)
          << "Non-zero at index " << i;
    } else {
      EXPECT_EQ(data[i], static_cast<TypeParam>(0))
          << "Non-zero at index " << i;
    }
  }
}

// =============================================================================
// Copy Tests
// =============================================================================

TYPED_TEST(SlimTensorDTypeTest, CopyContiguousTensor) {
  SlimTensor src = this->create_tensor();
  this->fill_tensor(src);

  SlimTensor dst = this->create_tensor();
  dst.copy_(src);

  this->verify_tensor_values(dst);
}

TYPED_TEST(SlimTensorDTypeTest, CopyPreservesSourceData) {
  SlimTensor src = this->create_tensor();
  this->fill_tensor(src);

  SlimTensor dst = this->create_tensor();
  dst.copy_(src);

  // Modify dst and verify src is unchanged
  std::memset(dst.data_ptr(), 0, dst.nbytes());

  // src should still have original values
  this->verify_tensor_values(src);
}

// =============================================================================
// Empty Strided Tests
// =============================================================================

TYPED_TEST(SlimTensorDTypeTest, EmptyStridedCreation) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};

  SlimTensor tensor =
      empty_strided(makeArrayRef(sizes), makeArrayRef(strides), this->kDType);

  EXPECT_EQ(tensor.dtype(), this->kDType);
  EXPECT_TRUE(tensor.is_contiguous());
}

TYPED_TEST(SlimTensorDTypeTest, NonContiguousStrides) {
  std::vector<int64_t> sizes = {3, 2};
  std::vector<int64_t> strides = {1, 3};

  SlimTensor tensor =
      empty_strided(makeArrayRef(sizes), makeArrayRef(strides), this->kDType);

  EXPECT_EQ(tensor.dtype(), this->kDType);
  EXPECT_FALSE(tensor.is_contiguous());
}

// =============================================================================
// Empty Like Tests
// =============================================================================

TYPED_TEST(SlimTensorDTypeTest, EmptyLikePreservesDType) {
  SlimTensor original = this->create_tensor();
  SlimTensor copy = empty_like(original);

  EXPECT_EQ(copy.dtype(), original.dtype());
  EXPECT_EQ(copy.numel(), original.numel());
  EXPECT_EQ(copy.dim(), original.dim());
  EXPECT_NE(copy.data_ptr(), original.data_ptr());
}

// =============================================================================
// Dimension and Shape Tests
// =============================================================================

TYPED_TEST(SlimTensorDTypeTest, OneDimensionalTensor) {
  SlimTensor tensor = empty({10}, this->kDType);

  EXPECT_EQ(tensor.dim(), 1u);
  EXPECT_EQ(tensor.numel(), 10u);
  EXPECT_EQ(tensor.size(0), 10);
  EXPECT_EQ(tensor.stride(0), 1);
}

TYPED_TEST(SlimTensorDTypeTest, FourDimensionalTensor) {
  SlimTensor tensor = empty({2, 3, 4, 5}, this->kDType);

  EXPECT_EQ(tensor.dim(), 4u);
  EXPECT_EQ(tensor.numel(), 120u);
  EXPECT_TRUE(tensor.is_contiguous());
}

TYPED_TEST(SlimTensorDTypeTest, ZeroSizedTensor) {
  SlimTensor tensor = empty({0, 5}, this->kDType);

  EXPECT_TRUE(tensor.is_empty());
  EXPECT_EQ(tensor.numel(), 0u);
  EXPECT_EQ(tensor.dtype(), this->kDType);
}

} // namespace executorch::backends::aoti::slim
