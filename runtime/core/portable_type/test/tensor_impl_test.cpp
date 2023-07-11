#include <executorch/runtime/core/portable_type/tensor_impl.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <random>

using namespace ::testing;

namespace torch {
namespace executor {

using SizesType = TensorImpl::SizesType;
using DimOrderType = TensorImpl::DimOrderType;
using StridesType = TensorImpl::StridesType;

class TensorImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(TensorImplTest, TestCtorAndGetters) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(ScalarType::Float, 2, sizes, data, dim_order, strides, 0);

  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.nbytes(), 6 * 4); // 6 4 byte floats
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.scalar_type(), ScalarType::Float);
  EXPECT_EQ(t.element_size(), 4);
  EXPECT_EQ(t.storage_offset(), 0);
  EXPECT_EQ(t.data(), data);
  EXPECT_EQ(t.mutable_data(), data);
  EXPECT_EQ(t.sizes().data(), sizes);
  EXPECT_EQ(t.sizes().size(), 2);
  EXPECT_EQ(t.strides().data(), strides);
  EXPECT_EQ(t.strides().size(), 2);
  EXPECT_EQ(t.storage_offset(), 0);
}

TEST_F(TensorImplTest, TestDataOffset) {
  SizesType sizes[1] = {1};
  DimOrderType dim_order[1] = {0};
  StridesType strides[1] = {1};
  float data[6] = {1.0, 2.0};

  TensorImpl t(ScalarType::Float, 1, sizes, data, dim_order, strides, 1);
  EXPECT_EQ(t.data(), data + 1);
}

// Verify that contig means stride[0] >= stride[1] >= ... stride[size-1] == 1
TEST_F(TensorImplTest, TestSetSizesContigContract) {
  const int RANK = 5;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 100);
  SizesType sizes[RANK] = {2, 2, 2, 2, 2};
  DimOrderType dim_order[RANK] = {0, 1, 2, 3, 4};
  StridesType strides[RANK] = {16, 8, 4, 2, 1};
  float* data = nullptr;
  TensorImpl t(
      ScalarType::Float,
      RANK,
      sizes,
      data,
      dim_order,
      strides,
      0,
      TensorShapeDynamism::DYNAMIC_UNBOUND);

  SizesType new_sizes[RANK] = {0, 0, 0, 0, 0};
  // assign random sizes between 1 and 100
  for (int i = 0; i < RANK; i++) {
    new_sizes[i] = distribution(generator);
  }
  t.set_sizes_contiguous({new_sizes, RANK});

  auto strides_ref = t.strides();
  StridesType prev = strides_ref[0];
  for (auto stride : strides_ref) {
    EXPECT_LE(stride, prev);
  }
  EXPECT_EQ(t.strides()[strides_ref.size() - 1], 1);
}

TEST_F(TensorImplTest, TestSetSizesContigZeroSizes) {
  SizesType sizes[3] = {2, 0, 3};
  DimOrderType dim_order[3] = {0, 1, 2};
  StridesType strides[3] = {3, 3, 1};
  float* data = nullptr;
  TensorImpl t(
      ScalarType::Float,
      3,
      sizes,
      data,
      dim_order,
      strides,
      0,
      TensorShapeDynamism::DYNAMIC_UNBOUND);

  SizesType new_sizes_1[3] = {1, 0, 2};
  t.set_sizes_contiguous({new_sizes_1, 3});
  EXPECT_EQ(t.size(1), 0);

  // Treat 0 dimensions as size 1 for stride calculation as thats what aten does
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 2);
  EXPECT_EQ(strides_ref[2], 1);

  // Numel is 0 for tensors with a 0 sized dimension
  EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorImplTest, TestSetSizesContigStatic) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(ScalarType::Float, 2, sizes, data, dim_order, strides, 0);

  SizesType new_sizes_1[2] = {3, 2};
  t.set_sizes_contiguous({new_sizes_1, 2});
  EXPECT_EQ(t.size(1), 2);

  // strides shouldnt change
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_2[2] = {2, 2};
  // Cant change size of a StaticShape Tensor
  ET_EXPECT_DEATH(t.set_sizes_contiguous({new_sizes_2, 2}), "");

  SizesType new_sizes_3[1] = {2};
  // Cant change rank of any Tensor
  ET_EXPECT_DEATH(t.set_sizes_contiguous({new_sizes_3, 1}), "");
}

TEST_F(TensorImplTest, TestSetSizesContigUpperBounded) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(
      ScalarType::Float,
      2,
      sizes,
      data,
      dim_order,
      strides,
      0,
      TensorShapeDynamism::DYNAMIC_BOUND);

  SizesType new_sizes_1[2] = {1, 1};
  // Can resize down
  t.set_sizes_contiguous({new_sizes_1, 2});
  EXPECT_EQ(t.size(1), 1);

  // strides contiguous
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 1);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_2[2] = {3, 2};
  // Can resize back up
  t.set_sizes_contiguous({new_sizes_2, 2});
  EXPECT_EQ(t.size(1), 2);

  // Back to original strides
  strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_3[2] = {4, 2};
  // Can't execeed capacity of UpperBounded Tensor
  ET_EXPECT_DEATH(t.set_sizes_contiguous({new_sizes_3, 2}), "");

  SizesType new_sizes_4[1] = {4};
  // Can't change rank of any Tensor
  ET_EXPECT_DEATH(t.set_sizes_contiguous({new_sizes_4, 1}), "");
}

TEST_F(TensorImplTest, TestWriteRead) {
  SizesType sizes[1] = {1};
  DimOrderType dim_order[1] = {0};
  StridesType strides[1] = {1};
  float data[6] = {1.0, 2.0};
  TensorImpl t(ScalarType::Float, 1, sizes, data, dim_order, strides, 1);

  float* x = t.mutable_data<float>();
  x[0] = -1.0;
  const float* y = t.data<float>();
  EXPECT_EQ(y[0], -1.0);
}

} // namespace executor
} // namespace torch
