#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

namespace torch {
namespace executor {

TEST(TensorTest, InvalidScalarType) {
  TensorImpl::SizesType sizes[1] = {1};
  // A type that executorch doesn't support yet.
  ET_EXPECT_DEATH({ TensorImpl x(ScalarType::BFloat16, 1, sizes); }, "");

  // The literal Undefined type.
  ET_EXPECT_DEATH({ TensorImpl y(ScalarType::Undefined, 1, sizes); }, "");

  // An int value that doesn't map to a valid enum value
  ET_EXPECT_DEATH({ TensorImpl y(ScalarType::NumOptions, 1, sizes); }, "");
}

TEST(TensorTest, StorageOffset) {
  TensorImpl::SizesType sizes[1] = {5};
  TensorImpl::DimOrderType dim_order[1] = {0};
  int32_t data[5] = {0, 0, 1, 0, 0};
  auto a_impl =
      TensorImpl(ScalarType::Int, 1, sizes, data, dim_order, nullptr, 0);
  auto b_impl =
      TensorImpl(ScalarType::Int, 1, sizes, data, dim_order, nullptr, 2);
  Tensor a(&a_impl);
  Tensor b(&b_impl);

  EXPECT_EQ(a_impl.scalar_type(), ScalarType::Int);
  EXPECT_EQ(b_impl.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.scalar_type(), ScalarType::Int);
  EXPECT_EQ(b.scalar_type(), ScalarType::Int);
  EXPECT_EQ(0, a.const_data_ptr<int32_t>()[0]);
  EXPECT_EQ(1, b.const_data_ptr<int32_t>()[0]);
}

TEST(TensorTest, SetData) {
  TensorImpl::SizesType sizes[1] = {5};
  TensorImpl::DimOrderType dim_order[1] = {0};
  int32_t data[5] = {0, 0, 1, 0, 0};
  auto a_impl =
      TensorImpl(ScalarType::Int, 1, sizes, data, dim_order, nullptr, 0);
  auto a = Tensor(&a_impl);
  EXPECT_EQ(a.const_data_ptr(), data);
  a.set_data(nullptr);
  EXPECT_EQ(a.const_data_ptr(), nullptr);
}

TEST(TensorTest, Strides) {
  TensorImpl::SizesType sizes[2] = {2, 2};
  TensorImpl::DimOrderType dim_order[2] = {0, 1};
  int32_t data[4] = {0, 0, 1, 1};
  TensorImpl::StridesType strides[2] = {2, 1};
  auto a_impl = TensorImpl(ScalarType::Int, 2, sizes, data, dim_order, strides);
  Tensor a(&a_impl);

  EXPECT_EQ(a_impl.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0], 0);
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0 + a.strides()[0]], 1);
}

TEST(TensorTest, ModifyDataOfConstTensor) {
  TensorImpl::SizesType sizes[1] = {1};
  TensorImpl::DimOrderType dim_order[2] = {0};
  int32_t data[1] = {1};
  auto a_impl = TensorImpl(ScalarType::Int, 1, sizes, data, dim_order);
  const Tensor a(&a_impl);
  a.mutable_data_ptr<int32_t>()[0] = 0;

  EXPECT_EQ(a_impl.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0], 0);
}

} // namespace executor
} // namespace torch
