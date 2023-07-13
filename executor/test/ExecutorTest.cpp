#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/values/Evalue.h>
#include <executorch/executor/Executor.h>
#include <executorch/pytree/pytree.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <executorch/util/TestMemoryConfig.h>

namespace torch {
namespace executor {

class ExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(ExecutorTest, Tensor) {
  TensorImpl::SizesType sizes[2] = {2, 2};
  int32_t data[4]{1, 2, 3, 4};
  auto a_impl = TensorImpl(ScalarType::Int, 2, sizes, data);
  Tensor a(&a_impl);

  auto data_p = a.const_data_ptr<int32_t>();
  ASSERT_EQ(data_p[0], 1);
  ASSERT_EQ(data_p[1], 2);
  ASSERT_EQ(data_p[2], 3);
  ASSERT_EQ(data_p[3], 4);
}

TEST_F(ExecutorTest, EValue) {
  TensorImpl::SizesType sizes[2] = {2, 2};
  int32_t data[4]{1, 2, 3, 4};
  auto a_impl = TensorImpl(ScalarType::Int, 2, sizes, data);
  Tensor a(&a_impl);

  EValue v(a);
  ASSERT_TRUE(v.isTensor());
  ASSERT_EQ(v.toTensor().nbytes(), 16);
}

TEST_F(ExecutorTest, Registry) {
  auto func = getOpsFn("aten::add.out");
  ASSERT_TRUE(func);

  EValue values[4];

  TensorImpl::SizesType a_sizes[2] = {2, 2};
  int32_t a_data[4]{1, 2, 3, 4};
  auto a_impl = TensorImpl(ScalarType::Int, 2, a_sizes, a_data);
  Tensor a(&a_impl);
  values[0] = EValue(a);

  TensorImpl::SizesType b_sizes[2] = {2, 2};
  int32_t b_data[4]{5, 6, 7, 8};
  auto b_impl = TensorImpl(ScalarType::Int, 2, b_sizes, b_data);
  Tensor b(&b_impl);
  values[1] = EValue(b);

  values[2] = Scalar(1);

  TensorImpl::SizesType c_sizes[2] = {2, 2};
  int32_t c_data[4]{0, 0, 0, 0};
  auto c_impl = TensorImpl(ScalarType::Int, 2, c_sizes, c_data);
  Tensor c(&c_impl);
  values[3] = EValue(c);

  EValue* kernel_values[5];
  for (size_t i = 0; i < 4; i++) {
    kernel_values[i] = &values[i];
  }
  // x and x_out args are same evalue for out variant kernels
  kernel_values[4] = &values[3];
  KernelRuntimeContext context{};
  func(context, kernel_values);
  auto c_ptr = values[3].toTensor().const_data_ptr<int32_t>();
  ASSERT_EQ(c_ptr[3], 12);
}

TEST_F(ExecutorTest, IntArrayRefSingleElement) {
  // Create an IntArrayRef with a single element. `ref` will contain a pointer
  // to `one`, which must outlive the array ref.
  const IntArrayRef::value_type one = 1;
  IntArrayRef ref(one);
  EXPECT_EQ(ref[0], 1);
}

TEST_F(ExecutorTest, IntArrayRefDataAndLength) {
  // Create an IntArrayRef from an array. `ref` will contain a pointer to
  // `array`, which must outlive the array ref.
  const IntArrayRef::value_type array[4] = {5, 6, 7, 8};
  const IntArrayRef::size_type length = 4;
  IntArrayRef ref(array, length);

  EXPECT_EQ(ref.size(), length);
  EXPECT_EQ(ref.front(), 5);
  EXPECT_EQ(ref.back(), 8);
}

TEST_F(ExecutorTest, EValueFromScalar) {
  Scalar b((bool)true);
  Scalar i((int64_t)2);
  Scalar d((double)3.0);

  EValue evalue_b(b);
  ASSERT_TRUE(evalue_b.isScalar());
  ASSERT_TRUE(evalue_b.isBool());
  ASSERT_EQ(evalue_b.toBool(), true);

  EValue evalue_i(i);
  ASSERT_TRUE(evalue_i.isScalar());
  ASSERT_TRUE(evalue_i.isInt());
  ASSERT_EQ(evalue_i.toInt(), 2);

  EValue evalue_d(d);
  ASSERT_TRUE(evalue_d.isScalar());
  ASSERT_TRUE(evalue_d.isDouble());
  ASSERT_NEAR(evalue_d.toDouble(), 3.0, 0.01);
}

TEST_F(ExecutorTest, EValueToScalar) {
  EValue v((int64_t)2);
  ASSERT_TRUE(v.isScalar());

  Scalar s = v.toScalar();
  ASSERT_TRUE(s.isIntegral(false));
  ASSERT_EQ(s.to<int64_t>(), 2);
}

void test_op(KernelRuntimeContext& /*unused*/, EValue** /*unused*/) {}

TEST_F(ExecutorTest, OpRegistration) {
  auto s1 = register_operators({Operator("test", test_op)});
  auto s2 = register_operators({Operator("test_2", test_op)});
  ASSERT_EQ(Error::Ok, s1);
  ASSERT_EQ(Error::Ok, s2);
  ET_EXPECT_DEATH(
      { auto s3 = register_operators({Operator("test", test_op)}); }, "");

  ASSERT_TRUE(hasOpsFn("test"));
  ASSERT_TRUE(hasOpsFn("test_2"));
}

TEST_F(ExecutorTest, OpRegistrationWithContext) {
  auto op = Operator(
      "test_op_with_context",
      [](KernelRuntimeContext& context, EValue** values) {
        (void)context;
        *(values[0]) = Scalar(100);
      });
  auto s1 = register_operators({op});
  ASSERT_EQ(Error::Ok, s1);
  ASSERT_TRUE(hasOpsFn("test_op_with_context"));

  auto func = getOpsFn("test_op_with_context");
  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  KernelRuntimeContext context{};
  func(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(ExecutorTest, OpRegistrationAddMul) {
  ASSERT_TRUE(hasOpsFn("aten::add.out"));
  ASSERT_TRUE(hasOpsFn("aten::mul.out"));
}

TEST(PyTreeEValue, List) {
  std::string spec = "L2#1#1($,$)";

  Scalar i((int64_t)2);
  Scalar d((double)3.0);
  EValue items[2] = {i, d};

  auto c = pytree::unflatten(spec, items);
  ASSERT_TRUE(c.isList());
  ASSERT_EQ(c.size(), 2);

  const auto& child0 = c[0];
  const auto& child1 = c[1];

  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());

  EValue ev_child0 = child0;
  ASSERT_TRUE(ev_child0.isScalar());
  ASSERT_TRUE(ev_child0.isInt());
  ASSERT_EQ(ev_child0.toInt(), 2);

  ASSERT_TRUE(child1.leaf().isScalar());
  ASSERT_TRUE(child1.leaf().isDouble());
  ASSERT_NEAR(child1.leaf().toDouble(), 3.0, 0.01);
}

auto unflatten(EValue* items) {
  std::string spec = "D4#1#1#1#1('key0':$,1:$,23:$,123:$)";
  return pytree::unflatten(spec, items);
}

TEST(PyTreeEValue, DestructedSpec) {
  Scalar i0((int64_t)2);
  Scalar d1((double)3.0);
  Scalar i2((int64_t)4);
  Scalar d3((double)5.0);
  EValue items[4] = {i0, d1, i2, d3};
  auto c = unflatten(items);

  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 4);

  auto& key0 = c.key(0);
  auto& key1 = c.key(1);

  ASSERT_TRUE(key0 == pytree::Key("key0"));
  ASSERT_TRUE(key1 == pytree::Key(1));

  const auto& child0 = c[0];
  const auto& child1 = c[1];
  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());

  EValue ev_child0 = child0;
  ASSERT_TRUE(ev_child0.isScalar());
  ASSERT_TRUE(ev_child0.isInt());
  ASSERT_EQ(ev_child0.toInt(), 2);

  ASSERT_TRUE(child1.leaf().isScalar());
  ASSERT_TRUE(child1.leaf().isDouble());
  ASSERT_NEAR(child1.leaf().toDouble(), 3.0, 0.01);
}

TEST_F(ExecutorTest, HierarchicalAllocator) {
  constexpr size_t n_allocators = 2;
  constexpr size_t size0 = 4;
  constexpr size_t size1 = 8;
  uint8_t mem0[size0];
  uint8_t mem1[size1];
  MemoryAllocator allocators[n_allocators]{
      MemoryAllocator(size0, mem0), MemoryAllocator(size1, mem1)};

  HierarchicalAllocator allocator(n_allocators, allocators);

  // get_offset_address() success cases
  {
    // Total size is 4, so off=0 + size=2 fits.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/0, /*size_bytes=*/2);
    ASSERT_TRUE(address.ok());
    ASSERT_NE(address.get(), nullptr);
  }
  {
    // Total size is 8, so off=4 + size=4 fits exactly.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/1, /*offset_bytes=*/4, /*size_bytes=*/4);
    ASSERT_TRUE(address.ok());
    ASSERT_NE(address.get(), nullptr);
  }

  // get_offset_address() failure cases
  {
    // Total size is 4, so off=0 + size=5 is too large.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/4, /*size_bytes=*/5);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
  {
    // Total size is 4, so off=8 + size=0 is off the end.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/8, /*size_bytes=*/0);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
  {
    // ID too large; only two zero-indexed entries in the allocator.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/2, /*offset_bytes=*/0, /*size_bytes=*/99);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
}
} // namespace executor
} // namespace torch
