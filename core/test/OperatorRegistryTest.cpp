#include <gtest/gtest.h>
#include <vector>

#include <executorch/core/OperatorRegistry.h>
#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/kernels/kernel_runtime_context.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;

namespace torch {
namespace executor {

class OperatorRegistryTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(OperatorRegistryTest, Basic) {
  Operator ops[] = {Operator("foo", [](RuntimeContext&, EValue**) {})};
  ArrayRef<Operator> ops_array = ArrayRef<Operator>(ops);
  auto s1 = register_operators(ops_array);
  EXPECT_FALSE(hasOpsFn("fpp"));
  EXPECT_TRUE(hasOpsFn("foo"));
}

TEST_F(OperatorRegistryTest, RegisterOpsMoreThanOnceDie) {
  Operator ops[] = {
      Operator("foo", [](RuntimeContext&, EValue**) {}),
      Operator("foo", [](RuntimeContext&, EValue**) {})};
  ArrayRef<Operator> ops_array = ArrayRef<Operator>(ops);
  ET_EXPECT_DEATH({ auto res = register_operators(ops_array); }, "");
}

void make_kernel_key(
    std::vector<std::pair<ScalarType, std::vector<exec_aten::DimOrderType>>>
        tensors,
    char* buf) {
  char* start = buf;
  strncpy(buf, "v0/", 3);
  buf += 3;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto& tensor = tensors[i];
    *buf = (char)tensor.first;
    buf += 1;
    *buf = ';';
    buf += 1;
    memcpy(buf, (char*)tensor.second.data(), tensor.second.size());
    buf += tensor.second.size();
    *buf = (i < (tensors.size() - 1)) ? '|' : 0xff;
    buf += 1;
  }
}

constexpr int BUF_SIZE = 307;

TEST_F(OperatorRegistryTest, KernelKeyEquals) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey long_contiguous = KernelKey(buf_long_contiguous);

  KernelKey long_key_1 = KernelKey(long_contiguous);

  KernelKey long_key_2 = KernelKey(long_contiguous);

  EXPECT_EQ(long_key_1, long_key_2);

  char buf_float_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Float, {0, 1, 2, 3}}}, buf_float_contiguous);
  KernelKey float_key = KernelKey(buf_float_contiguous);

  EXPECT_NE(long_key_1, float_key);

  char buf_channel_first[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 3, 1, 2}}}, buf_channel_first);
  KernelKey long_key_3 = KernelKey(buf_channel_first);

  EXPECT_NE(long_key_1, long_key_3);
}

TEST_F(OperatorRegistryTest, RegisterOpWithKernelKey) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey long_contiguous = KernelKey(buf_long_contiguous);

  KernelKey key = KernelKey(long_contiguous);
  Operator op =
      Operator("test::boo", key, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_operators({op});
  EXPECT_EQ(s1, torch::executor::Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = ArrayRef<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key = ArrayRef<TensorMeta>(meta, 1);
  EXPECT_TRUE(hasOpsFn("test::boo", user_kernel_key));
  // no fallback kernel is registered
  EXPECT_FALSE(hasOpsFn("test::boo", {}));
  OpFunction func = getOpsFn("test::boo", user_kernel_key);

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  RuntimeContext context{};
  func(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(OperatorRegistryTest, RegisterKernels) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 =
      Kernel("test::boo", key, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({kernel_1});
  EXPECT_EQ(s1, torch::executor::Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = ArrayRef<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key = ArrayRef<TensorMeta>(meta, 1);
  EXPECT_TRUE(hasOpsFn("test::boo", user_kernel_key));
  // no fallback kernel is registered
  EXPECT_FALSE(hasOpsFn("test::boo", {}));
  OpFunction func = getOpsFn("test::boo", user_kernel_key);

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  RuntimeContext context{};
  func(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(OperatorRegistryTest, RegisterTwoKernels) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key_1 = KernelKey(buf_long_contiguous);

  char buf_float_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Float, {0, 1, 2, 3}}}, buf_float_contiguous);
  KernelKey key_2 = KernelKey(buf_float_contiguous);
  Kernel kernel_1 =
      Kernel("test::boo", key_1, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Kernel kernel_2 =
      Kernel("test::boo", key_2, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(50);
      });
  Kernel kernels[] = {kernel_1, kernel_2};
  auto s1 = register_kernels(kernels);
  // has both kernels
  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = ArrayRef<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key_1 = ArrayRef<TensorMeta>(meta, 1);

  TensorMeta meta_2[] = {TensorMeta(ScalarType::Float, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key_2 = ArrayRef<TensorMeta>(meta_2, 1);

  EXPECT_TRUE(hasOpsFn("test::boo", user_kernel_key_1));
  EXPECT_TRUE(hasOpsFn("test::boo", user_kernel_key_2));

  // no fallback kernel is registered
  EXPECT_FALSE(hasOpsFn("test::boo", {}));

  EValue values[1];
  values[0] = Scalar(0);
  EValue* evalues[1];
  evalues[0] = &values[0];
  RuntimeContext context{};

  // test kernel_1
  OpFunction func_1 = getOpsFn("test::boo", user_kernel_key_1);
  func_1(context, evalues);

  auto val_1 = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val_1, 100);

  // test kernel_2
  values[0] = Scalar(0);
  OpFunction func_2 = getOpsFn("test::boo", user_kernel_key_2);
  func_2(context, evalues);

  auto val_2 = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val_2, 50);
}

TEST_F(OperatorRegistryTest, DoubleRegisterKernelsDies) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 =
      Kernel("test::boo", key, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Kernel kernel_2 =
      Kernel("test::boo", key, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(50);
      });
  Kernel kernels[] = {kernel_1, kernel_2};
  // clang-tidy off
  ET_EXPECT_DEATH({ auto s1 = register_kernels(kernels); }, "");
  // clang-tidy on
}

TEST_F(OperatorRegistryTest, ExecutorChecksKernel) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 =
      Kernel("test::boo", key, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({kernel_1});
  EXPECT_EQ(s1, torch::executor::Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = ArrayRef<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key_1 = ArrayRef<TensorMeta>(meta, 1);
  EXPECT_TRUE(hasOpsFn("test::boo", user_kernel_key_1));

  Tensor::DimOrderType dims_channel_first[] = {0, 3, 1, 2};
  auto dim_order_type_channel_first =
      ArrayRef<Tensor::DimOrderType>(dims_channel_first, 4);
  TensorMeta meta_channel_first[] = {
      TensorMeta(ScalarType::Long, dim_order_type_channel_first)};
  ArrayRef<TensorMeta> user_kernel_key_2 =
      ArrayRef<TensorMeta>(meta_channel_first, 1);
  EXPECT_FALSE(hasOpsFn("test::boo", user_kernel_key_2));

  TensorMeta meta_float[] = {TensorMeta(ScalarType::Float, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key_3 = ArrayRef<TensorMeta>(meta_float, 1);
  EXPECT_FALSE(hasOpsFn("test::boo", ArrayRef<TensorMeta>(user_kernel_key_3)));
}

TEST_F(OperatorRegistryTest, ExecutorUsesKernel) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 =
      Kernel("test::boo", key, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({kernel_1});
  EXPECT_EQ(s1, torch::executor::Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = ArrayRef<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  ArrayRef<TensorMeta> user_kernel_key_1 = ArrayRef<TensorMeta>(meta, 1);
  EXPECT_TRUE(hasOpsFn("test::boo", ArrayRef<TensorMeta>(meta)));

  OpFunction func = getOpsFn("test::boo", ArrayRef<TensorMeta>(meta));

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  RuntimeContext context{};
  func(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(OperatorRegistryTest, ExecutorUsesFallbackKernel) {
  Kernel kernel_1 = Kernel(
      "test::boo", KernelKey{}, [](RuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({kernel_1});
  EXPECT_EQ(s1, torch::executor::Error::Ok);

  EXPECT_TRUE(hasOpsFn("test::boo"));
  EXPECT_TRUE(hasOpsFn("test::boo", ArrayRef<TensorMeta>()));

  OpFunction func = getOpsFn("test::boo", ArrayRef<TensorMeta>());

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  RuntimeContext context{};
  func(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

} // namespace executor
} // namespace torch
