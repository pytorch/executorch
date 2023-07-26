#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor&
_full_out(const IntArrayRef sizes, const Scalar& fill_value, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::full_outf(context, sizes, fill_value, out);
}

TEST(OpFullOutTest, DtypeTest_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullOutTest, DtypeTest_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  ::std::vector<int64_t> size_vec = {2, 2};
  exec_aten::ArrayRef<int64_t> size =
      exec_aten::ArrayRef<int64_t>(size_vec.data(), size_vec.size());
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _full_out(size, fill_value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

template <ScalarType DTYPE>
void test_ones_out(std::vector<int32_t>&& size_int32_t) {
  TensorFactory<DTYPE> tf;
  std::vector<int64_t> size_int64_t(size_int32_t.begin(), size_int32_t.end());
  auto aref = IntArrayRef(size_int64_t.data(), size_int64_t.size());

  // Before: `out` consists of 0s.
  Tensor out = tf.zeros(size_int32_t);

  // After: `out` consists of 1s.
  _full_out(aref, 1, out);

  EXPECT_TENSOR_EQ(out, tf.ones(size_int32_t));
}

#define GENERATE_TEST(_, DTYPE)                  \
  TEST(OpFullOutTest, DTYPE##Tensors) {          \
    test_ones_out<ScalarType::DTYPE>({});        \
    test_ones_out<ScalarType::DTYPE>({1});       \
    test_ones_out<ScalarType::DTYPE>({1, 1, 1}); \
    test_ones_out<ScalarType::DTYPE>({2, 0, 4}); \
    test_ones_out<ScalarType::DTYPE>({2, 3, 4}); \
  }

ET_FORALL_REAL_TYPES(GENERATE_TEST)
