/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

namespace executorch::backends::qnn {
int ExportedOutputPosition(const std::string& name);

namespace {

std::shared_ptr<TensorWrapper> output(
    const std::string& name,
    float* data = nullptr) {
  const uint32_t dims[] = {2};
  auto tensor = CreateTensorWrapper(
      name,
      QNN_TENSOR_TYPE_APP_READ,
      QNN_DATATYPE_FLOAT_32,
      std::make_unique<UndefinedQuantizeParamsWrapper>(),
      1,
      dims,
      nullptr);
  if (data != nullptr) {
    tensor->FillDataBuffer(data);
  }
  return tensor;
}

class QnnOutputOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto backend = qnn_delegate::CreateQnnExecuTorchBackendOptions(
        builder_, qnn_delegate::QnnExecuTorchBackendType::kHtpBackend);
    qnn_delegate::QnnExecuTorchOptionsBuilder options(builder_);
    options.add_backend_options(backend);
    options.add_is_from_context_binary(from_context_binary());
    builder_.Finish(options.Finish());
    manager_ = std::make_unique<QnnManager>(
        qnn_delegate::GetQnnExecuTorchOptions(builder_.GetBufferPointer()),
        QnnExecuTorchContextBinary{nullptr, 0});
  }

  virtual bool from_context_binary() const {
    return false;
  }

  std::vector<std::shared_ptr<TensorWrapper>> allocate(
      std::vector<std::shared_ptr<TensorWrapper>> tensors) {
    std::vector<std::shared_ptr<TensorWrapper>> inputs;
    EXPECT_EQ(
        manager_->AllocateTensor("forward", inputs, tensors),
        executorch::runtime::Error::Ok);
    return manager_->GetGraphOutputs("forward");
  }

  flatbuffers::FlatBufferBuilder builder_;
  std::unique_ptr<QnnManager> manager_;
};

TEST_F(QnnOutputOrderTest, ReorderedSameShapedOutputsBindDistinctValues) {
  std::array<float, 2> key = {11, 12};
  std::array<float, 2> value = {21, 22};
  auto k = output("output_0_key@0", key.data());
  auto v = output("output_1_value@0", value.data());
  auto tensors = allocate({v, k});
  ASSERT_EQ(tensors.size(), 2);
  EXPECT_EQ(tensors[0], k);
  EXPECT_EQ(tensors[1], v);
  EXPECT_EQ(tensors[0]->GetStaticTensorData(), key.data());
  EXPECT_EQ(tensors[1]->GetStaticTensorData(), value.data());
  EXPECT_FLOAT_EQ(
      static_cast<const float*>(tensors[0]->GetStaticTensorData())[0], 11);
  EXPECT_FLOAT_EQ(
      static_cast<const float*>(tensors[1]->GetStaticTensorData())[0], 21);
}

TEST_F(QnnOutputOrderTest, NumericOrderAndEqualPositionsAreStable) {
  auto zero = output("output_0_zero@0");
  auto two = output("output_2_two@0");
  auto duplicate = output("output_2_duplicate@0");
  auto ten = output("output_10_ten@0");
  EXPECT_EQ(
      allocate({ten, two, duplicate, zero}),
      (std::vector<std::shared_ptr<TensorWrapper>>{zero, two, duplicate, ten}));
}

TEST_F(
    QnnOutputOrderTest,
    LegacyMutableAndDumpOutputsKeepRelativeOrderAndData) {
  float cache[2] = {4, 5};
  auto legacy = output("output_old@0");
  auto mutated = output("output_mutbuf_0_cache@0", cache);
  auto dump = output("intermediate@0");
  EXPECT_EQ(
      allocate({legacy, mutated, dump}),
      (std::vector<std::shared_ptr<TensorWrapper>>{legacy, mutated, dump}));
  auto numbered = output("output_0_new@0");
  EXPECT_EQ(
      allocate({legacy, numbered, mutated, dump}),
      (std::vector<std::shared_ptr<TensorWrapper>>{
          numbered, legacy, mutated, dump}));
  EXPECT_EQ(mutated->GetStaticTensorData(), cache);
  EXPECT_EQ(mutated->GetName(), "output_mutbuf_0_cache@0");
}

class QnnContextOutputOrderTest : public QnnOutputOrderTest {
 protected:
  bool from_context_binary() const override {
    return true;
  }
};

TEST_F(QnnContextOutputOrderTest, ContextBinaryOrderIsNotChanged) {
  auto one = output("output_1_first@0");
  auto zero = output("output_0_second@0");
  EXPECT_EQ(
      allocate({one, zero}),
      (std::vector<std::shared_ptr<TensorWrapper>>{one, zero}));
}

TEST(
    QnnOutputPositionTest,
    RejectsLegacyMalformedAndOverflowNamesWithoutThrowing) {
  for (const std::string name :
       {"output_old@0",
        "output_mutbuf_0_cache@0",
        "intermediate@0",
        "output_",
        "output_2",
        "output_2_",
        "output_2old@0",
        "output_-1_old@0",
        "output_+1_old@0",
        "output_ 1_old@0",
        "output_2147483648_old@0",
        "output_999999999999999999999_old@0"}) {
    EXPECT_NO_THROW(EXPECT_EQ(ExportedOutputPosition(name), -1)) << name;
  }
  EXPECT_EQ(ExportedOutputPosition("output_0_add@0"), 0);
  EXPECT_EQ(ExportedOutputPosition("output_10_getitem@2"), 10);
  EXPECT_EQ(
      ExportedOutputPosition("output_2147483647_last@0"),
      std::numeric_limits<int>::max());
}

} // namespace
} // namespace executorch::backends::qnn
