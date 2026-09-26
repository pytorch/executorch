/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/sampler/logit_processor.h>

#include <limits>
#include <memory>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <gtest/gtest.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::extension::llm::LogitProcessor;
using ::executorch::runtime::Error;
using ::executorch::runtime::testing::TensorFactory;

namespace {

// Shared by the test processors below to advance to the last sequence
// position (rank-3 case), per the LogitProcessor shape contract.
inline float* float_data_at_last_position(Tensor logits) {
  auto* data = logits.mutable_data_ptr<float>();
  if (logits.dim() == 3) {
    data += (logits.size(1) - 1) * logits.size(logits.dim() - 1);
  }
  return data;
}

// Adds a fixed bias to every logit slot in the last position. Records how
// many times it was invoked so tests can verify chain ordering.
class AddBiasProcessor : public LogitProcessor {
 public:
  explicit AddBiasProcessor(float bias) : bias_(bias) {}

  Error process(Tensor logits) override {
    ++call_count_;
    if (logits.scalar_type() != ScalarType::Float) {
      return Error::InvalidArgument;
    }
    auto* data = float_data_at_last_position(logits);
    const auto vocab_size = logits.size(logits.dim() - 1);
    for (ssize_t i = 0; i < vocab_size; ++i) {
      data[i] += bias_;
    }
    return Error::Ok;
  }

  int call_count() const {
    return call_count_;
  }

 private:
  float bias_;
  int call_count_ = 0;
};

class MultiplyProcessor : public LogitProcessor {
 public:
  explicit MultiplyProcessor(float factor) : factor_(factor) {}

  Error process(Tensor logits) override {
    if (logits.scalar_type() != ScalarType::Float) {
      return Error::InvalidArgument;
    }
    auto* data = float_data_at_last_position(logits);
    const auto vocab_size = logits.size(logits.dim() - 1);
    for (ssize_t i = 0; i < vocab_size; ++i) {
      data[i] *= factor_;
    }
    return Error::Ok;
  }

 private:
  float factor_;
};

class MaskTokenProcessor : public LogitProcessor {
 public:
  explicit MaskTokenProcessor(int32_t banned_token)
      : banned_token_(banned_token) {}

  Error process(Tensor logits) override {
    if (logits.scalar_type() != ScalarType::Float) {
      return Error::InvalidArgument;
    }
    auto* data = float_data_at_last_position(logits);
    const auto vocab_size = logits.size(logits.dim() - 1);
    if (banned_token_ >= 0 && banned_token_ < vocab_size) {
      data[banned_token_] = -std::numeric_limits<float>::infinity();
    }
    return Error::Ok;
  }

 private:
  int32_t banned_token_;
};

} // namespace

// A single processor mutates the rank-2 logits tensor in place.
TEST(LogitProcessorTest, SingleProcessorMutatesLogits) {
  TensorFactory<ScalarType::Float> tf;
  auto logits = tf.make({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});

  AddBiasProcessor bias{10.0f};
  ASSERT_EQ(bias.process(logits), Error::Ok);

  auto* data = logits.mutable_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 11.0f);
  EXPECT_FLOAT_EQ(data[3], 14.0f);
  EXPECT_EQ(bias.call_count(), 1);
}

// Multiply(×2) then Add(+1) gives (x*2)+1, which differs from
// Add(+1) then Multiply(×2) = (x+1)*2. Non-commutative operations
// verify that processors run in registration order.
TEST(LogitProcessorTest, ProcessorChainAppliesInOrder) {
  TensorFactory<ScalarType::Float> tf;
  auto logits = tf.make({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::shared_ptr<LogitProcessor>> chain;
  chain.push_back(std::make_shared<MultiplyProcessor>(2.0f));
  chain.push_back(std::make_shared<AddBiasProcessor>(1.0f));

  for (auto& p : chain) {
    ASSERT_EQ(p->process(logits), Error::Ok);
  }

  // (x*2)+1, NOT (x+1)*2
  auto* data = logits.mutable_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 3.0f);
  EXPECT_FLOAT_EQ(data[1], 5.0f);
  EXPECT_FLOAT_EQ(data[2], 7.0f);
  EXPECT_FLOAT_EQ(data[3], 9.0f);
}

// A masking processor sets a specific token's logit to -inf. This is the
// pattern grammar processors will follow.
TEST(LogitProcessorTest, MaskTokenDrivesArgmaxAway) {
  TensorFactory<ScalarType::Float> tf;
  auto logits = tf.make({1, 4}, {0.1f, 0.2f, 0.99f, 0.4f}); // argmax = 2

  MaskTokenProcessor mask{/*banned_token=*/2};
  ASSERT_EQ(mask.process(logits), Error::Ok);

  auto* data = logits.mutable_data_ptr<float>();
  EXPECT_EQ(data[2], -std::numeric_limits<float>::infinity());
  // Other slots untouched.
  EXPECT_FLOAT_EQ(data[0], 0.1f);
  EXPECT_FLOAT_EQ(data[1], 0.2f);
  EXPECT_FLOAT_EQ(data[3], 0.4f);
}

// Out-of-range banned token id is silently ignored — defensive behavior
// for grammar processors that may pass an EOS-or-similar id that the
// underlying vocab doesn't actually contain.
TEST(LogitProcessorTest, MaskTokenOutOfRangeIsNoOp) {
  TensorFactory<ScalarType::Float> tf;
  auto logits = tf.make({1, 3}, {1.0f, 2.0f, 3.0f});
  const std::vector<float> snapshot = {1.0f, 2.0f, 3.0f};

  MaskTokenProcessor mask_over{/*banned_token=*/99};
  ASSERT_EQ(mask_over.process(logits), Error::Ok);
  auto* data = logits.mutable_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], snapshot[0]);
  EXPECT_FLOAT_EQ(data[1], snapshot[1]);
  EXPECT_FLOAT_EQ(data[2], snapshot[2]);

  MaskTokenProcessor mask_neg{/*banned_token=*/-1};
  ASSERT_EQ(mask_neg.process(logits), Error::Ok);
  EXPECT_FLOAT_EQ(data[0], snapshot[0]);
  EXPECT_FLOAT_EQ(data[1], snapshot[1]);
  EXPECT_FLOAT_EQ(data[2], snapshot[2]);
}

// On a rank-3 [batch, seq, vocab] tensor, the processor must only mutate
// the LAST sequence position. Earlier positions stay untouched.
TEST(LogitProcessorTest, RespectsLastPositionOf3DTensor) {
  TensorFactory<ScalarType::Float> tf;
  // Shape [batch=1, seq=2, vocab=4]. First-position values are sentinels.
  auto logits = tf.make(
      {1, 2, 4},
      {
          99.0f,
          99.0f,
          99.0f,
          99.0f, // first position — must NOT change
          1.0f,
          2.0f,
          3.0f,
          4.0f, // last position — gets +10 from bias
      });

  AddBiasProcessor bias{10.0f};
  ASSERT_EQ(bias.process(logits), Error::Ok);

  auto* data = logits.mutable_data_ptr<float>();
  // First position untouched.
  EXPECT_FLOAT_EQ(data[0], 99.0f);
  EXPECT_FLOAT_EQ(data[3], 99.0f);
  // Last position got +10.
  EXPECT_FLOAT_EQ(data[4], 11.0f);
  EXPECT_FLOAT_EQ(data[7], 14.0f);
}

// Each processor declares its own dtype expectations. The test processors
// here only support Float; passing a Half tensor must surface
// InvalidArgument rather than silently corrupt memory.
TEST(LogitProcessorTest, ProcessorRejectsUnsupportedDtype) {
  TensorFactory<ScalarType::Half> tf;
  auto logits = tf.make({1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});

  AddBiasProcessor bias{1.0f};
  EXPECT_EQ(bias.process(logits), Error::InvalidArgument);
}
