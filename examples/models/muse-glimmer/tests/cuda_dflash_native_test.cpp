/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda_runtime.h>
#include <executorch/examples/models/muse-glimmer/runtime/cuda_dflash_executor.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace llm = executorch::extension::llm;
namespace b = llm::batching;
namespace cb = executorch::backends::cuda::batching;
namespace c = llm::cache;
namespace rt = executorch::runtime;
namespace aten = executorch::aten;
using DFlash = llm::MuseGlimmerCudaDFlashExecutor;
using Tokens = std::vector<b::Token>;
using executorch::extension::Module;
using executorch::extension::TensorPtr;

namespace executorch::extension::llm {
// Tiny artifacts cannot pass Muse's special-token vocabulary check. Keep the
// production factory unchanged and construct only the validated tiny geometry.
struct CudaDFlashTestPeer {
  static std::unique_ptr<DFlash> create(
      std::unique_ptr<Module> module,
      int proposals) {
    auto target = cb::CudaExecutor::create(
        std::move(module), 4, 96, static_cast<int>(aten::ScalarType::BFloat16));
    EXPECT_EQ(target.error(), rt::Error::Ok);
    if (!target.ok()) {
      return nullptr;
    }
    EXPECT_EQ(target.get()->vocab_size(), 256);
    EXPECT_EQ(target.get()->preferred_batch_tokens(), 32);
    auto& program = target.get()->module();
    auto mask = detail::read_int_method(program, "get_mask_token_id");
    auto layers = detail::read_int_method(program, "get_draft_n_caches");
    auto heads = detail::read_int_tensor_method(program, "get_draft_kv_heads");
    auto dims = detail::read_int_tensor_method(program, "get_draft_head_dims");
    auto windows = detail::read_int_tensor_method(program, "get_draft_windows");
    if (!mask.ok() || !*mask || !layers.ok() || !*layers || !heads.ok() ||
        !dims.ok() || !windows.ok()) {
      return nullptr;
    }
    EXPECT_EQ(**layers, heads->size());
    c::CacheGeometry geometry;
    geometry.dflash_block_size = 16;
    for (std::size_t i = 0; i < heads->size(); ++i) {
      geometry.layers.push_back(
          {{(*windows)[i] ? c::LayerPolicy::Kind::Ring
                          : c::LayerPolicy::Kind::Flat,
            (*windows)[i]},
           (*heads)[i],
           (*dims)[i]});
    }
    c::CacheConfig config{};
    config.capacity = 512;
    config.max_write = 96;
    config.kv_dtype = static_cast<int>(aten::ScalarType::BFloat16);
    auto draft = c::CacheFactory::global().build(
        executorch::backends::cuda::kCudaBackendId,
        c::kind::kBatchedCell,
        geometry,
        config);
    EXPECT_EQ(draft.error(), rt::Error::Ok);
    if (!draft.ok()) {
      return nullptr;
    }
    return std::unique_ptr<DFlash>(new DFlash(
        std::move(target.get()),
        std::move(draft.get()),
        16,
        proposals,
        static_cast<b::Token>(**mask),
        4,
        32,
        4,
        96,
        128));
  }
  static int frontier(DFlash& executor, b::SessionId sid) {
    return executor.sessions_.at(sid).frontier;
  }
  static int draft_position(DFlash& executor, b::SessionId sid) {
    return executor.draft_ctl_->pos(executor.sessions_.at(sid).draft_id);
  }
  static const std::vector<std::uint8_t>& hidden(
      DFlash& executor,
      b::SessionId sid) {
    return executor.sessions_.at(sid).hidden;
  }
  static std::optional<b::Token> pending(DFlash& executor, b::SessionId sid) {
    return executor.sessions_.at(sid).pending;
  }
  static void check(DFlash& executor, b::SessionId sid, int expected) {
    const auto& state = executor.sessions_.at(sid);
    EXPECT_FALSE(state.poisoned);
    EXPECT_EQ(state.frontier, expected);
    EXPECT_EQ(executor.target_->cache_control().pos(state.target_id), expected);
    const int draft = executor.draft_ctl_->pos(state.draft_id);
    ASSERT_GE(draft, 0);
    ASSERT_LE(draft, expected);
    EXPECT_EQ(state.hidden.size(), (expected - draft) * 256u);
  }
  static bool rewind(DFlash& executor, b::SessionId sid, int position) {
    return executor.rewind(executor.sessions_.at(sid), position);
  }
  static bool poisoned(DFlash& executor, b::SessionId sid) {
    return executor.sessions_.at(sid).poisoned;
  }
};
} // namespace executorch::extension::llm
using Peer = llm::CudaDFlashTestPeer;

namespace {
// Independent greedy fixture: prediction at absolute position p is 20+p.
// Hidden row p has finite BF16 values with bits 0x3f00 + 8*(p%16) + column%8.
// Real tiny CUDA target/draft methods still run and commit their actual caches;
// deterministic replacement outputs isolate adapter acceptance/lifetime logic
// from the tiny model's known numerical parity caveat.
b::Token prediction(int position) {
  return 20 + position;
}
std::vector<std::uint8_t> hidden_rows(int start, int rows) {
  std::vector<std::uint8_t> bytes(rows * 256);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < 128; ++column) {
      const std::uint16_t bits = 0x3f00 + 8 * ((start + row) % 16) + column % 8;
      std::memcpy(bytes.data() + row * 256 + column * 2, &bits, 2);
    }
  }
  return bytes;
}
b::Input
input(b::SessionId sid, int position, Tokens tokens, bool output = true) {
  const auto size = tokens.size();
  return {
      sid,
      output,
      0,
      size,
      std::make_shared<const Tokens>(std::move(tokens)),
      position};
}
Tokens predictions(int start, int count) {
  Tokens result(count);
  std::iota(result.begin(), result.end(), prediction(start));
  return result;
}

class FixtureModule final : public Module {
 public:
  FixtureModule(const std::string& dir, bool deterministic)
      : Module(
            dir + "/model.pte",
            std::vector<std::string>{dir + "/aoti_cuda_blob.ptd"}),
        deterministic_(deterministic) {}
  ~FixtureModule() override {
    if (taps_) {
      cudaFree(taps_);
    }
  }
  std::array<int, 4> rejected{15, 15, 15, 15};
  bool fail_target_sampler = false;
  rt::Result<std::vector<rt::EValue>> execute(
      const std::string& name,
      const std::vector<rt::EValue>& inputs) override {
    auto actual = Module::execute(name, inputs);
    if (!actual.ok()) {
      return actual.error();
    }
    const auto stream = executorch::extension::cuda::getCallerStream().value_or(
        cudaStreamPerThread);
    if (name == "sample_argmax") {
      if (taps_ && cudaMemsetAsync(taps_, 0, 32 * 256, stream) != cudaSuccess) {
        return rt::Error::Internal;
      }
      if (target_output_ && fail_target_sampler) {
        fail_target_sampler = false;
        return rt::Error::Internal;
      }
      return actual;
    }
    if (name != "draft_forward" && name.rfind("forward_", 0) != 0) {
      return actual;
    }
    target_output_ = name != "draft_forward";
    if (!deterministic_) {
      return actual;
    }
    const auto& positions = inputs[target_output_ ? 1 : 2].toTensor();
    const auto* p = positions.const_data_ptr<std::int64_t>();
    const auto& tokens = inputs[0].toTensor();
    const int width = tokens.size(1);
    const auto& selector = inputs[target_output_ ? 2 : 0].toTensor();
    const int rows = target_output_ ? selector.numel() : width;
    std::vector<float> logits(rows * 256, -100.0f);
    const int context = target_output_ ? 0 : inputs[1].toTensor().size(1);
    for (int row = 0; row < rows; ++row) {
      const int position = target_output_
          ? p[selector.const_data_ptr<std::int64_t>()[row]]
          : p[context + row] - 1;
      int token = static_cast<int>(prediction(position));
      if (!target_output_ && row % 16 == rejected[row / 16] + 1) {
        token += 100;
      }
      logits[row * 256 + token] = 100.0f;
    }
    logits_ = device_tensor(
        logits.data(),
        logits.size() * sizeof(float),
        {1, rows, 256},
        aten::ScalarType::Float);
    if (!logits_) {
      return rt::Error::Internal;
    }
    if (!target_output_) {
      return std::vector<rt::EValue>{logits_};
    }
    if (!taps_ && cudaMalloc(&taps_, 32 * 256) != cudaSuccess) {
      return rt::Error::MemoryAllocationFailed;
    }
    std::vector<std::uint8_t> bytes(width * 256, 255);
    for (int row = 0; row < width; ++row) {
      auto value = hidden_rows(p[row], 1);
      std::copy(value.begin(), value.end(), bytes.begin() + row * 256);
    }
    if (cudaMemcpyAsync(
            taps_,
            bytes.data(),
            bytes.size(),
            cudaMemcpyHostToDevice,
            stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return rt::Error::Internal;
    }
    tap_view_ = executorch::extension::from_blob(
        taps_,
        {1, width, 128},
        aten::ScalarType::BFloat16,
        aten::Device(aten::DeviceType::CUDA, 0));
    return std::vector<rt::EValue>{logits_, tap_view_};
  }

 private:
  static TensorPtr device_tensor(
      const void* data,
      std::size_t bytes,
      std::vector<aten::SizesType> sizes,
      aten::ScalarType type) {
    void* device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess) {
      return nullptr;
    }
    const auto stream = executorch::extension::cuda::getCallerStream().value_or(
        cudaStreamPerThread);
    if (cudaMemcpyAsync(device, data, bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaFree(device);
      return nullptr;
    }
    return executorch::extension::from_blob(
        device,
        std::move(sizes),
        type,
        aten::Device(aten::DeviceType::CUDA, 0),
        [](void* pointer) { cudaFree(pointer); });
  }
  bool deterministic_, target_output_ = false;
  void* taps_ = nullptr;
  TensorPtr logits_, tap_view_;
};

class CudaDFlashNative : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* dir = std::getenv("ET_MUSE_DFLASH_TINY_DIR");
    if (!dir || !*dir) {
      GTEST_SKIP() << "Set ET_MUSE_DFLASH_TINY_DIR to the output of "
                      "tests/export_cuda_dflash_tiny.py";
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      GTEST_SKIP() << "CUDA device required";
    }
    rt::runtime_init();
    dir_ = dir;
  }
  void create(bool deterministic = true, int proposals = 15) {
    auto module = std::make_unique<FixtureModule>(dir_, deterministic);
    fixture_ = module.get();
    executor_ = Peer::create(std::move(module), proposals);
    ASSERT_TRUE(executor_);
    ASSERT_TRUE(executor_->initialize());
  }
  b::SessionId open() {
    auto sid = executor_->open_session();
    EXPECT_TRUE(sid);
    if (!sid) {
      return -1;
    }
    executor_->set_sampling(*sid, {}, 0);
    return *sid;
  }
  void seed(b::SessionId sid, int rows) {
    b::BatchOutput out;
    ASSERT_TRUE(
        executor_->execute({{input(sid, 0, Tokens(rows, 1), false)}}, out));
    Peer::check(*executor_, sid, rows);
    EXPECT_FALSE(Peer::pending(*executor_, sid));
    EXPECT_EQ(Peer::hidden(*executor_, sid), hidden_rows(0, rows));
  }
  std::string dir_;
  FixtureModule* fixture_ = nullptr;
  std::unique_ptr<DFlash> executor_;
};

TEST_F(CudaDFlashNative, EveryRejectionPositionAndFullBonus) {
  create();
  ASSERT_TRUE(executor_);
  for (int rejected = 0; rejected <= 15; ++rejected) {
    SCOPED_TRACE(rejected);
    const auto sid = open();
    seed(sid, 3);
    fixture_->rejected.fill(rejected);
    b::BatchOutput preamble;
    ASSERT_TRUE(executor_->execute({{input(sid, 3, {1})}}, preamble));
    ASSERT_TRUE(preamble.outputs[0]);
    EXPECT_EQ(preamble.outputs[0]->tokens, predictions(3, rejected + 2));
    const int start = 5 + rejected;
    Peer::check(*executor_, sid, start);
    EXPECT_EQ(Peer::hidden(*executor_, sid), hidden_rows(4, rejected + 1));
    const auto before = executor_->stats();
    b::BatchOutput decode;
    ASSERT_TRUE(executor_->execute(
        {{input(sid, start, {prediction(start - 1)})}}, decode));
    ASSERT_TRUE(decode.outputs[0]);
    EXPECT_EQ(decode.outputs[0]->tokens, predictions(start, rejected + 1));
    Peer::check(*executor_, sid, start + rejected + 1);
    EXPECT_EQ(Peer::draft_position(*executor_, sid), start);
    EXPECT_EQ(Peer::hidden(*executor_, sid), hidden_rows(start, rejected + 1));
    EXPECT_EQ(Peer::pending(*executor_, sid), prediction(start + rejected));
    const auto after = executor_->stats();
    EXPECT_EQ(after.target_preamble_calls, before.target_preamble_calls);
    EXPECT_EQ(after.target_verify_calls - before.target_verify_calls, 1);
    EXPECT_EQ(after.pending_decode_owners - before.pending_decode_owners, 1);
    EXPECT_EQ(after.cycles - before.cycles, 1);
    EXPECT_EQ(after.proposed - before.proposed, 15);
    EXPECT_EQ(after.accepted - before.accepted, rejected);
    EXPECT_EQ(after.bonuses - before.bonuses, rejected == 15);
    EXPECT_EQ(after.rejected - before.rejected, rejected != 15);
    EXPECT_EQ(after.hidden_rows_d2d - before.hidden_rows_d2d, 16);
    EXPECT_EQ(after.hidden_rows_d2h - before.hidden_rows_d2h, rejected + 1);
    EXPECT_EQ(
        after.hidden_bytes_d2h - before.hidden_bytes_d2h,
        (rejected + 1) * 256u);
    executor_->close_session(sid);
  }
}

TEST_F(CudaDFlashNative, FourUnequalOwnersSplitCallbacksAndMixedFallback) {
  create();
  ASSERT_TRUE(executor_);
  std::array<b::SessionId, 4> owners;
  std::array<int, 4> starts;
  const std::array<int, 4> rejected{0, 2, 7, 15};
  for (int owner = 0; owner < 4; ++owner) {
    owners[owner] = open();
    seed(owners[owner], owner + 2);
    fixture_->rejected.fill(rejected[owner]);
    b::BatchOutput out;
    ASSERT_TRUE(
        executor_->execute({{input(owners[owner], owner + 2, {1})}}, out));
    ASSERT_TRUE(out.outputs[0]);
    EXPECT_EQ(
        out.outputs[0]->tokens, predictions(owner + 2, rejected[owner] + 2));
    starts[owner] = owner + 4 + rejected[owner];
  }
  fixture_->rejected = rejected;
  const auto before = executor_->stats();
  b::BatchInput batch;
  for (int owner = 0; owner < 4; ++owner) {
    batch.inputs.push_back(
        input(owners[owner], starts[owner], {prediction(starts[owner] - 1)}));
  }
  b::BatchOutput out;
  ASSERT_TRUE(executor_->execute(batch, out));
  for (int owner = 0; owner < 4; ++owner) {
    ASSERT_TRUE(out.outputs[owner]);
    EXPECT_EQ(out.outputs[owner]->sid, owners[owner]);
    EXPECT_EQ(
        out.outputs[owner]->tokens,
        predictions(starts[owner], rejected[owner] + 1));
    Peer::check(*executor_, owners[owner], starts[owner] + rejected[owner] + 1);
    EXPECT_EQ(
        Peer::hidden(*executor_, owners[owner]),
        hidden_rows(starts[owner], rejected[owner] + 1));
  }
  const auto after = executor_->stats();
  EXPECT_EQ(after.target_preamble_calls, before.target_preamble_calls);
  EXPECT_EQ(after.target_verify_calls - before.target_verify_calls, 1);
  EXPECT_EQ(after.target_verify_forwards - before.target_verify_forwards, 2);
  EXPECT_EQ(after.hidden_rows_d2d - before.hidden_rows_d2d, 64);
  EXPECT_EQ(after.hidden_rows_d2h - before.hidden_rows_d2h, 28);
  // A decode owner sharing a batch with new prompt rows retains the preamble.
  const int position0 = starts[0] + 1;
  const int position1 = starts[1] + 3;
  const auto idle_hidden = Peer::hidden(*executor_, owners[3]);
  const int idle_position = Peer::frontier(*executor_, owners[3]);
  fixture_->rejected.fill(0);
  ASSERT_TRUE(executor_->execute(
      {{input(owners[0], position0, {prediction(position0 - 1)}),
        input(owners[1], position1, {1, 2})}},
      out));
  EXPECT_EQ(
      executor_->stats().target_preamble_calls - after.target_preamble_calls,
      1);
  EXPECT_EQ(
      executor_->stats().pending_decode_batches, after.pending_decode_batches);
  EXPECT_EQ(out.outputs[0]->tokens, predictions(position0, 2));
  EXPECT_EQ(out.outputs[1]->tokens, predictions(position1 + 1, 2));
  EXPECT_EQ(Peer::hidden(*executor_, owners[3]), idle_hidden);
  Peer::check(*executor_, owners[3], idle_position);
}

TEST_F(CudaDFlashNative, PartialVerificationWidthExcludesStaticPadding) {
  create(true, 5);
  ASSERT_TRUE(executor_);
  for (int rejected : {2, 5}) {
    const auto sid = open();
    seed(sid, 3);
    fixture_->rejected.fill(rejected);
    b::BatchOutput out;
    ASSERT_TRUE(executor_->execute({{input(sid, 3, {1})}}, out));
    ASSERT_TRUE(out.outputs[0]);
    EXPECT_EQ(out.outputs[0]->tokens, predictions(3, rejected + 2));
    const int start = rejected + 5;
    const auto before = executor_->stats();
    ASSERT_TRUE(executor_->execute(
        {{input(sid, start, {prediction(start - 1)})}}, out));
    ASSERT_TRUE(out.outputs[0]);
    EXPECT_EQ(out.outputs[0]->tokens, predictions(start, rejected + 1));
    EXPECT_EQ(Peer::hidden(*executor_, sid), hidden_rows(start, rejected + 1));
    Peer::check(*executor_, sid, start + rejected + 1);
    const auto after = executor_->stats();
    EXPECT_EQ(after.target_verify_forwards - before.target_verify_forwards, 1);
    EXPECT_EQ(after.hidden_rows_d2d - before.hidden_rows_d2d, 6);
    EXPECT_EQ(after.hidden_rows_d2h - before.hidden_rows_d2h, rejected + 1);
    executor_->close_session(sid);
  }
}

TEST_F(CudaDFlashNative, WrongFeedbackAndNonOutputSlicesKeepPreamble) {
  create();
  ASSERT_TRUE(executor_);
  const auto sid = open();
  seed(sid, 3);
  fixture_->rejected.fill(0);
  b::BatchOutput out;
  ASSERT_TRUE(executor_->execute({{input(sid, 3, {1})}}, out));
  Peer::check(*executor_, sid, 5);
  const auto before = executor_->stats();
  ASSERT_TRUE(executor_->execute({{input(sid, 5, {1})}}, out));
  ASSERT_TRUE(out.outputs[0]);
  EXPECT_EQ(out.outputs[0]->tokens, predictions(5, 2));
  EXPECT_EQ(
      executor_->stats().pending_decode_batches, before.pending_decode_batches);
  EXPECT_EQ(
      executor_->stats().target_preamble_calls - before.target_preamble_calls,
      1);
  ASSERT_TRUE(
      executor_->execute({{input(sid, 7, {prediction(6)}, false)}}, out));
  EXPECT_FALSE(out.outputs[0]);
  EXPECT_FALSE(Peer::pending(*executor_, sid));
  Peer::check(*executor_, sid, 8);
  ASSERT_TRUE(executor_->execute({{input(sid, 8, {1})}}, out));
  EXPECT_EQ(
      executor_->stats().pending_decode_batches, before.pending_decode_batches);
  EXPECT_EQ(
      executor_->stats().target_preamble_calls - before.target_preamble_calls,
      3);
  // Unsupported temperature fails before a target/draft forward.
  executor_->set_sampling(sid, {0.5f, 1.0f, 0}, 0);
  const auto valid = executor_->stats();
  EXPECT_FALSE(executor_->execute({{input(sid, 10, {prediction(9)})}}, out));
  EXPECT_FALSE(out.outputs[0]);
  EXPECT_EQ(
      executor_->stats().target_preamble_calls, valid.target_preamble_calls);
  EXPECT_EQ(executor_->stats().target_verify_calls, valid.target_verify_calls);
  EXPECT_TRUE(Peer::poisoned(*executor_, sid));
}

TEST_F(CudaDFlashNative, FinalWholeBlockFitsExactlyAndPreservesAllPrefillRows) {
  create();
  ASSERT_TRUE(executor_);
  const auto sid = open();
  b::BatchOutput out;
  ASSERT_TRUE(executor_->execute({{input(sid, 0, Tokens(78, 1), false)}}, out));
  Peer::check(*executor_, sid, 78);
  EXPECT_EQ(executor_->stats().hidden_rows_d2h, 78);
  fixture_->rejected.fill(0);
  ASSERT_TRUE(executor_->execute({{input(sid, 78, {1})}}, out));
  EXPECT_EQ(out.outputs[0]->tokens, predictions(78, 2));
  Peer::check(*executor_, sid, 80);
  fixture_->rejected.fill(15);
  const auto before = executor_->stats();
  ASSERT_TRUE(executor_->execute({{input(sid, 80, {prediction(79)})}}, out));
  EXPECT_EQ(out.outputs[0]->tokens, predictions(80, 16));
  EXPECT_EQ(
      executor_->stats().target_preamble_calls, before.target_preamble_calls);
  Peer::check(*executor_, sid, 96);
  EXPECT_EQ(Peer::hidden(*executor_, sid), hidden_rows(80, 16));
}

TEST_F(CudaDFlashNative, CloneBudgetRewindCapacityAndFailureAreClosed) {
  create();
  ASSERT_TRUE(executor_);
  const auto sid = open();
  seed(sid, 3);
  b::BatchOutput out;
  ASSERT_TRUE(executor_->execute({{input(sid, 3, {1})}}, out));
  Peer::check(*executor_, sid, 20);
  auto full = executor_->clone(sid, 20);
  auto clipped = executor_->clone(sid, 7);
  ASSERT_TRUE(full);
  ASSERT_TRUE(clipped);
  Peer::check(*executor_, *full, 20);
  Peer::check(*executor_, *clipped, 7);
  EXPECT_EQ(Peer::hidden(*executor_, *clipped), hidden_rows(4, 3));
  EXPECT_EQ(Peer::pending(*executor_, *full), prediction(19));
  EXPECT_FALSE(Peer::pending(*executor_, *clipped));
  EXPECT_FALSE(executor_->clone(sid, 21));
  executor_->set_sampling(*full, {}, 0);
  executor_->set_sampling(*clipped, {}, 0);
  fixture_->rejected.fill(0);
  const auto before = executor_->stats();
  ASSERT_TRUE(executor_->execute({{input(*full, 20, {prediction(19)})}}, out));
  EXPECT_EQ(out.outputs[0]->tokens, Tokens({prediction(20)}));
  EXPECT_EQ(
      executor_->stats().target_preamble_calls, before.target_preamble_calls);
  Peer::check(*executor_, sid, 20);
  ASSERT_TRUE(Peer::rewind(*executor_, sid, 7));
  EXPECT_FALSE(Peer::pending(*executor_, sid));
  EXPECT_EQ(Peer::hidden(*executor_, sid), hidden_rows(4, 3));
  ASSERT_TRUE(executor_->execute({{input(sid, 7, {prediction(6)})}}, out));
  EXPECT_EQ(
      executor_->stats().target_preamble_calls - before.target_preamble_calls,
      1);
  EXPECT_EQ(out.outputs[0]->tokens, predictions(7, 2));
  // Refill without output to the final trained-block boundary, then decode.
  ASSERT_TRUE(executor_->execute({{input(sid, 9, Tokens(72, 1), false)}}, out));
  Peer::check(*executor_, sid, 81);
  const auto capacity_before = executor_->stats();
  ASSERT_TRUE(executor_->execute({{input(sid, 81, {1})}}, out));
  EXPECT_EQ(out.outputs[0]->tokens, Tokens({prediction(81)}));
  Peer::check(*executor_, sid, 82);
  EXPECT_EQ(
      executor_->stats().target_verify_calls,
      capacity_before.target_verify_calls);
  EXPECT_EQ(
      executor_->stats().pending_decode_batches,
      capacity_before.pending_decode_batches);
  // Fail the target sampler after its forward and snapshot have already run.
  fixture_->fail_target_sampler = true;
  EXPECT_FALSE(executor_->execute({{input(*full, 21, {prediction(20)})}}, out));
  EXPECT_FALSE(out.outputs[0]);
  EXPECT_EQ(executor_->stats().failed_batches, 1);
  for (const auto owner : {sid, *full, *clipped}) {
    EXPECT_TRUE(Peer::poisoned(*executor_, owner));
    EXPECT_TRUE(Peer::hidden(*executor_, owner).empty());
    EXPECT_FALSE(Peer::pending(*executor_, owner));
  }
  EXPECT_FALSE(executor_->open_session());
  EXPECT_FALSE(executor_->clone(*full, 0));
  EXPECT_FALSE(executor_->initialize());
}

TEST_F(CudaDFlashNative, UnmodifiedOldArtifactRunsSoloAndFourOwnerDecode) {
  create(false);
  ASSERT_TRUE(executor_);
  std::array<b::SessionId, 4> owners;
  b::BatchInput batch;
  for (int owner = 0; owner < 4; ++owner) {
    owners[owner] = open();
    Tokens prompt(owner + 3);
    std::iota(prompt.begin(), prompt.end(), 1 + 17 * owner);
    batch.inputs.push_back(input(owners[owner], 0, std::move(prompt)));
  }
  b::BatchOutput out;
  ASSERT_TRUE(executor_->execute(batch, out));
  // Frozen eager first-token goldens, not outputs from this implementation.
  const std::array<b::Token, 4> first{7, 194, 84, 217};
  for (int owner = 0; owner < 4; ++owner) {
    ASSERT_TRUE(out.outputs[owner]);
    ASSERT_FALSE(out.outputs[owner]->tokens.empty());
    EXPECT_EQ(out.outputs[owner]->tokens.front(), first[owner]);
  }
  for (int round = 0; round < 4; ++round) {
    batch.inputs.clear();
    for (int owner = 0; owner < 4; ++owner) {
      ASSERT_TRUE(out.outputs[owner]);
      ASSERT_FALSE(out.outputs[owner]->tokens.empty());
      const int start = Peer::frontier(*executor_, owners[owner]);
      Peer::check(*executor_, owners[owner], start);
      batch.inputs.push_back(
          input(owners[owner], start, {out.outputs[owner]->tokens.back()}));
    }
    const auto before = executor_->stats();
    ASSERT_TRUE(executor_->execute(batch, out));
    EXPECT_EQ(
        executor_->stats().target_preamble_calls, before.target_preamble_calls);
    EXPECT_EQ(
        executor_->stats().pending_decode_owners - before.pending_decode_owners,
        4);
    for (int owner = 0; owner < 4; ++owner) {
      ASSERT_TRUE(out.outputs[owner]);
      Peer::check(
          *executor_,
          owners[owner],
          batch.inputs[owner].position + out.outputs[owner]->tokens.size());
    }
  }
  const auto before = executor_->stats();
  const int start = Peer::frontier(*executor_, owners[0]);
  ASSERT_TRUE(executor_->execute(
      {{input(owners[0], start, {out.outputs[0]->tokens.back()})}}, out));
  ASSERT_TRUE(out.outputs[0]);
  Peer::check(*executor_, owners[0], start + out.outputs[0]->tokens.size());
  EXPECT_EQ(
      executor_->stats().pending_decode_owners - before.pending_decode_owners,
      1);
  EXPECT_EQ(
      executor_->stats().target_preamble_calls, before.target_preamble_calls);
  EXPECT_EQ(executor_->stats().failed_batches, 0);
}
} // namespace
