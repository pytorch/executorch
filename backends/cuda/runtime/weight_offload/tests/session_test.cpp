/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA-required unit tests for the weight-offload Session state machine.
// Unlike the Python end-to-end tests (which drive the whole export +
// AOTI run), these construct a Session directly from a synthetic
// Payload + AOTICatalog and a stub AOTI handle, then drive serve()
// through the hit / miss / evict / prefetch / pinned / budget-reject /
// out-of-range paths and assert on SessionStats + the budget
// invariants. This is the only coverage that exercises the eviction
// and prefetch branches at unit granularity.
//
// Requires a GPU (Session::create allocates dummies, a pinned host
// mirror, a cudaMemPool, and a copy stream). Each test SKIPs when no
// CUDA device is present.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/weight_offload/payload.h>
#include <executorch/backends/cuda/runtime/weight_offload/session.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/platform.h>

namespace {

using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::Span;
namespace aoti = executorch::backends::aoti;
namespace wo = executorch::backends::cuda::weight_offload;

constexpr int32_t kInt8 = 1; // element_size 1 => nbytes == numel

bool cudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Stub for the only AOTI handle function Session::create invokes. The
// real install validates coverage / device; here the catalog and
// dummies are synthetic, so a no-op success is the right stub.
aoti::AOTIRuntimeError stubUpdateUserManaged(
    aoti::AOTInductorModelContainerHandle,
    const aoti::AOTInductorConstantMapEntry*,
    size_t,
    bool,
    bool) {
  return Error::Ok;
}

std::string constName(size_t i) {
  return "w" + std::to_string(i);
}

class SessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    if (!cudaAvailable()) {
      GTEST_SKIP() << "CUDA device required for Session tests";
    }
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override {
    session_.reset(); // destroy Session before its compute stream
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  // Byte-precise budget via the internal runtime spec.
  void setBudget(uint64_t bytes) {
    BackendOption opt{};
    std::snprintf(
        opt.key, sizeof(opt.key), "_weight_offload_internal_budget_bytes");
    std::array<char, executorch::runtime::kMaxOptionValueLength> val{};
    std::snprintf(
        val.data(), val.size(), "%llu", static_cast<unsigned long long>(bytes));
    opt.value = val;
    specs_.assign(1, opt);
  }

  // Build + create a Session.
  //   const_bytes[i] : byte size of catalog constant i (int8 => numel)
  //   schedule_idx   : catalog indices, in probe order
  //   pin_idx        : catalog indices to pin resident
  // ``setBudget`` must have been called first. Returns the create
  // Result; on success the caller moves it into ``session_``.
  Result<std::unique_ptr<wo::Session>> build(
      const std::vector<uint64_t>& const_bytes,
      const std::vector<size_t>& schedule_idx,
      const std::vector<size_t>& pin_idx,
      uint64_t floor_bytes) {
    wo::Payload p;
    p.schema_version = 2;
    p.method_name = "m";
    p.floor_bytes = floor_bytes;
    for (size_t i : schedule_idx) {
      p.schedule.push_back(constName(i));
    }
    for (size_t i : pin_idx) {
      p.pin_fqns.push_back(constName(i));
    }
    std::unordered_set<std::string> seen;
    for (size_t i : schedule_idx) {
      const std::string nm = constName(i);
      if (!seen.insert(nm).second) {
        continue;
      }
      wo::ConstantMetadata m;
      m.fqn = nm;
      m.dtype = kInt8;
      m.sizes = {static_cast<int64_t>(const_bytes[i])};
      m.strides = {1};
      m.storage_offset = 0;
      m.nbytes = const_bytes[i];
      m.device_type = 1;
      m.device_index = 0;
      p.constants_metadata.push_back(std::move(m));
    }

    wo::AOTICatalog cat;
    cat.num_constants = const_bytes.size();
    uint64_t total = 0;
    for (size_t i = 0; i < const_bytes.size(); ++i) {
      cat.fqns.push_back(constName(i));
      cat.internal_names.push_back("c" + std::to_string(i));
      cat.data_sizes.push_back(static_cast<size_t>(const_bytes[i]));
      cat.fqn_to_index.emplace(constName(i), i);
      total += const_bytes[i];
    }
    blob_.assign(total, 0);

    handle_ = aoti::AOTIDelegateHandle{};
    handle_.update_user_managed_constant_buffer_pairs = &stubUpdateUserManaged;
    handle_.container_handle =
        reinterpret_cast<aoti::AOTInductorModelContainerHandle>(0x1);

    BackendInitContext ctx(
        /*runtime_allocator=*/nullptr,
        /*event_tracer=*/nullptr,
        /*method_name=*/"m",
        /*named_data_map=*/nullptr,
        Span<const BackendOption>(specs_.data(), specs_.size()));

    return wo::Session::create(
        p, &handle_, std::move(cat), blob_.data(), stream_, ctx);
  }

  // serve() one probe, assert success + a non-null borrowed tensor,
  // and free the wrapper (AOTI's RAII does this in production).
  void serveOk(wo::Session* s, int64_t probe_id) {
    aoti::Tensor* out = nullptr;
    ASSERT_EQ(s->serve(/*input=*/nullptr, probe_id, &out), Error::Ok);
    ASSERT_NE(out, nullptr);
    (void)executorch::backends::cuda::aoti_torch_delete_tensor_object(out);
  }

  cudaStream_t stream_{nullptr};
  std::unique_ptr<wo::Session> session_;
  std::vector<uint8_t> blob_;
  aoti::AOTIDelegateHandle handle_;
  std::vector<BackendOption> specs_;
};

TEST_F(SessionTest, CreateRejectsBudgetBelowFloor) {
  setBudget(150);
  auto res = build(
      /*const_bytes=*/{100, 100},
      /*schedule_idx=*/{0, 1},
      /*pin_idx=*/{},
      /*floor_bytes=*/200);
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.error(), Error::InvalidArgument);
}

TEST_F(SessionTest, AllFitNoEviction) {
  setBudget(100000);
  auto res = build({100, 100, 100}, {0, 1, 2}, {}, /*floor=*/200);
  ASSERT_TRUE(res.ok());
  session_ = std::move(res.get());

  EXPECT_EQ(session_->total_budget_bytes(), 100000u);
  EXPECT_EQ(session_->floor_bytes(), 200u);
  EXPECT_EQ(session_->streaming_budget_bytes(), 100000u);

  serveOk(session_.get(), 0);
  serveOk(session_.get(), 1);
  serveOk(session_.get(), 2);

  const auto& st = session_->stats();
  EXPECT_EQ(st.evictions, 0u);
  EXPECT_EQ(st.pool_hits + st.pool_misses, 3u);
  EXPECT_LE(session_->peak_live_bytes(), session_->streaming_budget_bytes());
}

TEST_F(SessionTest, MissThenHitSameWeight) {
  // Two probe sites reading the same constant; budget holds exactly
  // one copy. Second serve is a pool hit; the wrap-around prefetch
  // target is the same (live) FQN, so no prefetch is attempted.
  setBudget(100);
  auto res = build({100}, {0, 0}, {}, /*floor=*/100);
  ASSERT_TRUE(res.ok());
  session_ = std::move(res.get());

  serveOk(session_.get(), 0);
  serveOk(session_.get(), 1);

  const auto& st = session_->stats();
  EXPECT_EQ(st.pool_misses, 1u);
  EXPECT_EQ(st.pool_hits, 1u);
  EXPECT_EQ(st.evictions, 0u);
  EXPECT_EQ(st.prefetch_attempted, 0u);
}

TEST_F(SessionTest, PinnedResidentDoesNotStream) {
  // Constant 0 pinned, constant 1 streamed. The pinned serve takes the
  // resident path (no hit/miss bump); only the streaming serve counts.
  setBudget(/*pinned 100 + streaming 100*/ 200);
  auto res = build({100, 100}, {0, 1}, /*pin=*/{0}, /*floor=*/100);
  ASSERT_TRUE(res.ok());
  session_ = std::move(res.get());

  EXPECT_EQ(session_->pinned_bytes_total(), 100u);
  EXPECT_EQ(session_->streaming_budget_bytes(), 100u);

  serveOk(session_.get(), 0); // pinned
  serveOk(session_.get(), 1); // streamed (may be a prefetch hit)

  const auto& st = session_->stats();
  // Only the streaming constant participates in the pool; the pinned
  // serve bumps neither hits nor misses. Whether the streaming serve
  // lands as a hit (prefetched during the pinned serve) or a miss,
  // exactly one pool event is recorded.
  EXPECT_EQ(st.pool_hits + st.pool_misses, 1u);
}

TEST_F(SessionTest, PeakStaysWithinStreamingBudgetUnderEviction) {
  // 4 constants, budget holds ~2 -> eviction is forced. The invariant
  // under test: bytes_in_flight (hence peak) never exceeds the
  // streaming budget, no matter the access pattern.
  setBudget(250);
  auto res = build({100, 100, 100, 100}, {0, 1, 2, 3}, {}, /*floor=*/200);
  ASSERT_TRUE(res.ok());
  session_ = std::move(res.get());

  for (int round = 0; round < 2; ++round) {
    for (int64_t id = 0; id < 4; ++id) {
      serveOk(session_.get(), id);
    }
  }

  const auto& st = session_->stats();
  EXPECT_GT(st.evictions, 0u);
  EXPECT_GT(session_->peak_live_bytes(), 0u);
  EXPECT_LE(session_->peak_live_bytes(), session_->streaming_budget_bytes());
}

TEST_F(SessionTest, ProbeIdOutOfRangeRejected) {
  setBudget(100000);
  auto res = build({100, 100}, {0, 1}, {}, /*floor=*/200);
  ASSERT_TRUE(res.ok());
  session_ = std::move(res.get());

  aoti::Tensor* out = nullptr;
  EXPECT_EQ(session_->serve(nullptr, 9999, &out), Error::InvalidArgument);
  EXPECT_EQ(session_->serve(nullptr, -1, &out), Error::InvalidArgument);
}

} // namespace
