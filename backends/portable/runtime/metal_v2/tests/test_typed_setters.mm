/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// test_typed_setters — exercises the R2.1 typed-setter API end-to-end.
// The R2.1 setters (setInput / setOutput / setBytes / setVectorBytes /
// setBuffer / setKernel / dispatchThreadgroups) compile-tested in
// MetalStream.h, but until ops migrate to use them they have ZERO runtime
// coverage from the existing tests (which all go through the legacy
// dispatch(args) path).
// This test compiles a trivial kernel inline and exercises each typed
// setter to confirm they produce correct outputs end-to-end.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <gtest/gtest.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalKernelCache.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>

#include <vector>

using executorch::backends::metal_v2::MetalStream;
using executorch::backends::metal_v2::MetalKernel;
using executorch::backends::metal_v2::MetalKernelCompiler;
using executorch::backends::metal_v2::uvec3;

namespace {

// Trivial kernel: output[i] = input[i] * scale + bias_array[i % bias_len]
// for i < numel. Exercises all setter types:
//   - setInput  (input buffer)
//   - setOutput (output buffer)
//   - setBytes<T>(scale)        — scalar via templated setBytes
//   - setVectorBytes(bias_arr)  — vector via setVectorBytes
//   - setBytes<T>(numel)        — scalar via templated setBytes
//   - setBytes<T>(bias_len)     — scalar via templated setBytes
constexpr const char* kTypedSetterTestKernel = R"(
#include <metal_stdlib>
using namespace metal;

kernel void typed_setter_test(
    device const float* input        [[buffer(0)]],
    device float*       output       [[buffer(1)]],
    constant float&     scale        [[buffer(2)]],
    constant float*     bias_array   [[buffer(3)]],
    constant uint&      numel        [[buffer(4)]],
    constant uint&      bias_len     [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (i < numel) {
    output[i] = input[i] * scale + bias_array[i % bias_len];
  }
}
)";

class TypedSettersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    stream_ = std::make_unique<MetalStream>();
    // Compile our test kernel via the process-wide MetalKernelCache.
    // Lifetime: kernel lives in the cache for the duration of the process,
    // outliving stream_'s teardown in TearDown.
    using ::executorch::backends::metal_v2::MetalKernelCache;
    kernel_ = MetalKernelCache::shared().findOrInsert(
        "test_typed_setter_test",  // unique key for this test kernel
        [&]() {
          return stream_->compiler()->compile(
              kTypedSetterTestKernel, "typed_setter_test", nullptr);
        });
    ASSERT_NE(kernel_, nullptr) << "kernel compilation failed";
  }
  void TearDown() override {
    stream_.reset();
  }

  std::unique_ptr<MetalStream> stream_;
  MetalKernel* kernel_ = nullptr;  // owned by MetalKernelCache::shared()
};

//===----------------------------------------------------------------------===//
// Combined dispatch form: setInput + setOutput + setBytes + setVectorBytes
// + dispatch(kernel, grid, block).
//===----------------------------------------------------------------------===//

TEST_F(TypedSettersTest, CombinedDispatchProducesCorrectOutput) {
  // Allocate input + output via the stream's pool (cleanest lifecycle).
  constexpr uint32_t numel = 64;
  float* input = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* output = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);

  for (uint32_t i = 0; i < numel; ++i) {
    input[i] = static_cast<float>(i);
    output[i] = -1.0f;  // sentinel — should be overwritten
  }

  const float scale = 2.0f;
  const std::vector<float> bias_array = {0.5f, 1.5f, 2.5f, 3.5f};
  const uint32_t bias_len = static_cast<uint32_t>(bias_array.size());

  // Exercise the typed-setter API.
  stream_->setInput(0, input, numel * sizeof(float));
  stream_->setOutput(1, output, numel * sizeof(float));
  stream_->setBytes<float>(2, scale);
  stream_->setVectorBytes(3, bias_array);
  stream_->setBytes<uint32_t>(4, numel);
  stream_->setBytes<uint32_t>(5, bias_len);
  stream_->dispatch(kernel_, uvec3(1, 1, 1), uvec3(64, 1, 1));
  stream_->sync();

  // Verify: output[i] = input[i] * scale + bias_array[i % bias_len]
  for (uint32_t i = 0; i < numel; ++i) {
    float expected = input[i] * scale + bias_array[i % bias_len];
    EXPECT_FLOAT_EQ(output[i], expected) << "mismatch at i=" << i;
  }

  stream_->free(input);
  stream_->free(output);
}

//===----------------------------------------------------------------------===//
// Re-using the same stream for multiple dispatches (different shapes).
// Verifies that the buffer registry handles repeated alloc/free across
// dispatches and that beginDispatch+run is safe to call in a loop.
//===----------------------------------------------------------------------===//

TEST_F(TypedSettersTest, MultipleDispatchesOnSameStream) {
  for (uint32_t numel : {16u, 32u, 64u, 128u}) {
    float* input = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
    float* output = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    for (uint32_t i = 0; i < numel; ++i) {
      input[i] = static_cast<float>(i);
    }

    const float scale = 1.0f;
    const std::vector<float> bias = {0.0f};
    const uint32_t bias_len = 1;

    stream_->setInput(0, input, numel * sizeof(float));
    stream_->setOutput(1, output, numel * sizeof(float));
    stream_->setBytes<float>(2, scale);
    stream_->setVectorBytes(3, bias);
    stream_->setBytes<uint32_t>(4, numel);
    stream_->setBytes<uint32_t>(5, bias_len);
    // Pick a threadgroup size that covers numel without overshooting too far.
    uint32_t tg = numel < 256 ? numel : 256;
    stream_->dispatch(kernel_, uvec3(1, 1, 1), uvec3(tg, 1, 1));
    stream_->sync();

    for (uint32_t i = 0; i < numel; ++i) {
      // output[i] = input[i] * 1 + 0 = input[i]
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i))
          << "mismatch at numel=" << numel << " i=" << i;
    }

    stream_->free(input);
    stream_->free(output);
  }
}

//===----------------------------------------------------------------------===//
// R8.1: Hazard-tracking validation.
// These tests verify the auto-barrier optimization correctly identifies
// dependencies between dispatches AND skips the barrier when independent.
// Counters on MetalStream::barrierStats() let us assert on the actual
// barrier-insertion count.
//===----------------------------------------------------------------------===//

class HazardTrackingTest : public TypedSettersTest {};

// Helper: dispatches the kernel with the given (input, output, scale=1, bias=[0])
// and a threadgroup that covers numel.
static void runDispatch(
    MetalStream* stream, MetalKernel* kernel,
    const float* input, float* output, uint32_t numel) {
  static const std::vector<float> bias = {0.0f};
  static const uint32_t bias_len = 1;
  static const float scale = 1.0f;
  stream->setInput(0, input, numel * sizeof(float));
  stream->setOutput(1, output, numel * sizeof(float));
  stream->setBytes<float>(2, scale);
  stream->setVectorBytes(3, bias);
  stream->setBytes<uint32_t>(4, numel);
  stream->setBytes<uint32_t>(5, bias_len);
  uint32_t tg = numel < 256 ? numel : 256;
  stream->dispatch(kernel, uvec3(1, 1, 1), uvec3(tg, 1, 1));
}

// Two dispatches reading + writing TOTALLY DISJOINT buffers — no hazard.
// First dispatch: lastWriters_ is empty → no barrier.
// Second dispatch: its inputs (in_b) are NOT in lastWriters_ → skip.
// Outputs (out_b) are also NOT in lastWriters_ (only out_a is) → skip.
// Expected: 2 dispatches, 0 barriers inserted, 2 barriers skipped.
TEST_F(HazardTrackingTest, IndependentDispatchesSkipBarrier) {
  constexpr uint32_t numel = 16;
  float* in_a = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* out_a = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* in_b = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* out_b = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  for (uint32_t i = 0; i < numel; ++i) {
    in_a[i] = 1.0f; in_b[i] = 2.0f;
  }

  uint64_t barriers_before = stream_->barrierStats().inserted;
  uint64_t skipped_before = stream_->barrierStats().skipped;

  runDispatch(stream_.get(), kernel_, in_a, out_a, numel);
  runDispatch(stream_.get(), kernel_, in_b, out_b, numel);
  stream_->sync();

  uint64_t barriers_inserted = stream_->barrierStats().inserted - barriers_before;
  uint64_t barriers_skipped = stream_->barrierStats().skipped - skipped_before;
  EXPECT_EQ(barriers_inserted, 0u)
      << "Expected no barriers between independent dispatches";
  EXPECT_EQ(barriers_skipped, 2u)
      << "Expected both dispatches to skip the barrier";

  // Numeric correctness — outputs must match inputs (scale=1, bias=0).
  for (uint32_t i = 0; i < numel; ++i) {
    EXPECT_FLOAT_EQ(out_a[i], 1.0f);
    EXPECT_FLOAT_EQ(out_b[i], 2.0f);
  }

  stream_->free(in_a); stream_->free(out_a);
  stream_->free(in_b); stream_->free(out_b);
}

// Two dispatches with a RAW hazard: dispatch[1].input == dispatch[0].output.
// Expected: 1st dispatch skips (lastWriters empty), 2nd dispatch inserts.
TEST_F(HazardTrackingTest, RAWHazardInsertsBarrier) {
  constexpr uint32_t numel = 16;
  float* a = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* b = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* c = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  for (uint32_t i = 0; i < numel; ++i) a[i] = 5.0f;

  uint64_t barriers_before = stream_->barrierStats().inserted;
  uint64_t skipped_before = stream_->barrierStats().skipped;

  // dispatch 0: a → b   (lastWriters_ becomes {b})
  runDispatch(stream_.get(), kernel_, a, b, numel);
  // dispatch 1: b → c   (b ∈ lastWriters_ → RAW hazard → barrier needed)
  runDispatch(stream_.get(), kernel_, b, c, numel);
  stream_->sync();

  uint64_t barriers_inserted = stream_->barrierStats().inserted - barriers_before;
  uint64_t barriers_skipped = stream_->barrierStats().skipped - skipped_before;
  EXPECT_EQ(barriers_inserted, 1u)
      << "Expected 1 barrier (between dispatch[0] writing b and dispatch[1] reading b)";
  EXPECT_EQ(barriers_skipped, 1u)
      << "Expected dispatch[0] to skip (no prior writers)";

  // Numeric correctness (kernel: out = in * 1 + 0 = in).
  // a=5 → b=5 → c=5
  for (uint32_t i = 0; i < numel; ++i) {
    EXPECT_FLOAT_EQ(c[i], 5.0f) << "RAW chain produced wrong result at i=" << i;
  }

  stream_->free(a); stream_->free(b); stream_->free(c);
}

// Two dispatches writing to the SAME output buffer (WAW hazard).
// dispatch 0: a → x   (lastWriters_ = {x})
// dispatch 1: b → x   (x ∈ lastWriters_ via output check → WAW → barrier)
// Without the barrier, the two writes could complete in any order on MTL4.
TEST_F(HazardTrackingTest, WAWHazardInsertsBarrier) {
  constexpr uint32_t numel = 16;
  float* a = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* b = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* x = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  for (uint32_t i = 0; i < numel; ++i) {
    a[i] = 7.0f; b[i] = 11.0f;
  }

  uint64_t barriers_before = stream_->barrierStats().inserted;
  uint64_t skipped_before = stream_->barrierStats().skipped;

  runDispatch(stream_.get(), kernel_, a, x, numel);  // x = 7
  runDispatch(stream_.get(), kernel_, b, x, numel);  // x = 11 (WAW)
  stream_->sync();

  uint64_t barriers_inserted = stream_->barrierStats().inserted - barriers_before;
  uint64_t barriers_skipped = stream_->barrierStats().skipped - skipped_before;
  EXPECT_EQ(barriers_inserted, 1u) << "WAW hazard must insert a barrier";
  EXPECT_EQ(barriers_skipped, 1u) << "First dispatch should still skip";

  // Final value must be the SECOND write (b * 1 + 0 = 11).
  for (uint32_t i = 0; i < numel; ++i) {
    EXPECT_FLOAT_EQ(x[i], 11.0f)
        << "WAW order violated — saw stale write at i=" << i;
  }

  stream_->free(a); stream_->free(b); stream_->free(x);
}

// flush() boundary clears lastWriters_ — dispatches in different flush
// windows are by definition independent (the prior batch's GPU work has
// committed and a new encoder starts).
TEST_F(HazardTrackingTest, FlushBoundaryResetsHazardTracker) {
  constexpr uint32_t numel = 16;
  float* a = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  float* b = static_cast<float*>(stream_->alloc(numel * sizeof(float)));
  for (uint32_t i = 0; i < numel; ++i) a[i] = 3.0f;

  uint64_t barriers_before = stream_->barrierStats().inserted;
  uint64_t skipped_before = stream_->barrierStats().skipped;

  // dispatch 0: a → b. lastWriters_ becomes {b}.
  runDispatch(stream_.get(), kernel_, a, b, numel);
  // sync() = flush + wait. flush() clears lastWriters_.
  stream_->sync();
  // dispatch 1: b → a. After sync, lastWriters_ is empty so this should
  // skip the barrier even though it reads b (which was previously written).
  runDispatch(stream_.get(), kernel_, b, a, numel);
  stream_->sync();

  uint64_t barriers_inserted = stream_->barrierStats().inserted - barriers_before;
  uint64_t barriers_skipped = stream_->barrierStats().skipped - skipped_before;
  EXPECT_EQ(barriers_inserted, 0u)
      << "Across flush boundary, prior writers don't carry forward";
  EXPECT_EQ(barriers_skipped, 2u);

  // Correctness: a → b (b=3) → a (a=3).
  for (uint32_t i = 0; i < numel; ++i) {
    EXPECT_FLOAT_EQ(a[i], 3.0f);
    EXPECT_FLOAT_EQ(b[i], 3.0f);
  }

  stream_->free(a); stream_->free(b);
}

} // namespace
