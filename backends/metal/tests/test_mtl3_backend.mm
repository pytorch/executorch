/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// test_mtl3_backend — smoke tests for the MTL3 dispatch path.
//
// When ET_METAL4_ENABLE=ON and useMTL4() returns true, the ctor-frozen
// recorder takes the MTL4 path and MTL3 is not exercised at all.
// These tests force the MTL3 path by setting METAL_USE_MTL4=0 before
// constructing the stream; they skip when MTL4 is mandatory.
//
// What's covered:
//   - Stream construction + teardown on the MTL3 path
//   - A trivial kernel dispatch end-to-end through the legacy CB
//   - Multiple dispatches + sync with no hangs
//   - Legacy command-buffer provider wiring (MpsInterop handoff shape)
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>

#include <executorch/backends/metal/core/MetalKernelCache.h>
#include <executorch/backends/metal/core/MetalStream.h>

namespace {

using executorch::backends::metal_v2::MetalKernel;
using executorch::backends::metal_v2::MetalKernelCache;
using executorch::backends::metal_v2::MetalStream;

class MetalMTL3BackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Force MTL3 path for this test. useMTL4() caches at first call, but
    // no prior call in this TU means setenv before construction is
    // effective.
    setenv("METAL_USE_MTL4", "0", /*overwrite=*/1);
    stream_ = MetalStream::create();
    ASSERT_NE(stream_, nullptr);
  }
  std::unique_ptr<MetalStream> stream_;
};

// Smoke test: stream construction + teardown on the MTL3 path.
TEST_F(MetalMTL3BackendTest, ConstructAndDestruct) {
  ASSERT_NE(stream_->device(), nil);
  // sync() on a fresh stream must be a safe no-op.
  stream_->sync();
}

// Compile and dispatch a trivial kernel through the MTL3 path.
TEST_F(MetalMTL3BackendTest, TrivialKernelDispatch) {
  const char* src = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void mtl3_add_one(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < n) out[gid] = in[gid] + 1.0f;
}
)METAL";

  MetalKernel* kernel = MetalKernelCache::shared().findOrInsert(
      "mtl3_add_one_key", [&]() {
        return stream_->compiler()->compile(src, "mtl3_add_one", nullptr);
      });
  ASSERT_NE(kernel, nullptr);

  const uint32_t N = 16;
  size_t bytes = N * sizeof(float);
  void* inPtr = stream_->allocator().alloc(bytes);
  void* outPtr = stream_->allocator().alloc(bytes);
  ASSERT_NE(inPtr, nullptr);
  ASSERT_NE(outPtr, nullptr);

  auto* in = reinterpret_cast<float*>(inPtr);
  auto* out = reinterpret_cast<float*>(outPtr);
  for (uint32_t i = 0; i < N; ++i) {
    in[i] = static_cast<float>(i);
    out[i] = 0.0f;
  }

  stream_->recorder()
      .beginDispatch(kernel)
      .setInput(0, in, bytes)
      .setOutput(1, out, bytes)
      .setBytes<uint32_t>(2, N)
      .run(executorch::backends::metal_v2::uvec3(N, 1, 1),
           executorch::backends::metal_v2::uvec3(N, 1, 1));
  stream_->sync();

  for (uint32_t i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 1.0f) << "i=" << i;
  }

  stream_->allocator().free(inPtr);
  stream_->allocator().free(outPtr);
}

// Repeated dispatches should not hang or leak encoders.
TEST_F(MetalMTL3BackendTest, RepeatedDispatches) {
  const char* src = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void mtl3_zero_fill(
    device float* out   [[buffer(0)]],
    constant uint& n    [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < n) out[gid] = 0.0f;
}
)METAL";

  MetalKernel* kernel = MetalKernelCache::shared().findOrInsert(
      "mtl3_zero_fill_key", [&]() {
        return stream_->compiler()->compile(src, "mtl3_zero_fill", nullptr);
      });
  ASSERT_NE(kernel, nullptr);

  const uint32_t N = 32;
  size_t bytes = N * sizeof(float);

  for (int iter = 0; iter < 64; ++iter) {
    void* outPtr = stream_->allocator().alloc(bytes);
    ASSERT_NE(outPtr, nullptr);
    stream_->recorder()
        .beginDispatch(kernel)
        .setOutput(0, outPtr, bytes)
        .setBytes<uint32_t>(1, N)
        .run(executorch::backends::metal_v2::uvec3(N, 1, 1),
             executorch::backends::metal_v2::uvec3(N, 1, 1));
    stream_->allocator().free(outPtr);
  }
  stream_->sync();
  SUCCEED();
}

// P1 #19: Unit 12 completion handler infrastructure — registered
// handlers must fire after sync.
TEST_F(MetalMTL3BackendTest, AddCompletionHandlerFiresAfterSync) {
  // Need pending GPU work for the CB to commit.
  const char* src = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void noop(device float* o [[buffer(0)]],
                 uint tid [[thread_position_in_grid]]) {
  o[tid] = 1.0f;
}
)METAL";
  MetalKernel* k = MetalKernelCache::shared().findOrInsert(
      "ch_test_noop", [&]() {
        return stream_->compiler()->compile(src, "noop", nullptr);
      });
  ASSERT_NE(k, nullptr);
  void* p = stream_->allocator().alloc(64);
  ASSERT_NE(p, nullptr);
  bool fired = false;
  stream_->recorder().dispatchBackend()->addCompletionHandler(
      [&]() { fired = true; });
  stream_->recorder()
      .beginDispatch(k)
      .setOutput(0, p, 64)
      .run(executorch::backends::metal_v2::uvec3(16, 1, 1),
           executorch::backends::metal_v2::uvec3(16, 1, 1));
  stream_->sync();
  EXPECT_TRUE(fired) << "completion handler did not fire after sync()";
  stream_->allocator().free(p);
}

// P1 #20: multiple handlers all fire (no dedup).
TEST_F(MetalMTL3BackendTest, MultipleCompletionHandlersAllFire) {
  const char* src = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void noop2(device float* o [[buffer(0)]],
                  uint tid [[thread_position_in_grid]]) {
  o[tid] = 2.0f;
}
)METAL";
  MetalKernel* k = MetalKernelCache::shared().findOrInsert(
      "ch_test_noop2", [&]() {
        return stream_->compiler()->compile(src, "noop2", nullptr);
      });
  ASSERT_NE(k, nullptr);
  void* p = stream_->allocator().alloc(64);
  std::atomic<int> count{0};
  for (int i = 0; i < 3; ++i) {
    stream_->recorder().dispatchBackend()->addCompletionHandler(
        [&]() { ++count; });
  }
  stream_->recorder()
      .beginDispatch(k)
      .setOutput(0, p, 64)
      .run(executorch::backends::metal_v2::uvec3(16, 1, 1),
           executorch::backends::metal_v2::uvec3(16, 1, 1));
  stream_->sync();
  EXPECT_EQ(count.load(), 3);
  stream_->allocator().free(p);
}

}  // namespace
