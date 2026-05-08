/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// test_mps_interop — Unit 7 side-door binding contract end-to-end.
//
// Verifies MpsInterop::encodeWithLegacyCommandBuffer correctly:
//   1. Invokes the encode_fn with a valid MPSCommandBuffer.
//   2. Forwards side_door_binds to the recorder, populating the per-CB
//      binds_ vector (verified via boundBufferCountForTesting).
//   3. Produces correct numeric output when MPSGraph encodes a real op.
//   4. Defaults gracefully when side_door_binds is nullptr/0.
//
// Build gating: this file's tests only run under the MTL3 + MPSGraph
// configuration (ET_METAL_USE_MPSGRAPH=1). Under MTL4
// (ET_METAL4_ENABLE=1), MpsInterop.h's #error gate would prevent
// compilation, so we wrap the entire body in #if ET_METAL_USE_MPSGRAPH
// and the test binary becomes a trivial gtest_main with no test cases
// (gtest treats this as a clean run).
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <gtest/gtest.h>

#include <executorch/backends/metal/core/MpsInterop.h>

#if ET_METAL_USE_MPSGRAPH

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/core/MetalCommandRecorder.h>

#include <vector>

using executorch::backends::metal_v2::MetalStream;

namespace {

class MpsInteropTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MTLCreateSystemDefaultDevice()) {
      GTEST_SKIP() << "no Metal device available";
    }
    stream_ = std::make_unique<MetalStream>();
  }
  void TearDown() override {
    stream_.reset();
  }
  std::unique_ptr<MetalStream> stream_;
};

// Build a trivial 1-input MPSGraph that adds 1.0 to its input.
struct AddOneGraph {
  MPSGraph* graph = nil;
  MPSGraphTensor* input = nil;
  MPSGraphTensor* output = nil;
};

static AddOneGraph buildAddOneGraph(NSArray<NSNumber*>* shape) {
  AddOneGraph g;
  g.graph = [[MPSGraph alloc] init];
  g.input = [g.graph placeholderWithShape:shape
                                  dataType:MPSDataTypeFloat32
                                      name:@"input"];
  MPSGraphTensor* one =
      [g.graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
  g.output = [g.graph additionWithPrimaryTensor:g.input
                                secondaryTensor:one
                                           name:@"output"];
  return g;
}

static MPSGraphTensorData* makeTD(id<MTLBuffer> buf, NSArray<NSNumber*>* shape) {
  return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                  shape:shape
                                               dataType:MPSDataTypeFloat32];
}

// Defaults: nullptr / 0 side_door_binds — encode_fn must still run.
TEST_F(MpsInteropTest, EncodeWithoutSideDoorBindsRuns) {
  bool encode_fn_called = false;
  stream_->mps().encodeWithLegacyCommandBuffer(
      [&](MPSCommandBuffer* cb) {
        ASSERT_NE(cb, nil);
        encode_fn_called = true;
      },
      nullptr,
      0);
  EXPECT_TRUE(encode_fn_called);
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), 0u);
}

// Single buffer: side_door_binds populates the recorder's binds_.
TEST_F(MpsInteropTest, SideDoorBindsRecordsSingleBuffer) {
  constexpr size_t N = 4;
  float* in = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  ASSERT_NE(in, nullptr);
  for (size_t i = 0; i < N; ++i) in[i] = static_cast<float>(i);

  auto bind = stream_->allocator().bufferForPtr(in, N * sizeof(float));
  ASSERT_NE(bind.mtl, nil);

  id<MTLBuffer> binds[1] = { bind.mtl };
  size_t before = stream_->recorder().boundBufferCountForTesting();
  stream_->mps().encodeWithLegacyCommandBuffer(
      [&](MPSCommandBuffer* cb) {
        ASSERT_NE(cb, nil);
      },
      binds,
      1);
  EXPECT_GE(stream_->recorder().boundBufferCountForTesting(), before + 1);
  EXPECT_TRUE(stream_->recorder().isBoundForTesting(bind.mtl));

  stream_->allocator().free(in);
}

// Multiple buffers: all are recorded.
TEST_F(MpsInteropTest, SideDoorBindsRecordsMultipleBuffers) {
  constexpr size_t N = 4;
  float* a = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  float* b = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  float* c = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);

  auto ba = stream_->allocator().bufferForPtr(a, N * sizeof(float));
  auto bb = stream_->allocator().bufferForPtr(b, N * sizeof(float));
  auto bc = stream_->allocator().bufferForPtr(c, N * sizeof(float));
  ASSERT_NE(ba.mtl, nil);
  ASSERT_NE(bb.mtl, nil);
  ASSERT_NE(bc.mtl, nil);

  id<MTLBuffer> binds[3] = { ba.mtl, bb.mtl, bc.mtl };
  size_t before = stream_->recorder().boundBufferCountForTesting();
  stream_->mps().encodeWithLegacyCommandBuffer(
      [&](MPSCommandBuffer* cb) { ASSERT_NE(cb, nil); },
      binds,
      3);
  EXPECT_GE(stream_->recorder().boundBufferCountForTesting(), before + 3);
  EXPECT_TRUE(stream_->recorder().isBoundForTesting(ba.mtl));
  EXPECT_TRUE(stream_->recorder().isBoundForTesting(bb.mtl));
  EXPECT_TRUE(stream_->recorder().isBoundForTesting(bc.mtl));

  stream_->allocator().free(a);
  stream_->allocator().free(b);
  stream_->allocator().free(c);
}

// Duplicate buffers in side_door_binds are deduplicated by the recorder
// (recordBind uses bound_buffers_ set for dedup).
TEST_F(MpsInteropTest, SideDoorBindsDedupsDuplicates) {
  constexpr size_t N = 4;
  float* a = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  ASSERT_NE(a, nullptr);
  auto ba = stream_->allocator().bufferForPtr(a, N * sizeof(float));
  ASSERT_NE(ba.mtl, nil);

  id<MTLBuffer> binds[3] = { ba.mtl, ba.mtl, ba.mtl };
  size_t before = stream_->recorder().boundBufferCountForTesting();
  stream_->mps().encodeWithLegacyCommandBuffer(
      [&](MPSCommandBuffer* cb) { ASSERT_NE(cb, nil); },
      binds,
      3);
  // Even though we passed 3 entries, dedup means only one new entry.
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), before + 1);

  stream_->allocator().free(a);
}

// End-to-end: build a real MPSGraph (out = in + 1), encode it via
// MpsInterop, sync, and verify the output. This exercises the full
// side-door path including the actual GPU dispatch.
TEST_F(MpsInteropTest, MPSGraphEncodeAndExecuteProducesCorrectOutput) {
  constexpr size_t N = 16;
  float* in = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  float* out = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  ASSERT_NE(in, nullptr);
  ASSERT_NE(out, nullptr);
  for (size_t i = 0; i < N; ++i) in[i] = static_cast<float>(i);

  @autoreleasepool {
    NSArray<NSNumber*>* shape = @[ @(static_cast<NSInteger>(N)) ];
    AddOneGraph g = buildAddOneGraph(shape);

    auto bin = stream_->allocator().bufferForPtr(in, N * sizeof(float));
    auto bout = stream_->allocator().bufferForPtr(out, N * sizeof(float));
    ASSERT_NE(bin.mtl, nil);
    ASSERT_NE(bout.mtl, nil);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [NSMutableDictionary dictionary];
    feeds[g.input] = makeTD(bin.mtl, shape);
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [NSMutableDictionary dictionary];
    results[g.output] = makeTD(bout.mtl, shape);

    id<MTLBuffer> binds[2] = { bin.mtl, bout.mtl };
    stream_->mps().encodeWithLegacyCommandBuffer(
        [&](MPSCommandBuffer* mpsCB) {
          [g.graph encodeToCommandBuffer:mpsCB
                                    feeds:feeds
                         targetOperations:nil
                        resultsDictionary:results
                      executionDescriptor:nil];
        },
        binds,
        2);
  }
  stream_->sync();

  for (size_t i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 1.0f)
        << "MPSGraph add-one mismatch at i=" << i;
  }

  stream_->allocator().free(in);
  stream_->allocator().free(out);
}

// P0 #17: two consecutive encodeWithLegacyCommandBuffer calls in same
// CB. Both side-door bindings must record correctly + both encoded
// regions must execute.
TEST_F(MpsInteropTest, TwoConsecutiveEncodesBothExecute) {
  constexpr size_t N = 8;
  float* in1 = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  float* out1 = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  float* in2 = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  float* out2 = static_cast<float*>(stream_->allocator().alloc(N * sizeof(float)));
  for (size_t i = 0; i < N; ++i) { in1[i] = 10.0f; in2[i] = 20.0f; }

  @autoreleasepool {
    NSArray<NSNumber*>* shape = @[ @(static_cast<NSInteger>(N)) ];
    AddOneGraph g1 = buildAddOneGraph(shape);
    AddOneGraph g2 = buildAddOneGraph(shape);
    auto bi1 = stream_->allocator().bufferForPtr(in1, N * sizeof(float));
    auto bo1 = stream_->allocator().bufferForPtr(out1, N * sizeof(float));
    auto bi2 = stream_->allocator().bufferForPtr(in2, N * sizeof(float));
    auto bo2 = stream_->allocator().bufferForPtr(out2, N * sizeof(float));
    NSMutableDictionary* feeds1 = [NSMutableDictionary dictionary];
    feeds1[g1.input] = makeTD(bi1.mtl, shape);
    NSMutableDictionary* res1 = [NSMutableDictionary dictionary];
    res1[g1.output] = makeTD(bo1.mtl, shape);
    NSMutableDictionary* feeds2 = [NSMutableDictionary dictionary];
    feeds2[g2.input] = makeTD(bi2.mtl, shape);
    NSMutableDictionary* res2 = [NSMutableDictionary dictionary];
    res2[g2.output] = makeTD(bo2.mtl, shape);

    id<MTLBuffer> b1[2] = { bi1.mtl, bo1.mtl };
    stream_->mps().encodeWithLegacyCommandBuffer(
        [&](MPSCommandBuffer* cb) {
          [g1.graph encodeToCommandBuffer:cb feeds:feeds1
                          targetOperations:nil resultsDictionary:res1
                       executionDescriptor:nil];
        }, b1, 2);
    id<MTLBuffer> b2[2] = { bi2.mtl, bo2.mtl };
    stream_->mps().encodeWithLegacyCommandBuffer(
        [&](MPSCommandBuffer* cb) {
          [g2.graph encodeToCommandBuffer:cb feeds:feeds2
                          targetOperations:nil resultsDictionary:res2
                       executionDescriptor:nil];
        }, b2, 2);
  }
  stream_->sync();

  for (size_t i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(out1[i], 11.0f) << "graph 1 mismatch at i=" << i;
    EXPECT_FLOAT_EQ(out2[i], 21.0f) << "graph 2 mismatch at i=" << i;
  }
  stream_->allocator().free(in1); stream_->allocator().free(out1);
  stream_->allocator().free(in2); stream_->allocator().free(out2);
}

// P0 #15: encode_fn that does nothing (no actual MPSGraph encode). The
// CB is still committed; sync() must not crash. This stresses the
// "side-door declared binds without actual GPU work" path.
TEST_F(MpsInteropTest, EncodeFnThatDoesNothingIsSafe) {
  bool ran = false;
  stream_->mps().encodeWithLegacyCommandBuffer(
      [&](MPSCommandBuffer* cb) {
        // No MPSGraph encoding; the CB is empty.
        ASSERT_NE(cb, nil);
        ran = true;
      },
      nullptr, 0);
  stream_->sync();  // must not crash
  EXPECT_TRUE(ran);
}

}  // namespace

#endif  // ET_METAL_USE_MPSGRAPH