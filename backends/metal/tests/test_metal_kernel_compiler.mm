/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalKernelCompiler.h>
#include <executorch/backends/metal/core/MetalKernel.h>
#include <string>

using executorch::backends::metal_v2::MetalKernelCompiler;
using executorch::backends::metal_v2::MetalKernel;

namespace {

class MetalKernelCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) GTEST_SKIP();
    compiler_ = std::make_unique<MetalKernelCompiler>(device_);
  }
  void TearDown() override { compiler_.reset(); }
  id<MTLDevice> device_ = nil;
  std::unique_ptr<MetalKernelCompiler> compiler_;
};

// Trivial MSL kernel for compilation tests.
constexpr const char* kTrivialKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void trivial_kernel(device float* out [[buffer(0)]],
                           uint tid [[thread_position_in_grid]]) {
  out[tid] = float(tid);
}
)MSL";

TEST_F(MetalKernelCompilerTest, CompileTrivialKernelSucceeds) {
  auto k = compiler_->compile(kTrivialKernelSrc, "trivial_kernel");
  ASSERT_NE(k, nullptr);
  EXPECT_NE(k->pipeline(), nil);
  EXPECT_STREQ(k->name(), "trivial_kernel");
}

TEST_F(MetalKernelCompilerTest, CompileBadSourceReturnsNullptr) {
  auto k = compiler_->compile("not valid msl ;;;", "no_function");
  EXPECT_EQ(k, nullptr);
}

TEST_F(MetalKernelCompilerTest, CompileMissingFunctionReturnsNullptr) {
  auto k = compiler_->compile(kTrivialKernelSrc, "no_such_function");
  EXPECT_EQ(k, nullptr);
}

TEST_F(MetalKernelCompilerTest, FunctionConstantsEmptyByDefault) {
  MetalKernelCompiler::FunctionConstants fc;
  EXPECT_TRUE(fc.empty());
  EXPECT_EQ(fc.fingerprint(), "");
}

TEST_F(MetalKernelCompilerTest, FunctionConstantsBoolFingerprint) {
  MetalKernelCompiler::FunctionConstants fc;
  fc.bools.push_back({0, true});
  fc.bools.push_back({3, false});
  EXPECT_FALSE(fc.empty());
  // Format: @<idx><val>,...
  EXPECT_EQ(fc.fingerprint(), "@01,30,");
}

TEST_F(MetalKernelCompilerTest, FunctionConstantsIntFingerprint) {
  MetalKernelCompiler::FunctionConstants fc;
  fc.ints.push_back({26, 42});
  EXPECT_EQ(fc.fingerprint(), "#26=42,");
}

TEST_F(MetalKernelCompilerTest, FunctionConstantsMixedFingerprint) {
  MetalKernelCompiler::FunctionConstants fc;
  fc.bools.push_back({0, true});
  fc.ints.push_back({26, 42});
  // bools section first, ints second.
  EXPECT_EQ(fc.fingerprint(), "@01,#26=42,");
}

TEST_F(MetalKernelCompilerTest, FunctionConstantsFingerprintDistinguishesBoolFromInt) {
  MetalKernelCompiler::FunctionConstants fc_bool;
  fc_bool.bools.push_back({1, true});
  MetalKernelCompiler::FunctionConstants fc_int;
  fc_int.ints.push_back({1, 1});
  EXPECT_NE(fc_bool.fingerprint(), fc_int.fingerprint());
}

TEST_F(MetalKernelCompilerTest, HasBinaryArchiveFalseInitially) {
  EXPECT_FALSE(compiler_->hasBinaryArchive());
}

TEST_F(MetalKernelCompilerTest, LoadBinaryArchiveOnMissingPathReturnsFalse) {
  // Path that won't exist.
  EXPECT_FALSE(compiler_->loadBinaryArchive("/tmp/nonexistent.metallib.archive"));
}

// P1 #21: FunctionConstants actually specialize — two different bool FC
// values for the same source produce two distinct PSOs.
constexpr const char* kFcKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
constant bool use_double [[function_constant(0)]];
kernel void fc_kernel(device float* out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
  out[tid] = use_double ? 2.0f : 1.0f;
}
)MSL";

TEST_F(MetalKernelCompilerTest, FunctionConstantsProduceDistinctPSOs) {
  MetalKernelCompiler::FunctionConstants fc_false;
  fc_false.bools.push_back({0, false});
  MetalKernelCompiler::FunctionConstants fc_true;
  fc_true.bools.push_back({0, true});
  auto k_false = compiler_->compile(kFcKernelSrc, "fc_kernel", &fc_false);
  auto k_true = compiler_->compile(kFcKernelSrc, "fc_kernel", &fc_true);
  ASSERT_NE(k_false, nullptr);
  ASSERT_NE(k_true, nullptr);
  EXPECT_NE(k_false->pipeline(), k_true->pipeline())
      << "different FC values must produce distinct PSOs";
}

// P1 #23: getOrCompilePsoFromSource — lazy library + PSO cache.
TEST_F(MetalKernelCompilerTest, GetOrCompilePsoFromSourceCacheMissAndHit) {
  std::string src = kTrivialKernelSrc;
  int factory_calls = 0;
  auto factory = [&]() -> std::string { ++factory_calls; return src; };
  id<MTLComputePipelineState> pso1 = compiler_->getOrCompilePsoFromSource(
      "test_lib_key", factory, "trivial_kernel", nullptr);
  ASSERT_NE(pso1, nil);
  EXPECT_EQ(factory_calls, 1);
  // Second call with same key should hit cache (factory NOT called).
  id<MTLComputePipelineState> pso2 = compiler_->getOrCompilePsoFromSource(
      "test_lib_key", factory, "trivial_kernel", nullptr);
  EXPECT_EQ(pso2, pso1);
  EXPECT_EQ(factory_calls, 1) << "factory should not run on cache hit";
}

// P1 #22: binary archive save then load roundtrip.
TEST_F(MetalKernelCompilerTest, BinaryArchiveSaveLoadRoundtrip) {
  // First, compile something so the archive has content.
  auto k = compiler_->compile(kTrivialKernelSrc, "trivial_kernel");
  ASSERT_NE(k, nullptr);
  // Save to a temp path.
  NSString* tempDir = NSTemporaryDirectory();
  NSString* archivePath =
      [tempDir stringByAppendingPathComponent:@"metal_v2_test_archive.bin"];
  // Remove any leftover from prior runs.
  [[NSFileManager defaultManager] removeItemAtPath:archivePath error:nil];
  bool saved = compiler_->saveBinaryArchive([archivePath UTF8String]);
  if (!saved) {
    GTEST_SKIP() << "saveBinaryArchive not supported in this environment "
                 << "(may require MTL4-only paths or specific OS version)";
  }
  // Fresh compiler instance loads from disk.
  auto compiler2 = std::make_unique<MetalKernelCompiler>(device_);
  EXPECT_TRUE(compiler2->loadBinaryArchive([archivePath UTF8String]));
  EXPECT_TRUE(compiler2->hasBinaryArchive());
  // Cleanup.
  [[NSFileManager defaultManager] removeItemAtPath:archivePath error:nil];
}

}  // namespace
