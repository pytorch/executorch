/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// test_buffer_registry — structural unit tests on BufferRegistry.
// These tests catch the entire bug class that produced audit Finding C1:
//   - The previous design had 3 hand-maintained sets (ptrToBuffer_,
//     externalBuffers_, copiedBuffers_). The free() routine had to read
//     externalBuffers_ BEFORE erasing it from the set (otherwise the
//     check was always-true). One ordering drift = silent corruption
//     (external buffers wrongly returned to the pool, then handed out
//     to a different alloc, causing aliasing).
//   - After R1: ownership routing is encoded as Origin on each Entry.
//     A single map lookup, single Origin check, single dispose action.
//     Invariant drift becomes structurally impossible.
// These tests pin the semantics so future refactors can't accidentally
// regress them.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <gtest/gtest.h>

#include <executorch/backends/metal/core/BufferRegistry.h>

#include <vector>

using executorch::backends::metal_v2::BufferRegistry;
using Origin = BufferRegistry::Origin;

namespace {

// Test fixture — gets a real MTLDevice and creates real MTLBuffers via
// newBufferWithBytes:length:options:. This is the only way to exercise
// retain-count semantics correctly; mocking MTLBuffer would defeat the
// purpose (Origin routing affects who calls release).
class BufferRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device_, nullptr) << "no Metal device on this machine";
  }
  void TearDown() override {
    [device_ release];
  }

  // Helper: create a real MTLBuffer of the given size with shared storage.
  // Returned at +1 retain (caller's responsibility to release or hand off).
  id<MTLBuffer> makeBuffer(size_t size) {
    id<MTLBuffer> buf = [device_ newBufferWithLength:size
                                             options:MTLResourceStorageModeShared];
    EXPECT_NE(buf, nil);
    return buf;
  }

  id<MTLDevice> device_ = nil;
};

//===----------------------------------------------------------------------===//
// Basic insert / find / contains / size / clear
//===----------------------------------------------------------------------===//

TEST_F(BufferRegistryTest, EmptyRegistryHasNoEntries) {
  BufferRegistry reg;
  EXPECT_EQ(reg.size(), 0u);
  EXPECT_EQ(reg.find(reinterpret_cast<void*>(0xdeadbeef)), nullptr);
  EXPECT_FALSE(reg.contains(reinterpret_cast<void*>(0xdeadbeef)));
  EXPECT_EQ(reg.findBuffer(reinterpret_cast<void*>(0xdeadbeef)), nil);
}

TEST_F(BufferRegistryTest, InsertThenFindRoundtrip) {
  BufferRegistry reg;
  id<MTLBuffer> buf = makeBuffer(64);
  void* ptr = reinterpret_cast<void*>(0x1000);

  reg.insert(ptr, buf, Origin::Pool, 64);

  EXPECT_EQ(reg.size(), 1u);
  EXPECT_TRUE(reg.contains(ptr));

  const auto* entry = reg.find(ptr);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->mtl, buf);
  EXPECT_EQ(entry->origin, Origin::Pool);
  EXPECT_EQ(entry->size, 64u);

  EXPECT_EQ(reg.findBuffer(ptr), buf);

  // Cleanup: remove transfers ownership; release once + release the original.
  auto removed = reg.remove(ptr);
  ASSERT_TRUE(removed.has_value());
  [removed->mtl release];
  [buf release];  // original +1 from makeBuffer
}

TEST_F(BufferRegistryTest, InsertNullPtrIsRejected) {
  BufferRegistry reg;
  id<MTLBuffer> buf = makeBuffer(16);
  reg.insert(nullptr, buf, Origin::Pool, 16);
  EXPECT_EQ(reg.size(), 0u);
  [buf release];
}

TEST_F(BufferRegistryTest, InsertNilBufferIsRejected) {
  BufferRegistry reg;
  void* ptr = reinterpret_cast<void*>(0x2000);
  reg.insert(ptr, nil, Origin::Pool, 16);
  EXPECT_EQ(reg.size(), 0u);
}

TEST_F(BufferRegistryTest, ClearReleasesAllEntries) {
  BufferRegistry reg;
  std::vector<id<MTLBuffer>> bufs;
  for (int i = 0; i < 5; ++i) {
    id<MTLBuffer> b = makeBuffer(16 * (i + 1));
    bufs.push_back(b);
    reg.insert(reinterpret_cast<void*>(0x3000 + i * 0x100), b,
               Origin::Pool, 16 * (i + 1));
  }
  EXPECT_EQ(reg.size(), 5u);
  reg.clear();
  EXPECT_EQ(reg.size(), 0u);
  for (auto b : bufs) [b release];
}

//===----------------------------------------------------------------------===//
// Ownership transfer via remove() — the contract that prevented the
// double-release segfault we hit during R1.
//===----------------------------------------------------------------------===//

TEST_F(BufferRegistryTest, RemoveTransfersOwnershipToCaller) {
  BufferRegistry reg;
  id<MTLBuffer> buf = makeBuffer(32);
  void* ptr = reinterpret_cast<void*>(0x4000);

  // Pre-insert retainCount: at least 1 (we hold +1 from makeBuffer).
  // After insert: registry retains too, so at least 2.
  // After remove: registry's retain goes to the returned Entry.mtl.
  reg.insert(ptr, buf, Origin::Pool, 32);

  auto removed = reg.remove(ptr);
  ASSERT_TRUE(removed.has_value());
  EXPECT_EQ(removed->mtl, buf);
  EXPECT_EQ(removed->origin, Origin::Pool);
  EXPECT_EQ(reg.size(), 0u);
  EXPECT_FALSE(reg.contains(ptr));

  // Caller is now responsible for the +1 retain in removed->mtl.
  // Releasing it (plus the original +1 from makeBuffer) brings refcount to 0.
  [removed->mtl release];
  [buf release];
}

TEST_F(BufferRegistryTest, RemoveOnUnknownPtrReturnsNullopt) {
  BufferRegistry reg;
  auto removed = reg.remove(reinterpret_cast<void*>(0xbadbad));
  EXPECT_FALSE(removed.has_value());
}

//===----------------------------------------------------------------------===//
// Origin routing — the structural fix for audit Finding C1.
// Different Origin values must produce different free() routing without
// any cross-set bookkeeping.
//===----------------------------------------------------------------------===//

TEST_F(BufferRegistryTest, OriginIsPreservedThroughRemove) {
  BufferRegistry reg;
  id<MTLBuffer> b1 = makeBuffer(16);
  id<MTLBuffer> b2 = makeBuffer(16);
  id<MTLBuffer> b3 = makeBuffer(16);

  reg.insert(reinterpret_cast<void*>(0x100), b1, Origin::Pool, 16);
  reg.insert(reinterpret_cast<void*>(0x200), b2, Origin::ExternalAliased, 16);
  reg.insert(reinterpret_cast<void*>(0x300), b3, Origin::ExternalCopied, 16);

  auto r1 = reg.remove(reinterpret_cast<void*>(0x100));
  auto r2 = reg.remove(reinterpret_cast<void*>(0x200));
  auto r3 = reg.remove(reinterpret_cast<void*>(0x300));
  ASSERT_TRUE(r1 && r2 && r3);
  EXPECT_EQ(r1->origin, Origin::Pool);
  EXPECT_EQ(r2->origin, Origin::ExternalAliased);
  EXPECT_EQ(r3->origin, Origin::ExternalCopied);

  [r1->mtl release];
  [r2->mtl release];
  [r3->mtl release];
  [b1 release];
  [b2 release];
  [b3 release];
}

//===----------------------------------------------------------------------===//
// refreshIfCopied — historical fix for the BinaryOps stack-local race
// (current ops snapshot via setBytes / setVectorBytes; refreshIfCopied is
// retained for true External buffer cases where caller mutates the source).
//===----------------------------------------------------------------------===//

TEST_F(BufferRegistryTest, RefreshUpdatesCopiedSnapshot) {
  BufferRegistry reg;
  // Shared-storage buffer so we can read [contents] from host side.
  id<MTLBuffer> buf = [device_ newBufferWithLength:32
                                           options:MTLResourceStorageModeShared];
  ASSERT_NE(buf, nil);

  int32_t source[4] = {1, 2, 3, 4};
  reg.insert(source, buf, Origin::ExternalCopied, sizeof(source));

  // Initially the buffer is whatever newBufferWithLength initialized it to.
  // refreshIfCopied should memcpy source into [buf contents].
  reg.refreshIfCopied(source, sizeof(source));

  int32_t* contents = reinterpret_cast<int32_t*>([buf contents]);
  EXPECT_EQ(contents[0], 1);
  EXPECT_EQ(contents[1], 2);
  EXPECT_EQ(contents[2], 3);
  EXPECT_EQ(contents[3], 4);

  // Mutate source and refresh again — the buffer must pick up the new values.
  source[0] = 100;
  source[1] = 200;
  source[2] = 300;
  source[3] = 400;
  reg.refreshIfCopied(source, sizeof(source));
  EXPECT_EQ(contents[0], 100);
  EXPECT_EQ(contents[1], 200);
  EXPECT_EQ(contents[2], 300);
  EXPECT_EQ(contents[3], 400);

  auto removed = reg.remove(source);
  [removed->mtl release];
  [buf release];
}

TEST_F(BufferRegistryTest, RefreshIsNoOpForNonCopiedOrigin) {
  BufferRegistry reg;
  id<MTLBuffer> buf = [device_ newBufferWithLength:16
                                           options:MTLResourceStorageModeShared];
  ASSERT_NE(buf, nil);

  // Pre-zero the buffer.
  memset([buf contents], 0, 16);

  int32_t source[4] = {7, 8, 9, 10};

  // Aliased and Pool entries should NOT be refreshed (no-op).
  reg.insert(source, buf, Origin::ExternalAliased, sizeof(source));
  reg.refreshIfCopied(source, sizeof(source));
  int32_t* contents = reinterpret_cast<int32_t*>([buf contents]);
  EXPECT_EQ(contents[0], 0);  // Should still be zero — no copy happened.

  auto r = reg.remove(source);
  [r->mtl release];
  [buf release];
}

TEST_F(BufferRegistryTest, RefreshOnUnregisteredPtrIsNoOp) {
  BufferRegistry reg;
  int32_t source[4] = {1, 2, 3, 4};
  // Should not crash, should not insert anything.
  reg.refreshIfCopied(source, sizeof(source));
  EXPECT_EQ(reg.size(), 0u);
}

TEST_F(BufferRegistryTest, RefreshHandlesZeroBytesGracefully) {
  BufferRegistry reg;
  id<MTLBuffer> buf = [device_ newBufferWithLength:16
                                           options:MTLResourceStorageModeShared];
  int32_t source[4] = {1, 2, 3, 4};
  reg.insert(source, buf, Origin::ExternalCopied, sizeof(source));
  reg.refreshIfCopied(source, 0);  // zero bytes — should no-op
  auto r = reg.remove(source);
  [r->mtl release];
  [buf release];
}

TEST_F(BufferRegistryTest, RefreshDeathOnOversizedRefresh) {
  // refreshIfCopied now hard-fails when caller asks to refresh
  // more bytes than the entry was registered with. Previously it
  // silently truncated, leaving the kernel that thinks it has `size`
  // bytes to read stale / garbage data past entry.size.
  BufferRegistry reg;
  id<MTLBuffer> buf = [device_ newBufferWithLength:8
                                           options:MTLResourceStorageModeShared];
  memset([buf contents], 0, 8);

  int32_t source[4] = {1, 2, 3, 4};  // 16 bytes
  reg.insert(source, buf, Origin::ExternalCopied, 8);

  // size > entry.size → ET_CHECK fails. EXPECT_DEATH catches the abort.
  EXPECT_DEATH({ reg.refreshIfCopied(source, 16); }, ".*");

  // Cleanup happens via the death-test child process; no need to remove
  // here since the parent process never executed the failing call.
  // But we still need to free the registry's retain in the parent.
  auto r = reg.remove(source);
  [r->mtl release];
  [buf release];
}

TEST_F(BufferRegistryTest, RefreshAcceptsExactAndSmallerSizes) {
  // The contract permits size <= entry.size: the caller asked to refresh
  // a prefix of the registered region.
  BufferRegistry reg;
  id<MTLBuffer> buf = [device_ newBufferWithLength:16
                                           options:MTLResourceStorageModeShared];
  memset([buf contents], 0, 16);

  int32_t source[4] = {10, 20, 30, 40};
  reg.insert(source, buf, Origin::ExternalCopied, 16);

  reg.refreshIfCopied(source, 16);  // exact size — OK
  int32_t* contents = reinterpret_cast<int32_t*>([buf contents]);
  EXPECT_EQ(contents[0], 10);
  EXPECT_EQ(contents[3], 40);

  source[0] = 99;
  reg.refreshIfCopied(source, 8);  // smaller — refresh first 2 ints only
  EXPECT_EQ(contents[0], 99);
  EXPECT_EQ(contents[1], 20);
  EXPECT_EQ(contents[2], 30);  // unchanged from prior refresh
  EXPECT_EQ(contents[3], 40);

  auto r = reg.remove(source);
  [r->mtl release];
  [buf release];
}

//===----------------------------------------------------------------------===//
// forEach iteration — historical helper, retained for residency / debug use.
//===----------------------------------------------------------------------===//

TEST_F(BufferRegistryTest, ForEachVisitsAllEntries) {
  BufferRegistry reg;
  std::vector<id<MTLBuffer>> bufs;
  std::vector<void*> ptrs;
  for (int i = 0; i < 4; ++i) {
    id<MTLBuffer> b = makeBuffer(16);
    bufs.push_back(b);
    void* p = reinterpret_cast<void*>(0x5000 + i * 0x100);
    ptrs.push_back(p);
    reg.insert(p, b, Origin::Pool, 16);
  }

  int visit_count = 0;
  std::vector<void*> seen_ptrs;
  reg.forEach([&](void* ptr, const BufferRegistry::Entry& e) {
    visit_count++;
    seen_ptrs.push_back(ptr);
    EXPECT_NE(e.mtl, nil);
    EXPECT_EQ(e.origin, Origin::Pool);
    EXPECT_EQ(e.size, 16u);
  });
  EXPECT_EQ(visit_count, 4);
  // Order is unordered_map's hash order — just verify all 4 ptrs were seen.
  for (void* p : ptrs) {
    EXPECT_NE(std::find(seen_ptrs.begin(), seen_ptrs.end(), p),
              seen_ptrs.end());
  }

  reg.clear();
  for (auto b : bufs) [b release];
}

TEST_F(BufferRegistryTest, ForEachOnEmptyRegistryIsNoOp) {
  BufferRegistry reg;
  int visit_count = 0;
  reg.forEach([&](void*, const BufferRegistry::Entry&) { visit_count++; });
  EXPECT_EQ(visit_count, 0);
}

//===----------------------------------------------------------------------===//
// Combined scenario — mimics the actual MetalStream usage pattern that
// triggered audit Finding C1. Verifies that mixing Pool + External
// entries produces correct routing on remove().
//===----------------------------------------------------------------------===//

TEST_F(BufferRegistryTest, MixedOriginsRouteIndependently) {
  BufferRegistry reg;

  id<MTLBuffer> pool_buf = makeBuffer(64);
  id<MTLBuffer> ext_buf = makeBuffer(64);
  void* pool_ptr = reinterpret_cast<void*>(0x6000);
  void* ext_ptr = reinterpret_cast<void*>(0x7000);

  reg.insert(pool_ptr, pool_buf, Origin::Pool, 64);
  reg.insert(ext_ptr, ext_buf, Origin::ExternalAliased, 64);
  EXPECT_EQ(reg.size(), 2u);

  // Remove the external entry first. This must NOT affect the Pool entry's
  // routing (the bug from C1: shared sets meant erasing one could perturb
  // the other's invariant).
  auto removed_ext = reg.remove(ext_ptr);
  ASSERT_TRUE(removed_ext.has_value());
  EXPECT_EQ(removed_ext->origin, Origin::ExternalAliased);
  EXPECT_EQ(reg.size(), 1u);

  // Pool entry should still be findable + still tagged Pool.
  const auto* pool_entry = reg.find(pool_ptr);
  ASSERT_NE(pool_entry, nullptr);
  EXPECT_EQ(pool_entry->origin, Origin::Pool);

  // Now remove the pool entry.
  auto removed_pool = reg.remove(pool_ptr);
  ASSERT_TRUE(removed_pool.has_value());
  EXPECT_EQ(removed_pool->origin, Origin::Pool);
  EXPECT_EQ(reg.size(), 0u);

  [removed_ext->mtl release];
  [removed_pool->mtl release];
  [pool_buf release];
  [ext_buf release];
}

} // namespace
