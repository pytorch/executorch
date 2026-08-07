/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <gtest/gtest.h>

using executorch::backends::webgpu::WebGPUGraph;

namespace {

WebGPUGraph::SliceChain armed_chain() {
  WebGPUGraph::SliceChain chain;
  chain.valid = true;
  chain.out_id = 7;
  chain.dispatch_idx = 3;
  chain.in_buffer = reinterpret_cast<WGPUBuffer>(0x1000);
  chain.in_nbytes = 64;
  chain.out_buffer = reinterpret_cast<WGPUBuffer>(0x2000);
  chain.out_nbytes = 64;
  chain.out_meta_buf = reinterpret_cast<WGPUBuffer>(0x3000);
  chain.in_meta_buf = reinterpret_cast<WGPUBuffer>(0x4000);
  chain.params_buf = reinterpret_cast<WGPUBuffer>(0x5000);
  return chain;
}

// A fresh graph must never observe another graph's offered chain. Before the
// chain became graph-instance state it lived in a function-local `static`, so
// two graphs shared one dispatch index and one set of buffer handles.
TEST(SliceChainLifecycle, SeparateGraphsDoNotShareState) {
  WebGPUGraph first;
  WebGPUGraph second;

  first.offer_slice_chain(armed_chain());

  EXPECT_TRUE(first.slice_chain().valid);
  EXPECT_FALSE(second.slice_chain().valid);
  EXPECT_EQ(second.slice_chain().out_id, -1);
  EXPECT_EQ(second.slice_chain().in_buffer, nullptr);
  EXPECT_EQ(second.slice_chain().out_buffer, nullptr);
}

// Interleaved offers must stay independent: arming the second graph must not
// disturb the first, and clearing one must not clear the other.
TEST(SliceChainLifecycle, InterleavedGraphsAreIndependent) {
  WebGPUGraph first;
  WebGPUGraph second;

  first.offer_slice_chain(armed_chain());
  WebGPUGraph::SliceChain other = armed_chain();
  other.out_id = 11;
  other.dispatch_idx = 5;
  second.offer_slice_chain(other);

  EXPECT_EQ(first.slice_chain().out_id, 7);
  EXPECT_EQ(second.slice_chain().out_id, 11);

  first.clear_slice_chain();
  EXPECT_FALSE(first.slice_chain().valid);
  EXPECT_TRUE(second.slice_chain().valid);
  EXPECT_EQ(second.slice_chain().dispatch_idx, 5u);
}

// A build that fails before completing must not leave a chain armed for the
// next build on the same graph. build() clears on entry and again via its
// scope guard; with no WebGPU context it fails early, which is the cheapest
// reachable failure path.
TEST(SliceChainLifecycle, FailedBuildClearsState) {
  WebGPUGraph graph;
  graph.offer_slice_chain(armed_chain());
  ASSERT_TRUE(graph.slice_chain().valid);

  try {
    graph.build(nullptr, nullptr, 0, nullptr, {});
  } catch (...) {
    // A build failure is expected here; the contract under test is the state.
  }

  EXPECT_FALSE(graph.slice_chain().valid);
  EXPECT_EQ(graph.slice_chain().out_id, -1);
  EXPECT_EQ(graph.slice_chain().dispatch_idx, 0u);
}

// Clearing must null every handle, so a later slice cannot re-bind a dispatch
// against buffers owned by a previous build.
TEST(SliceChainLifecycle, ClearDropsEveryStaleHandle) {
  WebGPUGraph graph;
  graph.offer_slice_chain(armed_chain());
  graph.clear_slice_chain();

  const WebGPUGraph::SliceChain& chain = graph.slice_chain();
  EXPECT_FALSE(chain.valid);
  EXPECT_EQ(chain.in_buffer, nullptr);
  EXPECT_EQ(chain.out_buffer, nullptr);
  EXPECT_EQ(chain.out_meta_buf, nullptr);
  EXPECT_EQ(chain.in_meta_buf, nullptr);
  EXPECT_EQ(chain.params_buf, nullptr);
  EXPECT_EQ(chain.in_nbytes, 0u);
  EXPECT_EQ(chain.out_nbytes, 0u);
}

// Destroying a graph must not leave anything for the next graph to claim: a
// newly constructed graph always starts disarmed.
TEST(SliceChainLifecycle, DestructionLeavesNoResidueForTheNextGraph) {
  {
    WebGPUGraph doomed;
    doomed.offer_slice_chain(armed_chain());
    ASSERT_TRUE(doomed.slice_chain().valid);
  }

  WebGPUGraph fresh;
  EXPECT_FALSE(fresh.slice_chain().valid);
  EXPECT_EQ(fresh.slice_chain().out_id, -1);
  EXPECT_EQ(fresh.slice_chain().params_buf, nullptr);
}

} // namespace
