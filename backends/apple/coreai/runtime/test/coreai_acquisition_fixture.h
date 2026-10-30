/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "coreai_cache_entry_fixture.h"
#include "coreai_fake_loader.h"
#include "coreai_fault_injection.h"
#include "coreai_load_coordinator.h"
#include "coreai_source_fixture.h"

namespace executorch::backends::coreai::testing {

runtime::Result<NSURL*> acquisition_staged_url(
    NSURL* root,
    const SelectedBundle& bundle);
runtime::Result<id<ETCoreAIPreparedModel>> acquire_test_model(
    const SelectedHostBundle& bundle,
    const runtime::NamedDataMap* data,
    NSString* cache_path,
    id<ETCoreAIModelLoading> loader);
runtime::Error clear_test_cache_entry(
    NSString* cache_path,
    NSString* key,
    id<ETCoreAIModelLoading> loader);

struct ParallelAcquisitionObservation {
  int creation[2] = {-1, -1};
  int joined[2] = {-1, -1};
  bool completed[2] = {false, false};
  runtime::Error errors[2] = {
      runtime::Error::Internal,
      runtime::Error::Internal};
  bool models[2] = {false, false};
  bool first_entered = false;
  bool second_observed = false;
  int gate_timeouts = 0;
  bool callbacks_drained = false;
};
ParallelAcquisitionObservation parallel_acquisitions(
    ScopedFakeBridgeState& bridge,
    BookmarkFakeLoader* loader,
    NSURL* root,
    const SelectedBundle& first,
    const SelectedBundle& second,
    TestData& data,
    bool same_key);

::testing::AssertionResult spawn_acquisition_crash(
    BookmarkChild& child,
    NSURL* root,
    FaultPoint boundary,
    NSDictionary* manifest);
::testing::AssertionResult spawn_parallel_acquisition(
    BookmarkChild& child,
    NSURL* root,
    NSDictionary* first,
    NSDictionary* second,
    bool same_key);
// Dispatch before InitGoogleTest. Unrelated modes return -1.
int acquisition_child_mode(int argc, char** argv);

} // namespace executorch::backends::coreai::testing
