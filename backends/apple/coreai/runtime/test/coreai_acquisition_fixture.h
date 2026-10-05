/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "coreai_bookmark_fixture.h"
#include "coreai_fake_loader.h"
#include "coreai_source_fixture.h"
#include "coreai_storage.h"

namespace executorch::backends::coreai::testing {

runtime::Result<NSURL*> acquisition_staged_url(
    NSURL* root,
    const Manifest& manifest);

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
    const Manifest& first,
    const Manifest& second,
    TestData& data,
    bool same_key);

// Uses the existing bounded, scoped child process owner.
void set_acquisition_test_executable(const char* path);
::testing::AssertionResult spawn_acquisition_crash(
    BookmarkChild& child,
    NSURL* root,
    StorageOperation boundary,
    NSDictionary* manifest);
::testing::AssertionResult spawn_parallel_acquisition(
    BookmarkChild& child,
    NSURL* root,
    NSDictionary* first,
    NSDictionary* second,
    bool same_key);
// Dispatch before bookmark_child_mode and InitGoogleTest. Unrelated modes
// return -1.
int acquisition_child_mode(int argc, char** argv);

} // namespace executorch::backends::coreai::testing
