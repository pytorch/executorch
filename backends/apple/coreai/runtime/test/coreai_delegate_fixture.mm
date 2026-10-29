/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_delegate_fixture.h"

namespace executorch::backends::coreai::testing {
using namespace executorch::runtime;

DelegateFixture::DelegateFixture(
    const BackendInterface* backend,
    ScopedFakeBridgeState& bridge,
    NSDictionary* manifest)
    : json(encode(manifest)),
      processed(
          json.bytes,
          json.length,
          [](void* context, void*, size_t) { ++*static_cast<int*>(context); },
          &freed),
      backend_(backend),
      bridge_(bridge) {}

DelegateFixture::~DelegateFixture() {
  reset();
  EXPECT_TRUE(bridge_.wait_for_callbacks(
      dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)));
  processed.Free();
  if (json != nil) {
    EXPECT_EQ(freed, 1);
  }
  EXPECT_EQ(data.requests.load(), data.releases.load());
}

Error DelegateFixture::init(NSString* root) {
  if (storage.url == nil || json == nil) {
    return Error::Internal;
  }
  BackendOptions<1> options;
  const Error error = options.set_option(
      "coreai_assets_dir", (root ?: storage.url.path).UTF8String);
  if (error != Error::Ok) {
    return error;
  }
  return init_with_options({options.view().data(), options.view().size()});
}

Error DelegateFixture::init_with_options(
    Span<const BackendOption> options,
    MemoryAllocator* allocator) {
  reset();
  BackendInitContext context(allocator, nullptr, "forward", &data, options);
  auto result = backend_->init(context, &processed, {});
  if (result.ok()) {
    handle = result.get();
  }
  auto drained = bridge_.wait_for_callbacks(
      dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
  if (!drained) {
    ADD_FAILURE() << drained.message();
    reset();
    return Error::Internal;
  }
  return result.ok() ? Error::Ok : result.error();
}

void DelegateFixture::reset() {
  if (handle != nullptr) {
    backend_->destroy(handle);
  }
  handle = nullptr;
}

Error DelegateFixture::execute(Span<EValue*> args) {
  BackendExecutionContext context;
  return backend_->execute(context, handle, args);
}

NSMutableDictionary* multiple_io_manifest() {
  NSMutableDictionary* manifest = manifest_dict();
  manifest[@"input_names"] = @[ @"input_1", @"input_0" ];
  manifest[@"output_names"] = @[ @"output_1", @"output_0" ];
  return manifest;
}

} // namespace executorch::backends::coreai::testing
