/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <gtest/gtest.h>
#include <pthread.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "coreai_bookmark_fixture.h"
#import "coreai_fake_loader.h"
#include "coreai_source_fixture.h"

namespace executorch::backends::coreai::testing {
namespace {
using namespace executorch::runtime;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;

class CoreAIDelegateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend = get_backend_class("CoreAIBackend");
    ASSERT_NE(backend, nullptr);
  }
  ScopedFakeBridgeState bridge;
  const BackendInterface* backend = nullptr;
};

// Keep the caller-owned manifest alive across failed initialization attempts.
class DelegateFixture final {
 public:
  DelegateFixture(const BackendInterface* backend, ScopedFakeBridgeState& bridge,
                  NSDictionary* manifest = manifest_dict())
      : json(encode(manifest)),
        processed(
            json.bytes, json.length,
            [](void* context, void*, size_t) { ++*static_cast<int*>(context); }, &freed),
        backend_(backend),
        bridge_(bridge) {}

  ~DelegateFixture() {
    reset();
    EXPECT_TRUE(bridge_.wait_for_callbacks(dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)));
    processed.Free();
    if (json != nil) EXPECT_EQ(freed, 1);
    EXPECT_EQ(data.requests.load(), data.releases.load());
  }
  DelegateFixture(const DelegateFixture&) = delete;
  DelegateFixture& operator=(const DelegateFixture&) = delete;

  Error init(NSString* root = nil) {
    if (storage.url == nil || json == nil) return Error::Internal;
    BackendOptions<1> options;
    auto error = options.set_option("coreai_assets_dir", (root ?: storage.url.path).UTF8String);
    if (error != Error::Ok) return error;
    return init_with_options({options.view().data(), options.view().size()});
  }

  Error init_with_options(Span<const BackendOption> options, MemoryAllocator* allocator = nullptr) {
    reset();
    BackendInitContext context(allocator, nullptr, "forward", &data, options);
    auto result = backend_->init(context, &processed, {});
    if (!result.ok()) return result.error();
    handle = result.get();
    return Error::Ok;
  }

  void reset() {
    if (handle != nullptr) backend_->destroy(handle);
    handle = nullptr;
  }

  Error execute(Span<EValue*> args) {
    BackendExecutionContext context;
    return backend_->execute(context, handle, args);
  }

  TestData data;
  BookmarkDirectory storage;
  int freed = 0;
  NSData* json;
  FreeableBuffer processed;
  DelegateHandle* handle = nullptr;

 private:
  const BackendInterface* backend_;
  ScopedFakeBridgeState& bridge_;
};

struct FloatTensors {
  float input[2] = {3, 7};
  float output[2] = {-1, -1};
  TensorImpl::SizesType input_size[1] = {2}, output_size[1] = {2};
  TensorImpl::DimOrderType order[1] = {0};
  TensorImpl::StridesType input_stride[1] = {1}, output_stride[1] = {1};
  TensorImpl input_impl{ScalarType::Float, 1, input_size, input, order, input_stride};
  TensorImpl output_impl{ScalarType::Float,
                         1,
                         output_size,
                         output,
                         order,
                         output_stride,
                         TensorShapeDynamism::DYNAMIC_BOUND};
  EValue in{Tensor(&input_impl)}, out{Tensor(&output_impl)};
  EValue* args[2] = {&in, &out};
};

template <typename T>
struct RangeTensors {
  static constexpr ScalarType dtype =
      std::is_same_v<T, float> ? ScalarType::Float : ScalarType::Half;
  T input[12] = {}, output[12] = {};
  TensorImpl::SizesType input_shape[2] = {4, 2}, output_shape[2] = {4, 2};
  TensorImpl::DimOrderType order[2] = {0, 1};
  TensorImpl::StridesType input_strides[2] = {2, 1}, output_strides[2] = {2, 1};
  TensorImpl input_impl{
      dtype, 2, input_shape, input + 2, order, input_strides, TensorShapeDynamism::DYNAMIC_BOUND};
  TensorImpl output_impl{dtype,
                         2,
                         output_shape,
                         output + 2,
                         order,
                         output_strides,
                         TensorShapeDynamism::DYNAMIC_BOUND};
  EValue in{Tensor(&input_impl)}, out{Tensor(&output_impl)};
  EValue* args[2] = {&in, &out};

  RangeTensors() {
    for (size_t i = 0; i < 12; ++i) input[i] = static_cast<T>(i + 1);
    fill_output();
  }
  void fill_output() { std::fill_n(output, 12, static_cast<T>(99)); }
};

// All preflight rows must fail before reading payloads or creating storage.
void expect_preflight_rejection(const BackendInterface* backend, ScopedFakeBridgeState& bridge,
                                NSDictionary* manifest, Error expected) {
  DelegateFixture fixture(backend, bridge, manifest);
  ASSERT_NE(fixture.storage.url, nil);
  NSURL* absent = [fixture.storage.url URLByAppendingPathComponent:@"uncreated/models"];
  NSString* previous_bundle = bridge.state().last_bundle;
  EXPECT_EQ(fixture.init(absent.path), expected);
  EXPECT_EQ(fixture.data.attempts.load(), 0);
  EXPECT_EQ(fixture.data.metadata_requests.load(), 0);
  EXPECT_EQ(fixture.freed, 0);
  EXPECT_EQ(bridge.state().last_bundle, previous_bundle);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:absent.path]);
  NSError* error = nil;
  NSArray* children =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:fixture.storage.url.path
                                                        error:&error];
  EXPECT_NE(children, nil);
  EXPECT_EQ(error, nil);
  EXPECT_EQ(children.count, 0u);
}

TEST_F(CoreAIDelegateTest, AvailabilityRejectsWithoutTakingManifestOwnership) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    bridge.state().available = false;
    EXPECT_FALSE(backend->is_available());
    EXPECT_EQ(fixture.init(), Error::NotSupported);
    EXPECT_EQ(fixture.freed, 0);
    EXPECT_EQ(fixture.data.attempts.load(), 0);
    EXPECT_EQ(bridge.state().sessions.load(), 0);
    bridge.state().available = true;
    EXPECT_TRUE(backend->is_available());
    ASSERT_EQ(fixture.init(), Error::Ok);
    EXPECT_EQ(fixture.freed, 1);
    EXPECT_EQ(bridge.state().sessions.load(), 1);
  }
}

TEST_F(CoreAIDelegateTest, LoaderFactoryFailureLeavesManifestAndStorageUntouched) {
  @autoreleasepool {
    bridge.state().fail_loader_factory = true;
    expect_preflight_rejection(backend, bridge, manifest_dict(), Error::Internal);
    EXPECT_EQ(bridge.state().loaders.load(), 0);
    EXPECT_EQ(bridge.state().sessions.load(), 0);
  }
}

TEST_F(CoreAIDelegateTest, PreflightRejectsBeforePayloadReadsOrStorageCreation) {
  @autoreleasepool {
    auto missing_function = aot_manifest_dict();
    [missing_function removeObjectForKey:@"function"];
    auto numeric_platform = aot_manifest_dict();
    numeric_platform[@"platform"] = @1;
    auto escaping_arch = aot_manifest_dict();
    NSMutableDictionary* archs = [escaping_arch[@"archs"] mutableCopy];
    archs[@"arch_a"] = @"../model.arch_a.aimodelc";
    escaping_arch[@"archs"] = archs;
    auto escaping_file = aot_manifest_dict();
    escaping_file[@"files"] = @{@"model.arch_a.aimodelc/../escape" : @1};
    auto ios = aot_manifest_dict();
    ios[@"platform"] = @"iOS";
    const struct {
      const char* name;
      NSMutableDictionary* manifest;
      NSString* device_architecture;
      Error expected;
    } rows[] = {
        {"missing function", missing_function, @"arch_b", Error::InvalidProgram},
        {"non-string platform", numeric_platform, @"arch_b", Error::InvalidProgram},
        {"escaping architecture path", escaping_arch, @"arch_b", Error::InvalidProgram},
        {"escaping file entry", escaping_file, @"arch_b", Error::InvalidProgram},
        {"incompatible platform", ios, @"arch_b", Error::DelegateInvalidCompatibility},
        {"missing device architecture", aot_manifest_dict(), nil,
         Error::DelegateInvalidCompatibility},
    };
    for (const auto& row : rows) {
      SCOPED_TRACE(row.name);
      bridge.state().device_architecture = row.device_architecture;
      ASSERT_NO_FATAL_FAILURE(
          expect_preflight_rejection(backend, bridge, row.manifest, row.expected));
    }
    bridge.state().device_architecture = @"arch_b";
  }
}

TEST_F(CoreAIDelegateTest, PreflightRejectsFutureDeploymentForSourceAndAot) {
  @autoreleasepool {
    for (bool aot : {false, true}) {
      SCOPED_TRACE(aot ? "AOT" : "source");
      auto dict = aot ? aot_manifest_dict() : manifest_dict();
      dict[@"min_deployment_version"] = @"999999.0";
      expect_preflight_rejection(backend, bridge, dict, Error::DelegateInvalidCompatibility);
    }
  }
}

TEST_F(CoreAIDelegateTest, AotInitPassesOnlySelectedArchitectureBundleToSdk) {
  @autoreleasepool {
    NSString* arch = @"arch_a";
    DelegateFixture fixture(backend, bridge, aot_manifest_dict());
    aot_data(fixture.data, arch);
    bridge.state().device_architecture = arch;
    auto parsed = parse_manifest(fixture.json);
    ASSERT_TRUE(parsed.ok());
    auto selected = select_assets(parsed.get(), arch, @"macOS");
    ASSERT_TRUE(selected.ok());
    ASSERT_EQ(fixture.init(), Error::Ok);
    auto key = bookmark_key(selected.get(), @"macOS", arch);
    ASSERT_TRUE(key.ok());
    NSString* expected = [fixture.storage.url.path
        stringByAppendingPathComponent:[NSString stringWithFormat:@"staging/%@/%@", key.get(),
                                                                  selected->path.lastPathComponent]];
    EXPECT_TRUE([bridge.state().last_bundle isEqualToString:expected]);
    fixture.reset();
    EXPECT_EQ(bridge.state().sessions.load(), 0);
    EXPECT_EQ(fixture.data.attempts.load(), 2);
    EXPECT_EQ(fixture.data.releases.load(), fixture.data.requests.load());
  }
}

TEST_F(CoreAIDelegateTest, InitBindingFailurePreservesManifestAndStagedSource) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    bridge.state().fail_load = true;
    EXPECT_EQ(fixture.init(), Error::InvalidProgram);
    EXPECT_EQ(fixture.freed, 0);
    EXPECT_EQ(bridge.state().sessions.load(), 0);
    ASSERT_NE(bridge.state().last_bundle, nil);
    EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:bridge.state().last_bundle]);
    bridge.state().fail_load = false;
    ASSERT_EQ(fixture.init(), Error::Ok);
    EXPECT_EQ(fixture.freed, 1);
    EXPECT_EQ(bridge.state().sessions.load(), 1);
    EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:bridge.state().last_bundle]);
    EXPECT_TRUE([bridge.state().last_bundle
        hasPrefix:[fixture.storage.url.path stringByAppendingString:@"/"]]);
  }
  EXPECT_EQ(bridge.state().sessions.load(), 0);
}

TEST_F(CoreAIDelegateTest, OrderedIoBindingUsesManifestFunctionAndNames) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    EXPECT_EQ(bridge.state().binding_mismatches.load(), 0);
    FloatTensors tensors;
    ASSERT_EQ(fixture.execute({tensors.args, 2}), Error::Ok);
    EXPECT_EQ(tensors.output[0], 3);
    EXPECT_EQ(tensors.output[1], 7);
    EXPECT_EQ(bridge.state().last_input_bytes.load(), tensors.input);
    EXPECT_EQ(bridge.state().executions.load(), 1);
  }
}

TEST_F(CoreAIDelegateTest, ArgumentValidationRejectsBeforeSdkExecution) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    EXPECT_EQ(fixture.execute({tensors.args, 1}), Error::InvalidArgument);
    EValue scalar(int64_t(3));
    EValue* args[] = {&scalar, &tensors.out};
    EXPECT_EQ(fixture.execute({args, 2}), Error::InvalidArgument);
    EXPECT_EQ(bridge.state().executions.load(), 0);
  }
}

// No assertion may leave a caller thread blocked behind its input-read gate.
class BorrowedInputWorker final {
 public:
  BorrowedInputWorker(DelegateFixture& fixture, FloatTensors& tensors, FakeBridgeState& state)
      : fixture_(fixture), tensors_(tensors), state_(state) {
    state_.input_ready = dispatch_semaphore_create(0);
    state_.allow_input_read = dispatch_semaphore_create(0);
  }
  ~BorrowedInputWorker() {
    release_and_join();
    state_.input_ready = nil;
    state_.allow_input_read = nil;
  }
  BorrowedInputWorker(const BorrowedInputWorker&) = delete;
  BorrowedInputWorker& operator=(const BorrowedInputWorker&) = delete;
  int start() {
    const int error = pthread_create(&thread_, nullptr, run, this);
    started_ = error == 0;
    return error;
  }
  long wait_ready() {
    return dispatch_semaphore_wait(state_.input_ready,
                                   dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
  }
  long wait_finished(int64_t timeout) {
    return dispatch_semaphore_wait(finished_, dispatch_time(DISPATCH_TIME_NOW, timeout));
  }
  void release() { dispatch_semaphore_signal(state_.allow_input_read); }
  void release_and_join() {
    release();
    if (started_) {
      const long returned =
          dispatch_semaphore_wait(returned_, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
      if (returned != 0) {
        ADD_FAILURE() << "Delegate worker did not return after releasing its input gate";
        // A stuck worker still borrows this stack; unwinding would be unsafe.
        std::_Exit(EXIT_FAILURE);
      }
      EXPECT_EQ(pthread_join(thread_, nullptr), 0);
      started_ = false;
    }
  }
  Error status = Error::Internal;

 private:
  static void* run(void* context) {
    auto& worker = *static_cast<BorrowedInputWorker*>(context);
    @autoreleasepool {
      worker.status = worker.fixture_.execute({worker.tensors_.args, 2});
    }
    dispatch_semaphore_signal(worker.finished_);
    dispatch_semaphore_signal(worker.returned_);
    return nullptr;
  }
  DelegateFixture& fixture_;
  FloatTensors& tensors_;
  FakeBridgeState& state_;
  dispatch_semaphore_t finished_ = dispatch_semaphore_create(0);
  dispatch_semaphore_t returned_ = dispatch_semaphore_create(0);
  pthread_t thread_{};
  bool started_ = false;
};

TEST_F(CoreAIDelegateTest, BorrowedInputStorageStaysLiveUntilSuccessOrError) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    for (bool failure : {false, true}) {
      SCOPED_TRACE(failure ? "runtime error" : "success");
      bridge.state().fail_execute = failure;
      BorrowedInputWorker worker(fixture, tensors, bridge.state());
      const int started = worker.start();
      if (started != 0) {
        worker.release_and_join();
        FAIL() << "pthread_create failed: " << started;
      }
      const long ready = worker.wait_ready();
      const void* observed = bridge.state().last_input_bytes.load();
      const long premature = worker.wait_finished(50 * NSEC_PER_MSEC);
      worker.release();
      const long finished = worker.wait_finished(5 * NSEC_PER_SEC);
      worker.release_and_join();
      ASSERT_TRUE(bridge.wait_for_callbacks(dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)));
      EXPECT_EQ(ready, 0);
      EXPECT_EQ(observed, tensors.input);
      EXPECT_NE(premature, 0);
      EXPECT_EQ(finished, 0);
      EXPECT_EQ(worker.status, failure ? Error::Internal : Error::Ok);
      EXPECT_EQ(bridge.state().last_input_first_byte.load(),
                reinterpret_cast<const unsigned char*>(tensors.input)[0]);
    }
  }
}

TEST_F(CoreAIDelegateTest, ExecuteMapsRuntimeErrors) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    bridge.state().fail_execute = true;
    EXPECT_EQ(fixture.execute({tensors.args, 2}), Error::Internal);
  }
}

TEST_F(CoreAIDelegateTest, MalformedOutputDoesNotWriteCallerStorage) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    bridge.state().bad_output = true;
    EXPECT_EQ(fixture.execute({tensors.args, 2}), Error::InvalidExternalData);
    EXPECT_EQ(tensors.output[0], -1);
    EXPECT_EQ(tensors.output[1], -1);
  }
}

TEST_F(CoreAIDelegateTest, NoncontiguousInputIsNotSupported) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    tensors.input_stride[0] = 2;
    EXPECT_EQ(fixture.execute({tensors.args, 2}), Error::NotSupported);
    EXPECT_EQ(bridge.state().executions.load(), 0);
  }
}

TEST_F(CoreAIDelegateTest, NonemptyOutputRequiresStorage) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    tensors.output_impl.set_data(nullptr);
    EXPECT_EQ(fixture.execute({tensors.args, 2}), Error::InvalidArgument);
  }
}

TEST_F(CoreAIDelegateTest, OutputResizesDownAndBackWithinCapacity) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    TensorImpl::SizesType one[] = {1};
    TensorImpl smaller(ScalarType::Float, 1, one, tensors.input, tensors.order,
                       tensors.input_stride);
    EValue small{Tensor(&smaller)};
    EValue* args[] = {&small, &tensors.out};
    ASSERT_EQ(fixture.execute({args, 2}), Error::Ok);
    EXPECT_EQ(tensors.out.toTensor().size(0), 1);
    ASSERT_EQ(fixture.execute({tensors.args, 2}), Error::Ok);
    EXPECT_EQ(tensors.out.toTensor().size(0), 2);
  }
}

TEST_F(CoreAIDelegateTest, OutputResizeFailurePreservesStorage) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    TensorImpl::SizesType capacity[] = {1};
    TensorImpl output(ScalarType::Float, 1, capacity, tensors.output, tensors.order,
                      tensors.output_stride, TensorShapeDynamism::DYNAMIC_BOUND);
    EValue too_small{Tensor(&output)};
    EValue* args[] = {&tensors.in, &too_small};
    tensors.output[0] = -2;
    EXPECT_NE(fixture.execute({args, 2}), Error::Ok);
    EXPECT_EQ(tensors.output[0], -2);
  }
}

TEST_F(CoreAIDelegateTest, HalfDtypePreservesBitsAndRejectsMixedOutputDtype) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    uint16_t input[] = {0x3c00, 0x4000}, output[] = {0, 0};
    TensorImpl input_impl(ScalarType::Half, 1, tensors.input_size, input, tensors.order,
                          tensors.input_stride);
    TensorImpl output_impl(ScalarType::Half, 1, tensors.output_size, output, tensors.order,
                           tensors.output_stride);
    EValue in{Tensor(&input_impl)}, out{Tensor(&output_impl)};
    EValue* args[] = {&in, &out};
    ASSERT_EQ(fixture.execute({args, 2}), Error::Ok);
    EXPECT_EQ(output[0], input[0]);
    EXPECT_EQ(output[1], input[1]);
    EXPECT_EQ(bridge.state().last_input_bytes.load(), input);
    EValue* mixed[] = {&in, &tensors.out};
    EXPECT_EQ(fixture.execute({mixed, 2}), Error::InvalidExternalData);
  }
}

TEST_F(CoreAIDelegateTest, ScalarTensorsExecuteAndRejectOutputRankMismatch) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    TensorImpl input(ScalarType::Float, 0, nullptr, tensors.input);
    TensorImpl output(ScalarType::Float, 0, nullptr, tensors.output);
    EValue in{Tensor(&input)}, out{Tensor(&output)};
    EValue* args[] = {&in, &out};
    ASSERT_EQ(fixture.execute({args, 2}), Error::Ok);
    EXPECT_EQ(tensors.output[0], tensors.input[0]);
    EXPECT_EQ(bridge.state().last_input_bytes.load(), tensors.input);
    EValue* wrong_rank[] = {&tensors.in, &out};
    EXPECT_EQ(fixture.execute({wrong_rank, 2}), Error::InvalidExternalData);
  }
}

TEST_F(CoreAIDelegateTest, EmptyVectorAndMatrixBorrowNullStorage) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    for (int rank : {1, 2}) {
      SCOPED_TRACE(rank);
      TensorImpl::SizesType shape[] = {2, 0};
      TensorImpl::DimOrderType order[] = {0, 1};
      TensorImpl::StridesType strides[] = {1, 1};
      auto* sizes = rank == 1 ? shape + 1 : shape;
      TensorImpl input(ScalarType::Float, rank, sizes, nullptr, order, strides);
      TensorImpl output(ScalarType::Float, rank, sizes, nullptr, order, strides);
      EValue in{Tensor(&input)}, out{Tensor(&output)};
      EValue* args[] = {&in, &out};
      ASSERT_EQ(fixture.execute({args, 2}), Error::Ok);
      EXPECT_EQ(bridge.state().last_input_bytes.load(), nullptr);
    }
  }
}

TEST_F(CoreAIDelegateTest, BorrowedStorageMayAliasOutputAfterExecution) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    tensors.output_impl.set_data(tensors.input);
    ASSERT_EQ(fixture.execute({tensors.args, 2}), Error::Ok);
    EXPECT_EQ(tensors.input[0], 3);
    EXPECT_EQ(tensors.input[1], 7);
  }
}

TEST_F(CoreAIDelegateTest, BorrowedStorageUsesCurrentInputPointer) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    FloatTensors tensors;
    ASSERT_EQ(fixture.execute({tensors.args, 2}), Error::Ok);
    float next_input[] = {11, 13};
    tensors.input_impl.set_data(next_input);
    ASSERT_EQ(fixture.execute({tensors.args, 2}), Error::Ok);
    EXPECT_EQ(bridge.state().last_input_bytes.load(), next_input);
    EXPECT_EQ(tensors.output[0], 11);
    EXPECT_EQ(tensors.output[1], 13);
  }
}

TEST_F(CoreAIDelegateTest, DynamicShapesBorrowCurrentSubrange) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    ASSERT_EQ(fixture.init(), Error::Ok);
    RangeTensors<float> tensors;
    for (TensorImpl::SizesType rows : {3, 0}) {
      SCOPED_TRACE(rows);
      TensorImpl::SizesType shape[] = {rows, 2};
      ASSERT_EQ(resize_tensor(tensors.in.toTensor(), {shape, 2}), Error::Ok);
      tensors.fill_output();
      ASSERT_EQ(fixture.execute({tensors.args, 2}), Error::Ok);
      const size_t elements = static_cast<size_t>(rows) * 2;
      EXPECT_EQ(bridge.state().last_input_bytes.load(), tensors.input + 2);
      EXPECT_EQ(bridge.state().last_input_byte_count.load(), elements * sizeof(float));
      EXPECT_EQ(tensors.out.toTensor().size(0), rows);
      EXPECT_EQ(tensors.out.toTensor().const_data_ptr(), tensors.output + 2);
      for (size_t i = 0; i < 12; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(tensors.output[i], i >= 2 && i < 2 + elements ? tensors.input[i] : 99.0f);
      }
    }
  }
}

TEST_F(CoreAIDelegateTest, StorageOptionsRejectWrongTypeAndRelativePaths) {
  @autoreleasepool {
    DelegateFixture fixture(backend, bridge);
    {
      SCOPED_TRACE("integer assets directory");
      BackendOption wrong_type{"coreai_assets_dir", 1};
      EXPECT_EQ(fixture.init_with_options({&wrong_type, 1}), Error::InvalidArgument);
    }
    {
      SCOPED_TRACE("relative assets directory");
      BackendOptions<1> relative;
      ASSERT_EQ(relative.set_option("coreai_assets_dir", "relative"), Error::Ok);
      EXPECT_EQ(fixture.init_with_options({relative.view().data(), relative.view().size()}),
                Error::InvalidArgument);
    }
    EXPECT_EQ(fixture.freed, 0);
  }
}

TEST_F(CoreAIDelegateTest, LifetimeDestroyRetainsSourceAndReloadIsolatesRoots) {
  @autoreleasepool {
    DelegateFixture first(backend, bridge);
    ASSERT_EQ(first.init(), Error::Ok);
    EXPECT_EQ(first.freed, 1);
    EXPECT_EQ(bridge.state().sessions.load(), 1);
    NSString* first_bundle = bridge.state().last_bundle;
    ASSERT_NE(first_bundle, nil);
    first.reset();
    EXPECT_EQ(bridge.state().sessions.load(), 0);
    EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:first_bundle]);
    backend->destroy(nullptr);

    DelegateFixture reloaded(backend, bridge);
    ASSERT_EQ(reloaded.init(first.storage.url.path), Error::Ok);
    EXPECT_TRUE([bridge.state().last_bundle isEqualToString:first_bundle]);
    DelegateFixture other(backend, bridge);
    NSURL* nested = [other.storage.url URLByAppendingPathComponent:@"nested/models"];
    EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:nested.path]);
    ASSERT_EQ(other.init(nested.path), Error::Ok);
    EXPECT_EQ(bridge.state().sessions.load(), 2);
    EXPECT_TRUE([bridge.state().last_bundle
        hasPrefix:[other.storage.url.path stringByAppendingString:@"/"]]);
    EXPECT_FALSE([bridge.state().last_bundle isEqualToString:first_bundle]);
    EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:first_bundle]);
    other.reset();
    reloaded.reset();
    EXPECT_EQ(bridge.state().sessions.load(), 0);
  }
  EXPECT_EQ(bridge.state().sessions.load(), 0);
  EXPECT_EQ(bridge.state().prepared_models.load(), 0);
  EXPECT_EQ(bridge.state().loaders.load(), 0);
}

}  // namespace
}  // namespace executorch::backends::coreai::testing
