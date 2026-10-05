/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_acquisition_fixture.h"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include "coreai_fault_scope.h"
#include "coreai_load_coordinator.h"

namespace executorch::backends::coreai::testing {
namespace {
using runtime::Error;
using runtime::Result;

dispatch_time_t deadline() { return dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC); }

struct AcquisitionWorker {
  Manifest manifest;
  TestData* data = nullptr;
  NSString* root = nil;
  BookmarkFakeLoader* loader = nil;
  Error error = Error::Internal;
  bool model = false;
  dispatch_semaphore_t started = dispatch_semaphore_create(0);
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  static void* run(void* opaque) {
    auto* worker = static_cast<AcquisitionWorker*>(opaque);
    dispatch_semaphore_signal(worker->started);
    @autoreleasepool {
      auto acquired = acquire_bookmark_model(worker->manifest, worker->data, worker->root, @"macOS",
                                             @"arch_b", worker->loader);
      worker->error = acquired.ok() ? Error::Ok : acquired.error();
      worker->model = acquired.ok() && acquired.get() != nil;
    }
    dispatch_semaphore_signal(worker->done);
    return nullptr;
  }
};

struct AcquisitionWorkers {
  ScopedFakeBridgeState& bridge;
  BookmarkFakeLoader* loader;
  dispatch_semaphore_t proceed = dispatch_semaphore_create(0);
  pthread_t threads[2]{};
  bool started[2] = {false, false};
  AcquisitionWorker workers[2];
  ParallelAcquisitionObservation& observation;
  bool finished = false;

  AcquisitionWorkers(ScopedFakeBridgeState& state, BookmarkFakeLoader* fake,
                     ParallelAcquisitionObservation& result)
      : bridge(state), loader(fake), observation(result) {}

  void finish() {
    if (finished) return;
    for (int i = 0; i < 2; ++i) dispatch_semaphore_signal(proceed);
    for (int i = 0; i < 2; ++i) {
      if (!started[i]) continue;
      observation.completed[i] = dispatch_semaphore_wait(workers[i].done, deadline()) == 0;
      observation.joined[i] = pthread_join(threads[i], nullptr);
      observation.errors[i] = workers[i].error;
      observation.models[i] = workers[i].model;
    }
    observation.callbacks_drained = bridge.wait_for_callbacks(deadline());
    if (observation.callbacks_drained) loader.onSpecialize = nil;
    finished = true;
  }
  ~AcquisitionWorkers() { finish(); }
};

bool send_child_byte(int fd, char value) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < end) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                               end - std::chrono::steady_clock::now())
                               .count();
    pollfd entry{fd, POLLOUT, 0};
    const int ready = poll(&entry, 1, static_cast<int>(remaining > 0 ? remaining : 0));
    if (ready < 0 && errno == EINTR) continue;
    if (ready <= 0 || !(entry.revents & POLLOUT)) return false;
    const ssize_t written = write(fd, &value, 1);
    if (written == 1) return true;
    if (written < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    return false;
  }
  return false;
}
}  // namespace

Result<NSURL*> acquisition_staged_url(NSURL* root, const Manifest& manifest) {
  auto key = test_bookmark_key(manifest);
  if (!key.ok()) return key.error();
  return [[[root URLByAppendingPathComponent:@"staging"] URLByAppendingPathComponent:key.get()]
      URLByAppendingPathComponent:manifest.path.lastPathComponent];
}

ParallelAcquisitionObservation parallel_acquisitions(ScopedFakeBridgeState& bridge,
                                                     BookmarkFakeLoader* loader, NSURL* root,
                                                     const Manifest& first, const Manifest& second,
                                                     TestData& data, bool same_key) {
  ParallelAcquisitionObservation observation;
  auto timeouts = std::make_shared<std::atomic<int>>(0);
  dispatch_semaphore_t entered = dispatch_semaphore_create(0);
  AcquisitionWorkers group(bridge, loader, observation);
  dispatch_semaphore_t proceed = group.proceed;
  loader.onSpecialize = ^(NSURL*) {
    dispatch_semaphore_signal(entered);
    if (dispatch_semaphore_wait(proceed, deadline()) != 0) ++*timeouts;
  };
  for (int i = 0; i < 2; ++i) {
    auto& worker = group.workers[i];
    worker.manifest = i == 0 ? first : second;
    worker.data = &data;
    worker.root = root.path;
    worker.loader = loader;
  }
  observation.creation[0] =
      pthread_create(&group.threads[0], nullptr, AcquisitionWorker::run, &group.workers[0]);
  group.started[0] = observation.creation[0] == 0;
  if (group.started[0]) {
    observation.first_entered = dispatch_semaphore_wait(entered, deadline()) == 0;
  }
  if (observation.first_entered) {
    observation.creation[1] =
        pthread_create(&group.threads[1], nullptr, AcquisitionWorker::run, &group.workers[1]);
    group.started[1] = observation.creation[1] == 0;
    if (group.started[1] && dispatch_semaphore_wait(group.workers[1].started, deadline()) == 0) {
      const auto second_deadline = same_key
          ? dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC)
          : deadline();
      const bool second_entered = dispatch_semaphore_wait(entered, second_deadline) == 0;
      observation.second_observed = same_key ? !second_entered : second_entered;
    }
  }
  group.finish();
  observation.gate_timeouts = timeouts->load();
  return observation;
}

void set_acquisition_test_executable(const char* path) { set_bookmark_test_executable(path); }

::testing::AssertionResult spawn_acquisition_crash(BookmarkChild& child, NSURL* root,
                                                   StorageOperation boundary,
                                                   NSDictionary* manifest) {
  // Serialize the shared fixture in the parent; its builders contain GTest checks.
  NSData* payload = encode(@{@"boundary" : @(static_cast<int>(boundary)), @"manifest" : manifest});
  if (payload == nil) return ::testing::AssertionFailure() << "Cannot encode crash child input";
  NSString* argument = [[NSString alloc] initWithData:payload encoding:NSUTF8StringEncoding];
  return child.spawn(@"--bookmark-crash", root.path, argument);
}

::testing::AssertionResult spawn_parallel_acquisition(BookmarkChild& child, NSURL* root,
                                                      NSDictionary* first, NSDictionary* second,
                                                      bool same_key) {
  NSData* payload = encode(@{@"manifest" : first, @"second" : second, @"same_key" : @(same_key)});
  if (payload == nil) return ::testing::AssertionFailure() << "Cannot encode parallel child input";
  NSString* argument = [[NSString alloc] initWithData:payload encoding:NSUTF8StringEncoding];
  return child.spawn(@"--acquisition-parallel", root.path, argument);
}

namespace {
[[noreturn]] void parallel_child(NSString* root, const Manifest& first, const Manifest& second,
                                 bool same_key, int fd) {
  ScopedFakeBridgeState bridge;
  @autoreleasepool {
    BookmarkFakeLoader* loader = [[BookmarkFakeLoader alloc] init];
    if (loader == nil) _exit(8);
    bridge.state().bookmark_loader = loader;
    TestData data;
    auto result = parallel_acquisitions(bridge, loader, [NSURL fileURLWithPath:root], first, second,
                                        data, same_key);
    if (!result.callbacks_drained) _exit(11);
    // These bytes are observations, not child assertions. The parent checks each one.
    const bool harness_ok = result.creation[0] == 0 && result.creation[1] == 0 &&
                            result.first_entered && result.joined[0] == 0 &&
                            result.joined[1] == 0 && result.completed[0] && result.completed[1] &&
                            result.gate_timeouts == 0;
    const bool both_ok = result.errors[0] == Error::Ok && result.models[0] &&
                         result.errors[1] == Error::Ok && result.models[1];
    const char observations[] = {static_cast<char>(harness_ok),
                                 static_cast<char>(result.second_observed),
                                 static_cast<char>(both_ok),
                                 static_cast<char>(loader->specializations.load()),
                                 static_cast<char>(data.attempts.load()),
                                 static_cast<char>(loader->restores.load())};
    for (char observation : observations) {
      if (!send_child_byte(fd, observation)) _exit(12);
    }
    if (loader->evictions.load() != 0 || data.requests.load() != data.releases.load()) _exit(14);
    [loader clearRetainedState];
    bridge.state().bookmark_loader = nil;
    loader = nil;
  }
  auto& state = bridge.state();
  const bool none_live = state.sessions.load() == 0 && state.prepared_models.load() == 0 &&
                         state.loaders.load() == 0 && state.binding_mismatches.load() == 0 &&
                         state.missing_bundles.load() == 0 && state.input_wait_timeouts.load() == 0;
  if (!send_child_byte(fd, static_cast<char>(none_live))) _exit(12);
  // No GTest-bearing destructors run before InitGoogleTest, even on a failed child.
  _exit(close(fd) == 0 ? 0 : 13);
}
}  // namespace

int acquisition_child_mode(int argc, char** argv) {
  if (argc < 2) return -1;
  const bool crash = std::strcmp(argv[1], "--bookmark-crash") == 0;
  const bool parallel = std::strcmp(argv[1], "--acquisition-parallel") == 0;
  if (!crash && !parallel) return -1;
  if (argc != 5) return 1;
  NSString* root = [NSString stringWithUTF8String:argv[2]];
  NSString* argument = [NSString stringWithUTF8String:argv[3]];
  if (root == nil || argument == nil) return 2;
  id payload =
      [NSJSONSerialization JSONObjectWithData:[argument dataUsingEncoding:NSUTF8StringEncoding]
                                      options:0
                                        error:nil];
  if (![payload isKindOfClass:NSDictionary.class]) return 3;
  id number = payload[@"boundary"];
  id dictionary = payload[@"manifest"];
  if (![dictionary isKindOfClass:NSDictionary.class]) return 3;
  StorageOperation boundary = StorageOperation::AfterSDK;
  if (crash) {
    if (![number isKindOfClass:NSNumber.class]) return 3;
    boundary = static_cast<StorageOperation>([number intValue]);
    if (boundary != StorageOperation::AfterSDK) return 4;
  }
  auto parsed = parse_manifest(encode(dictionary));
  if (!parsed.ok()) return 5;
  auto selected = select_assets(parsed.get(), @"arch_b", @"macOS");
  if (!selected.ok()) return 5;
  char* end = nullptr;
  errno = 0;
  const long descriptor = std::strtol(argv[4], &end, 10);
  if (errno != 0 || end == argv[4] || *end != '\0' || descriptor < 0 || descriptor > INT_MAX)
    return 6;
  const int fd = static_cast<int>(descriptor);
  if (fcntl(fd, F_GETFD) < 0) return 6;
  signal(SIGPIPE, SIG_IGN);
  if (!send_child_byte(fd, 'C')) return 7;
  if (parallel) {
    id second_dictionary = payload[@"second"];
    id same_key = payload[@"same_key"];
    if (![second_dictionary isKindOfClass:NSDictionary.class] ||
        ![same_key isKindOfClass:NSNumber.class])
      return 3;
    auto second_parsed = parse_manifest(encode(second_dictionary));
    if (!second_parsed.ok()) return 5;
    auto second = select_assets(second_parsed.get(), @"arch_b", @"macOS");
    if (!second.ok()) return 5;
    parallel_child(root, selected.get(), second.get(), [same_key boolValue], fd);
  }
  if (close(fd) != 0) return 7;

  ScopedFakeBridgeState bridge;
  @autoreleasepool {
    // Terminal child paths deliberately bypass assertion-bearing scope destructors.
    BookmarkFakeLoader* loader = [[BookmarkFakeLoader alloc] init];
    if (loader == nil) _exit(8);
    bridge.state().bookmark_loader = loader;
    TestData data;
    StorageFaultScope fault(boundary);
    fault.armed = false;
    fault.observe = ^(StorageOperation event) {
      if (event == boundary) _exit(73);
    };
    auto result = acquire_bookmark_model(selected.get(), &data, root, @"macOS", @"arch_b", loader);
    _exit(result.ok() ? 9 : 10);
  }
}

}  // namespace executorch::backends::coreai::testing
