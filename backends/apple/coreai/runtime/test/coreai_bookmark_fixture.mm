/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_bookmark_fixture.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include "coreai_manifest_fixture.h"

extern char** environ;

namespace executorch::backends::coreai::testing {
namespace {
using runtime::Error;
using runtime::Result;
using ::testing::AssertionFailure;
using ::testing::AssertionSuccess;
using Clock = std::chrono::steady_clock;
constexpr auto kChildTimeout = std::chrono::seconds(10);
constexpr auto kContentionObservation = std::chrono::milliseconds(100);
constexpr int kChildSignalFD = 3;
std::string executable;

int remaining_ms(Clock::time_point deadline) {
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
  return remaining.count() <= 0 ? 0 : static_cast<int>(remaining.count());
}

bool send_signal_byte(int fd, char byte) {
  const auto deadline = Clock::now() + kChildTimeout;
  while (const int remaining = remaining_ms(deadline)) {
    pollfd entry{fd, POLLOUT, 0};
    const int ready = poll(&entry, 1, remaining);
    if (ready < 0 && errno == EINTR) continue;
    if (ready <= 0 || !(entry.revents & POLLOUT)) return false;
    const ssize_t written = write(fd, &byte, 1);
    if (written == 1) return true;
    if (written < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    return false;
  }
  return false;
}
}  // namespace

Result<Manifest> bookmark_manifest(bool aot) {
  NSData* encoded = encode(aot ? aot_manifest_dict() : manifest_dict());
  if (encoded == nil) return Error::InvalidExternalData;
  auto parsed = parse_manifest(encoded);
  if (!parsed.ok()) return parsed.error();
  return select_assets(parsed.get(), @"arch_b", @"macOS");
}

Result<NSString*> test_bookmark_key(const Manifest& manifest) {
  return bookmark_key(manifest, @"macOS", @"arch_b");
}

NSURL* test_bookmark_url(NSURL* root, NSString* key) {
  return [[root URLByAppendingPathComponent:@"bookmarks"]
      URLByAppendingPathComponent:[key stringByAppendingString:@".bookmark"]];
}

Result<NSData*> saved_bookmark(NSURL* root, NSString* key) {
  auto locked = lock_bookmark(root.path, key);
  if (!locked.ok()) return locked.error();
  return read_bookmark(*locked.get());
}

Result<NSData*> sized_bookmark(size_t size, uint32_t tag) {
  if (size < sizeof(tag)) return Error::InvalidArgument;
  NSMutableData* bytes = [NSMutableData dataWithLength:size];
  if (bytes == nil) return Error::MemoryAllocationFailed;
  std::memcpy(bytes.mutableBytes, &tag, sizeof(tag));
  return [bytes copy];
}

void set_bookmark_test_executable(const char* path) { executable = path == nullptr ? "" : path; }

int bookmark_child_mode(int argc, char** argv) {
  if (argc < 2 || std::strncmp(argv[1], "--bookmark-", 11) != 0) return -1;
  if (argc != 5) return 1;
  const bool hold = std::strcmp(argv[1], "--bookmark-hold") == 0;
  const bool lock = std::strcmp(argv[1], "--bookmark-lock") == 0;
  const bool other = std::strcmp(argv[3], "other") == 0;
  const bool same = std::strcmp(argv[3], "same") == 0;
  if ((!hold && !lock) || (!other && !same)) return 2;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(argv[4], &end, 10);
  if (errno != 0 || end == argv[4] || *end != '\0' || value < 0 || value > INT_MAX) return 3;
  FileDescriptor signal_fd(static_cast<int>(value));
  if (fcntl(signal_fd.get(), F_GETFD) < 0) return 4;
  signal(SIGPIPE, SIG_IGN);
  NSString* root = [NSString stringWithUTF8String:argv[2]];
  if (root == nil) return 5;
  // Child modes run before InitGoogleTest and must not call assertion helpers.
  Manifest manifest;
  manifest.path = @"ab/model.aimodel";
  manifest.bundle_digests = @{
    @"model.aimodel" : [NSString stringWithUTF8String:std::string(64, other ? '2' : '1').c_str()]
  };
  auto key = test_bookmark_key(manifest);
  if (!key.ok()) return 6;
  if (lock && same && !send_signal_byte(signal_fd.get(), 'B')) return 7;
  auto locked = lock_bookmark(root, key.get());
  if (!locked.ok()) return 7;
  if (!send_signal_byte(signal_fd.get(), 'L')) return 8;
  if (close(signal_fd.release()) != 0) return 9;
  if (hold) {
    // The parent owns this PID and always kills and reaps it, including on failure.
    for (;;) pause();
  }
  return 0;
}

BookmarkChild::~BookmarkChild() {
  if (pid_ > 0) {
    const auto cleaned = kill_and_reap();
    if (!cleaned) ADD_FAILURE() << cleaned.message();
  }
  if (signal_fd_ >= 0 && close(signal_fd_) != 0) {
    ADD_FAILURE() << "Cannot close child signal descriptor: " << errno;
  }
}

::testing::AssertionResult BookmarkChild::spawn(NSString* mode, NSString* root,
                                                NSString* argument) {
  if (pid_ > 0 || signal_fd_ >= 0 || executable.empty() || root == nil) {
    return AssertionFailure() << "Invalid bookmark child setup";
  }
  int descriptors[2];
  if (pipe(descriptors) != 0) return AssertionFailure() << "pipe: " << errno;
  FileDescriptor reader(descriptors[0]);
  FileDescriptor writer(descriptors[1]);
  for (int fd : descriptors) {
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 || fcntl(fd, F_SETFL, O_NONBLOCK) != 0) {
      return AssertionFailure() << "fcntl pipe: " << errno;
    }
  }
  posix_spawn_file_actions_t actions;
  int error = posix_spawn_file_actions_init(&actions);
  if (error != 0) return AssertionFailure() << "spawn actions: " << error;
  error = posix_spawn_file_actions_adddup2(&actions, writer.get(), kChildSignalFD);
  std::string descriptor = std::to_string(kChildSignalFD);
  char* arguments[] = {const_cast<char*>(executable.c_str()),
                       const_cast<char*>(mode.UTF8String),
                       const_cast<char*>(root.fileSystemRepresentation),
                       const_cast<char*>(argument.UTF8String),
                       descriptor.data(),
                       nullptr};
  if (error == 0) {
    error = posix_spawn(&pid_, arguments[0], &actions, nullptr, arguments, environ);
  }
  const int destroyed = posix_spawn_file_actions_destroy(&actions);
  if (error != 0) {
    pid_ = -1;
    return AssertionFailure() << "posix_spawn: " << error;
  }
  signal_fd_ = reader.release();
  if (destroyed != 0) return AssertionFailure() << "destroy spawn actions: " << destroyed;
  return AssertionSuccess();
}

::testing::AssertionResult BookmarkChild::receive(char expected) {
  const auto deadline = Clock::now() + kChildTimeout;
  while (const int remaining = remaining_ms(deadline)) {
    pollfd entry{signal_fd_, POLLIN, 0};
    const int ready = poll(&entry, 1, remaining);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) return AssertionFailure() << "poll: " << errno;
    if (ready == 0) break;
    if (!(entry.revents & POLLIN))
      return AssertionFailure() << "Child closed signal pipe before " << expected;
    char byte = 0;
    const ssize_t count = read(signal_fd_, &byte, 1);
    if (count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (count != 1 || byte != expected) {
      return AssertionFailure() << "Expected child signal " << expected << ", got " << int(byte);
    }
    return AssertionSuccess();
  }
  return AssertionFailure() << "Timed out waiting for child signal " << expected;
}

::testing::AssertionResult BookmarkChild::expect_blocked() {
  const auto deadline = Clock::now() + kContentionObservation;
  while (const int remaining = remaining_ms(deadline)) {
    pollfd entry{signal_fd_, POLLIN, 0};
    const int ready = poll(&entry, 1, remaining);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) return AssertionFailure() << "poll: " << errno;
    if (ready > 0) return AssertionFailure() << "Child completed while lock was held";
    return AssertionSuccess();
  }
  return AssertionSuccess();
}

::testing::AssertionResult BookmarkChild::reap(int& status) {
  if (pid_ <= 0) return AssertionFailure() << "No owned child to reap";
  const auto deadline = Clock::now() + kChildTimeout;
  while (const int remaining = remaining_ms(deadline)) {
    const pid_t result = waitpid(pid_, &status, WNOHANG);
    if (result == pid_) {
      pid_ = -1;
      return AssertionSuccess();
    }
    if (result < 0 && errno == EINTR) continue;
    if (result < 0) return AssertionFailure() << "waitpid: " << errno;
    poll(nullptr, 0, std::min(remaining, 10));
  }
  return AssertionFailure() << "Timed out reaping child " << pid_;
}

::testing::AssertionResult BookmarkChild::expect_exit(int expected) {
  int status = 0;
  auto result = reap(status);
  if (!result) return result;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != expected) {
    return AssertionFailure() << "Expected child exit " << expected << ", wait status " << status;
  }
  return AssertionSuccess();
}

::testing::AssertionResult BookmarkChild::kill_and_reap() {
  if (pid_ <= 0) return AssertionFailure() << "No owned child to kill";
  const int killed = kill(pid_, SIGKILL);
  const int error = errno;
  int status = 0;
  auto result = reap(status);
  if (!result) return result;
  if (killed != 0 || !WIFSIGNALED(status) || WTERMSIG(status) != SIGKILL) {
    return AssertionFailure() << "Expected SIGKILL, kill errno " << (killed == 0 ? 0 : error)
                              << ", wait status " << status;
  }
  return AssertionSuccess();
}

}  // namespace executorch::backends::coreai::testing
