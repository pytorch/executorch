/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_source_fixture.h"
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <iterator>
#include "coreai_file.h"
#include "coreai_storage.h"

namespace executorch::backends::coreai::testing {
using runtime::Error;
using runtime::Result;

Result<const runtime::TensorLayout> TestData::get_tensor_layout(std::string_view) const {
  return Error::NotSupported;
}
Result<runtime::FreeableBuffer> TestData::get_data(std::string_view key) const {
  if (!required_prefix.empty() && key.substr(0, required_prefix.size()) != required_prefix) {
    ADD_FAILURE() << "NamedDataMap key " << key << " is outside selected prefix "
                  << required_prefix;
    return Error::InvalidArgument;
  }
  if (++attempts == fail_at) return Error::AccessFailed;
  auto found = files.find(std::string(key));
  if (found == files.end()) return Error::NotFound;
  ++requests;
  return runtime::FreeableBuffer(
      found->second.data(), found->second.size(),
      [](void* context, void*, size_t) { ++static_cast<TestData*>(context)->releases; },
      const_cast<TestData*>(this));
}
Error TestData::load_data_into(std::string_view, void*, size_t) const {
  return Error::NotSupported;
}
Result<uint32_t> TestData::get_num_keys() const {
  ++metadata_requests;
  return static_cast<uint32_t>(files.size());
}
Result<const char*> TestData::get_key(uint32_t i) const {
  ++metadata_requests;
  if (i >= files.size()) return Error::NotFound;
  auto found = files.begin();
  std::advance(found, i);
  return found->first.c_str();
}

void aot_data(TestData& data, NSString* arch) {
  data.required_prefix =
      [NSString stringWithFormat:@"coreai/ab/model.%@.aimodelc/", arch].UTF8String;
  data.files.clear();
  data.files[data.required_prefix + "graph.bin"] = std::string("compiled\0graph", 14);
  data.files[data.required_prefix + "nested/weights.bin"] = "weights";
}

void CoreAISourceTest::SetUp() {
  ASSERT_NE(locks_.url, nil);
  struct stat info;
  ASSERT_EQ(lstat(locks_.url.fileSystemRepresentation, &info), 0);
  ASSERT_TRUE(S_ISDIR(info.st_mode));
}

Result<NSURL*> CoreAISourceTest::prepare_source_bundle(const Manifest& manifest,
                                                       const runtime::NamedDataMap* data,
                                                       NSString* staging_root) {
  NSString* key =
      manifest.path == nil ? nil : manifest.bundle_digests[manifest.path.lastPathComponent];
  FileDescriptor lock(-1);
  if (key != nil) {
    FileDescriptor parent(
        open(locks_.url.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (parent.get() < 0) {
      ADD_FAILURE() << "Cannot open source fixture lock root: " << strerror(errno);
      return Error::AccessFailed;
    }
    NSString* name = [key stringByAppendingString:@".lock"];
    lock.reset(open_storage_shared_file(parent.get(), name.fileSystemRepresentation));
    if (lock.get() < 0 || retry_eintr([&] { return flock(lock.get(), LOCK_EX); }) != 0) {
      ADD_FAILURE() << "Cannot acquire source fixture lock: " << strerror(errno);
      return Error::AccessFailed;
    }
  }
  return executorch::backends::coreai::prepare_source_bundle(manifest, data, staging_root, key);
}

}  // namespace executorch::backends::coreai::testing
