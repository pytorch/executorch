// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>

namespace ptn {

// Owning, read-only bytes: either a heap buffer or a read-only file mapping.
//
// Hands out spans that alias the storage. Each alternative keeps its payload
// address across a move, so spans taken before a move stay valid for as long as
// the OwnedBytes lives. Copy is deleted: this holds a whole model.
//
// The mapped alternative is the one that matters for large packages: nothing is
// copied, the pages are demand-paged, and they are shared with any other
// process mapping the same file.
class OwnedBytes {
 private:
  // Releases a mapping. Carries the length because that is what munmap needs,
  // which lets a unique_ptr supply the whole move-only lifetime — no
  // hand-written destructor or move operations.
  struct Unmap {
    size_t size = 0;

    void operator()(void* base) const noexcept;
  };

  struct HeapBuffer {
    std::unique_ptr<uint8_t[]> data;
    size_t size = 0;
  };

  // A read-only mapping of an entire file.
  using MappedFile = std::unique_ptr<void, Unmap>;

  std::variant<std::vector<uint8_t>, HeapBuffer, MappedFile> storage_;

  explicit OwnedBytes(std::vector<uint8_t> bytes)
      : storage_(std::move(bytes)) {}
  explicit OwnedBytes(HeapBuffer buffer) : storage_(std::move(buffer)) {}
  explicit OwnedBytes(MappedFile mapped_file)
      : storage_(std::move(mapped_file)) {}

 public:
  // Empty, owning nothing.
  OwnedBytes() = default;

  ~OwnedBytes() = default;
  OwnedBytes(OwnedBytes&&) noexcept = default;
  OwnedBytes& operator=(OwnedBytes&&) noexcept = default;
  OwnedBytes(const OwnedBytes&) = delete;
  OwnedBytes& operator=(const OwnedBytes&) = delete;

  // The whole payload. Valid for this OwnedBytes' lifetime.
  ByteSpan span() const;

  // True when these bytes are a file mapping rather than a heap buffer.
  bool is_mapped() const;

  // Take ownership of a buffer the caller already has, without copying it.
  static OwnedBytes from_vector(std::vector<uint8_t> bytes);

  // Acquire the contents of `path`. Maps it read-only by default: nothing is
  // copied, pages arrive on demand, and they are shared with any other process
  // mapping the same file. Pass use_mmap=false to read it into the heap
  // instead, which is worth it only when the file must outlive edits to it on
  // disk. A mapping sees concurrent edits, and accessing pages past a
  // concurrent truncation can terminate the process with SIGBUS.
  //
  // `path` must report its complete size through the filesystem. Virtual files
  // such as procfs entries that report zero bytes but produce data are outside
  // this API's scope.
  //
  // Throws std::runtime_error if the file cannot be read, or cannot be mapped
  // when mapping was asked for (including on a platform with no mmap). An empty
  // file yields empty heap bytes either way, since mmap rejects a zero length.
  static OwnedBytes from_file(const std::string& path, bool use_mmap = true);

 private:
  static OwnedBytes read_file(const std::string& path);
  static OwnedBytes map_file(const std::string& path);
};

} // namespace ptn
