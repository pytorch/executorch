/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/memory_allocator.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#endif
#include <executorch/schema/program_generated.h>
#include <xnnpack.h>
#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::MemoryAllocator;

XNNWeightsCache::XNNWeightsCache() {
  weights_cache_.context = this;
  weights_cache_.look_up = (size_t(*)(
      void*, const xnn_weights_cache_look_up_key*))XNNWeightsCache::look_up;
  weights_cache_.reserve_space =
      (void* (*)(void*, size_t))XNNWeightsCache::reserve_space;
  weights_cache_.look_up_or_insert =
      (size_t(*)(void*, const xnn_weights_cache_look_up_key*, void*, size_t))
          XNNWeightsCache::look_up_or_insert;
  weights_cache_.is_finalized = (bool (*)(void*))XNNWeightsCache::is_finalized;
  weights_cache_.offset_to_addr =
      (void* (*)(void*, size_t))XNNWeightsCache::offset_to_addr;
  weights_cache_.delete_cache =
      (enum xnn_status(*)(void*))XNNWeightsCache::delete_cache;
}

XNNWeightsCache::~XNNWeightsCache() {
#ifndef _WIN32
  for (auto& region : mmap_regions_) {
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
    }
  }
  mmap_regions_.clear();
  if (packed_file_fd_ >= 0) {
    close(packed_file_fd_);
    packed_file_fd_ = -1;
  }
#endif
}

// Trivial helpers for little-endian byte serialization of the trailer.
template <typename T>
static void append_le(std::vector<uint8_t>& buf, T value) {
  const auto* p = reinterpret_cast<const uint8_t*>(&value);
  buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
static T read_le(const uint8_t* src) {
  T value;
  memcpy(&value, src, sizeof(T));
  return value;
}

#ifndef _WIN32
namespace internal {
ssize_t
read_fully_at(int fd, void* buf, size_t n, off_t offset, ReadAtFn read_at) {
  auto* out = static_cast<uint8_t*>(buf);
  size_t done = 0;
  while (done < n) {
    const ssize_t r =
        read_at(fd, out + done, n - done, offset + static_cast<off_t>(done));
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (r == 0) {
      break; // End of file.
    }
    done += static_cast<size_t>(r);
  }
  return static_cast<ssize_t>(done);
}
} // namespace internal

// Open the cache file and take an advisory exclusive lock. Returns the
// fd, or -1 if open/flock failed (logs the failure). The caller decides
// how to recover (typically: skip the mmap path for this init).
// out_errno receives the errno of whichever call failed. Reading errno at the
// call site does not work: the flock path closes the fd first, and close() (or
// ET_LOG) can overwrite it.
static int open_locked(const std::string& path, int flags, int* out_errno) {
  if (out_errno != nullptr) {
    *out_errno = 0;
  }
  int fd = open(path.c_str(), flags, 0600);
  if (fd < 0) {
    const int err = errno;
    if (out_errno != nullptr) {
      *out_errno = err;
    }
    ET_LOG(Error, "open(%s) failed (errno=%d)", path.c_str(), err);
    return -1;
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    const int err = errno;
    if (out_errno != nullptr) {
      *out_errno = err;
    }
    ET_LOG(Error, "flock(%s) failed (errno=%d)", path.c_str(), err);
    close(fd);
    return -1;
  }
  return fd;
}

// Whether fd holds an ExecuTorch program (.pte) or data (.ptd) file.
// A program is a flatbuffer whose 4-byte file identifier is "ET" plus two
// version digits (the current one comes from schema/program.fbs); any version
// counts, not just the one this runtime loads. A data file is a flatbuffer
// with the identifier "FT01" (extension/flat_tensor/serialize/flat_tensor.fbs)
// and the flat tensor header magic "FH01" at offset 8
// (FlatTensorHeader::kMagic at kHeaderOffset). Those two are spelled out so
// the backend doesn't depend on the flat tensor extension; the tests check
// them against FlatTensorHeader.
//
// Both are flatbuffers, so their first 4 bytes are the little-endian offset of
// the root table. The file also has to pass the bound the loaders themselves
// apply to that offset (Program::load and FlatTensorDataMap::load): at least
// 8, past the root offset and the identifier, and leaving room for the
// table's 4-byte vtable offset. The loaders measure against the flatbuffer
// region, which is never larger than the file, so checking against the file
// size accepts everything they accept. A cache left without a trailer, whose
// first bytes are packed weights that happen to look like an identifier,
// usually fails it and is rebuilt.
//
// A file whose start or size can't be read is reported as Unreadable, so the
// caller leaves it alone rather than truncating something it couldn't check.
enum class FileKind { Other, ExecuTorch, Unreadable };
static ssize_t system_pread(int fd, void* buf, size_t n, off_t offset) {
  return pread(fd, buf, n, offset);
}
static FileKind executorch_file_kind(int fd, int* out_errno) {
  constexpr size_t kDataMagicOffset = 8;
  std::array<uint8_t, kDataMagicOffset + 4> head{};
  const ssize_t n =
      internal::read_fully_at(fd, head.data(), head.size(), 0, &system_pread);
  if (n < 0) {
    *out_errno = errno;
    return FileKind::Unreadable;
  }
  if (n != static_cast<ssize_t>(head.size())) {
    // read_fully_at only stops short at end of file, so the file really is
    // shorter than any program or data file head.
    return FileKind::Other;
  }
  const char* id = flatbuffers::GetBufferIdentifier(head.data());
  const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
  const bool is_program =
      executorch_flatbuffer::ProgramBufferHasIdentifier(head.data()) ||
      (id[0] == 'E' && id[1] == 'T' && is_digit(id[2]) && is_digit(id[3]));
  const bool is_data = std::memcmp(id, "FT01", 4) == 0 &&
      std::memcmp(head.data() + kDataMagicOffset, "FH01", 4) == 0;
  if (!is_program && !is_data) {
    return FileKind::Other;
  }
  // The 4-byte root offset plus the 4-byte identifier: ExtendedHeader's and
  // FlatTensorHeader's kHeaderOffset, which the loaders use as this bound.
  constexpr uint64_t kMinRootOffset = 8;
  const uint64_t root_offset =
      flatbuffers::ReadScalar<flatbuffers::uoffset_t>(head.data());
  if (root_offset < kMinRootOffset) {
    return FileKind::Other;
  }
  // lseek rather than fstat, which is on Apple's required-reason API list.
  // Only pread, pwrite, ftruncate and mmap use this fd, so its position
  // doesn't matter.
  const off_t size = lseek(fd, 0, SEEK_END);
  if (size < 0) {
    *out_errno = errno;
    return FileKind::Unreadable;
  }
  if (root_offset + sizeof(flatbuffers::soffset_t) >
      static_cast<uint64_t>(size)) {
    return FileKind::Other;
  }
  return FileKind::ExecuTorch;
}

bool XNNWeightsCache::may_write_cache_file(int fd) {
  int read_errno = 0;
  const FileKind kind = executorch_file_kind(fd, &read_errno);
  if (kind == FileKind::Other) {
    return true;
  }
  // Message first: ET_LOG cuts long lines, and these paths are long.
  if (kind == FileKind::Unreadable) {
    ET_LOG(
        Error,
        "Could not read the start of the file at the cache path (errno=%d); leaving it untouched and packing in memory: %s",
        read_errno,
        packed_cache_path_.c_str());
  } else {
    ET_LOG(
        Error,
        "Cache path names an ExecuTorch program or data file, not a packed weight cache; leaving it untouched and packing in memory: %s",
        packed_cache_path_.c_str());
    cache_file_protected_ = true;
  }
  record_cache_failure(
      PackedCacheFailure::OpenFailed,
      kind == FileKind::Unreadable ? read_errno : EEXIST);
  return false;
}

int XNNWeightsCache::open_cache_for_write(int flags, bool check_content) {
  if (cache_file_protected_) {
    record_cache_failure(PackedCacheFailure::OpenFailed, EEXIST);
    return -1;
  }
  int open_errno = 0;
  const int fd = open_locked(packed_cache_path_, flags, &open_errno);
  if (fd < 0) {
    record_cache_failure(PackedCacheFailure::OpenFailed, open_errno);
    return -1;
  }
  if (check_content && !may_write_cache_file(fd)) {
    close(fd);
    return -1;
  }
  return fd;
}

std::optional<XNNWeightsCache::CacheFooter> XNNWeightsCache::read_cache_footer(
    int fd,
    size_t file_size) const {
  // Footer: [index_start:u64][entry_count:u32][magic:u32][version:u32]
  std::array<uint8_t, 20> footer{};
  if (file_size < footer.size() ||
      internal::read_fully_at(
          fd,
          footer.data(),
          footer.size(),
          static_cast<off_t>(file_size - footer.size()),
          &system_pread) != static_cast<ssize_t>(footer.size())) {
    return std::nullopt;
  }
  const uint64_t index_start = read_le<uint64_t>(footer.data());
  const uint32_t entry_count = read_le<uint32_t>(footer.data() + 8);
  const uint32_t magic = read_le<uint32_t>(footer.data() + 12);
  const uint32_t version = read_le<uint32_t>(footer.data() + 16);
  if (magic != kCacheMagic || version != kCacheVersion ||
      index_start >= file_size - footer.size()) {
    return std::nullopt;
  }
  return CacheFooter{index_start, entry_count};
}

bool XNNWeightsCache::has_cache_trailer(int fd) const {
  const off_t end = lseek(fd, 0, SEEK_END);
  return end >= 0 &&
      read_cache_footer(fd, static_cast<size_t>(end)).has_value();
}

int XNNWeightsCache::reopen_cache_for_write() {
  const int fd = open_cache_for_write(O_RDWR, /*check_content=*/false);
  if (fd < 0) {
    return -1;
  }
  // Trust the file through its own trailer, read on the descriptor that would
  // be written: an ExecuTorch program or data file never ends with it, and a
  // saved cache always does. No head check: the start of a cache holds packed
  // weights, which can happen to read like a program head.
  if (!has_cache_trailer(fd)) {
    ET_LOG(
        Error,
        "The file at the cache path doesn't end with a packed weight cache trailer; not writing to it, packing in memory this init: %s",
        packed_cache_path_.c_str());
    // Not latched in cache_file_protected_: a cache this instance created and
    // hasn't saved yet has no trailer either, and a later init may find a
    // saved cache there again.
    record_cache_failure(PackedCacheFailure::OpenFailed, 0);
    close(fd);
    return -1;
  }
  return fd;
}

// Drop in-memory state that referenced a now-truncated cache file.
// Heap-backed entries (live in packed_pointer_to_container_) stay; their
// packed_data_ptrs_ slots remain valid so existing offsets don't shift.
void XNNWeightsCache::reset_for_fresh_write() {
  for (auto& region : mmap_regions_) {
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
    }
  }
  mmap_regions_.clear();
  mmap_regions_synced_ = 0;
  packed_file_used_ = 0;
  ptr_to_file_offset_.clear();
  file_ptr_to_region_index_.clear();
  for (auto it = name_to_packed_data_metadata_.begin();
       it != name_to_packed_data_metadata_.end();) {
    bool is_heap_backed = false;
    if (it->second.offset < packed_data_ptrs_.size()) {
      void* ptr = packed_data_ptrs_[it->second.offset];
      if (ptr != nullptr &&
          packed_pointer_to_container_.find(ptr) !=
              packed_pointer_to_container_.end()) {
        is_heap_backed = true;
      }
    }
    if (is_heap_backed) {
      ++it;
    } else {
      it = name_to_packed_data_metadata_.erase(it);
    }
  }
}
#endif

void XNNWeightsCache::record_cache_failure(
    PackedCacheFailure failure,
    int err) noexcept {
  state_.store(
      static_cast<int32_t>(PackedCacheState::HeapFallback),
      std::memory_order_relaxed);
  failure_.store(static_cast<int32_t>(failure), std::memory_order_relaxed);
  last_errno_.store(err, std::memory_order_relaxed);
}

PackedCacheStats XNNWeightsCache::stats() const noexcept {
  PackedCacheStats out;
  out.state =
      static_cast<PackedCacheState>(state_.load(std::memory_order_relaxed));
  out.failure =
      static_cast<PackedCacheFailure>(failure_.load(std::memory_order_relaxed));
  out.last_errno = last_errno_.load(std::memory_order_relaxed);
  out.file_bytes = file_bytes_.load(std::memory_order_relaxed);
  out.mapped_bytes = mapped_bytes_.load(std::memory_order_relaxed);
  int64_t worst = 0;
  for (size_t i = 0; i < heap_bytes_by_reason_.size(); ++i) {
    const int64_t bytes =
        heap_bytes_by_reason_[i].load(std::memory_order_relaxed);
    out.heap_bytes_by_reason[i] = bytes;
    if (i == static_cast<size_t>(PackedCacheHeapReason::NotOptedIn)) {
      continue; // intended heap use, not a fallback
    }
    out.heap_bytes += bytes;
    if (bytes > worst) {
      worst = bytes;
      out.heap_reason = static_cast<PackedCacheHeapReason>(i);
    }
  }
  return out;
}

void XNNWeightsCache::record_heap_alloc(
    size_t n,
    PackedCacheHeapReason reason) noexcept {
  // Re-bucket every reason to NotOptedIn when no path was configured. Such an
  // instance is the shared heap-only cache handed to callers that never asked
  // for file backing; counting its bytes as a fallback would inflate the
  // metric for whichever model in the process *did* opt in.
  // packed_cache_path_ is set once before the instance is published and never
  // mutated, so this read needs no synchronization.
  const PackedCacheHeapReason bucket =
      packed_cache_path_.empty() ? PackedCacheHeapReason::NotOptedIn : reason;
  heap_bytes_by_reason_[static_cast<size_t>(bucket)].fetch_add(
      static_cast<int64_t>(n), std::memory_order_relaxed);
}

void XNNWeightsCache::record_mapped_alloc(size_t n) noexcept {
  mapped_bytes_.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
}

void XNNWeightsCache::mark_cache_file_backed() noexcept {
  // Only ever upgrades Disabled -> FileBacked. A fallback already recorded
  // describes memory the process is carrying, so a later success must not
  // mask it.
  int32_t expected = static_cast<int32_t>(PackedCacheState::Disabled);
  state_.compare_exchange_strong(
      expected,
      static_cast<int32_t>(PackedCacheState::FileBacked),
      std::memory_order_relaxed);
}

Error XNNWeightsCache::initialize_for_runtime(
    MemoryAllocator* runtime_allocator,
    const NamedDataMap* named_data_map) {
  runtime_allocator_ = runtime_allocator;
  named_data_map_ = named_data_map;
  is_finalized_ = false;

  // An earlier compile can bail out between its first load_unpacked_data and
  // its finalize_for_runtime, leaving its constants here. A successful compile
  // frees whatever is left in finalize_for_runtime, but a failed one never
  // reaches it, so back to back failures would pile up on an instance that
  // lives as long as the process. The model that loaded them is gone, so free
  // them here.
  for (FreeableBuffer& buffer : unpacked_data_) {
    buffer.Free();
  }
  unpacked_data_.clear();
  unpacked_data_to_name_.clear();

#ifndef _WIN32
  if (packed_cache_path_.empty() || packed_file_fd_ >= 0) {
    return Error::Ok;
  }
  // An earlier init found an ExecuTorch file at this path, so this instance
  // keeps packing in memory and never opens it for writing.
  // open_cache_for_write refuses too; this just skips the reads.
  if (cache_file_protected_) {
    return Error::Ok;
  }

  // Entries already in memory (from a prior load_packed_cache, a prior
  // fresh-write session, or packs that went to heap). Using metadata emptiness
  // (not a separate flag) as the gate avoids a latent bug where
  // fresh-write→save→re-init re-enters load_packed_cache and double-mmaps the
  // same file.
  if (!name_to_packed_data_metadata_.empty()) {
    if (!cache_file_accepted_) {
      // No init of this instance accepted a file at this path (an earlier
      // open failed, or the path changed), so the entries belong to heap or
      // to another file, and nothing vouches for the file here. Pack in
      // memory rather than open it for writing.
      record_cache_failure(PackedCacheFailure::OpenFailed, 0);
      return Error::Ok;
    }
    // This instance loaded or created the file at this path. Reopen it only
    // if it still ends with a cache trailer.
    packed_file_fd_ = reopen_cache_for_write();
    if (packed_file_fd_ >= 0) {
      mark_cache_file_backed();
    }
    return Error::Ok;
  }

  // No in-memory entries: try to load the saved trailer; on success open
  // a write fd for any new entries. If load fails, fall through to
  // fresh-write below.
  if (load_packed_cache()) {
    ET_LOG(
        Info,
        "Loaded packed weight cache: %s (%zu entries)",
        packed_cache_path_.c_str(),
        name_to_packed_data_metadata_.size());
    // load_packed_cache just validated the trailer and index, so the file is a
    // cache. The write descriptor checks the trailer again, on the file it
    // would write, in case the path was replaced in between. Loaded entries
    // are already mmap'd, so reads stay file-backed even if the write fd could
    // not be reopened; the failure is recorded anyway: without it a partial
    // cache silently re-packs to heap every launch with no reason.
    cache_file_accepted_ = true;
    packed_file_fd_ = reopen_cache_for_write();
    mark_cache_file_backed();
    return Error::Ok;
  }

  // Fresh write. Skip O_TRUNC in open_locked so a concurrent holder's
  // mmap stays valid; truncate explicitly only after we hold the lock.
  // The file failed the cache check, and is about to be truncated. A cache
  // path that names a model or its data file (a shortened or wrong path)
  // must not destroy it: the runtime may still have it mapped.
  // The only place the head is checked, because it is the only place a file
  // is about to be destroyed. open_cache_for_write refuses such a file.
  packed_file_fd_ =
      open_cache_for_write(O_RDWR | O_CREAT, /*check_content=*/true);
  if (packed_file_fd_ < 0) {
    ET_LOG(
        Error,
        "Not writing a packed weight file at the cache path; heap fallback this init: %s",
        packed_cache_path_.c_str());
    return Error::Ok;
  }
  if (ftruncate(packed_file_fd_, 0) != 0) {
    const int err = errno;
    ET_LOG(
        Error,
        "ftruncate(0) failed for %s (errno=%d); heap fallback this init",
        packed_cache_path_.c_str(),
        err);
    record_cache_failure(PackedCacheFailure::TruncateFailed, err);
    close(packed_file_fd_);
    packed_file_fd_ = -1;
    return Error::Ok;
  }
  reset_for_fresh_write();
  cache_file_accepted_ = true;
  mark_cache_file_backed();
  ET_LOG(
      Info,
      "Opened packed weight file for writing: %s",
      packed_cache_path_.c_str());
#endif

  return Error::Ok;
}

void XNNWeightsCache::take_unpacked_data_from(
    size_t first_index,
    std::vector<FreeableBuffer>& out) {
  // No bounds check: the loop below is empty when first_index is at or past
  // the end, and the list only grows during a compile so that cannot happen
  // anyway. An index that is too small cannot be detected here, and would hand
  // the executor buffers of values XNNPACK did pack; callers must pass the
  // get_num_unpacked_data() taken immediately before the value was defined.
  for (size_t i = first_index; i < unpacked_data_.size(); i++) {
    // The name map is keyed on the data pointer, which a move preserves, so
    // look_up_or_insert keeps naming packed entries correctly after this.
    out.push_back(std::move(unpacked_data_[i]));
  }
  // The moved-from entries stay in the list rather than being erased:
  // FreeableBuffer deletes move assignment, so vector::erase does not compile,
  // and a moved-from buffer holds a null pointer, which makes the Free() in
  // finalize_for_runtime a no-op. Leaving them also keeps the indices handed
  // out by get_num_unpacked_data() stable across calls.
}

Result<std::vector<std::string>> XNNWeightsCache::finalize_for_runtime() {
  is_finalized_ = true;

  // Everything still here belongs to a value XNNPACK packed, so create_runtime
  // has copied it into the packed region and the unpacked copy can go. Buffers
  // for values XNNPACK does not pack (PReLU slopes, for example) were handed to
  // the executor by take_unpacked_data_from while the graph was being built;
  // the subgraph keeps pointers into those, so freeing them here would leave
  // the runtime reading freed memory.
  for (FreeableBuffer& buffer : unpacked_data_) {
    buffer.Free();
  }
  unpacked_data_.clear();
  unpacked_data_to_name_.clear();

  std::vector<std::string> packed_data_names;
  // update the reference count of all the packed data
  // used by this runtime
  for (auto& entry : name_to_packed_data_metadata_) {
    if (entry.second.in_current_runtime) {
      entry.second.ref_count++;
      entry.second.in_current_runtime = false;
      packed_data_names.push_back(entry.first);
    }
  }

#ifndef _WIN32
  // Synchronous flush for newly added regions. MS_SYNC blocks until the
  // dirty pages are written to disk and marked clean
  if (mmap_regions_.size() > mmap_regions_synced_) {
    size_t new_count = mmap_regions_.size() - mmap_regions_synced_;
    for (size_t i = mmap_regions_synced_; i < mmap_regions_.size(); ++i) {
      if (mmap_regions_[i].addr != nullptr) {
        msync(mmap_regions_[i].addr, mmap_regions_[i].size, MS_SYNC);
      }
    }
    mmap_regions_synced_ = mmap_regions_.size();
    ET_LOG(
        Info,
        "Synced %zu new regions (%zu total), %zu MB packed weights",
        new_count,
        mmap_regions_.size(),
        packed_file_used_ / (1024 * 1024));
  }
#endif

  // Auto-trigger save_packed_index when this compile session added new
  // packed entries to the file.
  if (!packed_cache_path_.empty() &&
      mmap_regions_.size() > mmap_regions_at_last_save_) {
    (void)save_packed_index();
  }

  return packed_data_names;
}

Result<const uint8_t*> XNNWeightsCache::load_unpacked_data(
    const std::string& name) {
  Result<FreeableBuffer> named_data = named_data_map_->get_data(name.c_str());
  if (!named_data.ok()) {
    ET_LOG(Error, "Failed to load constant data for key %s", name.c_str());
    return Error::InvalidExternalData;
  }
  const uint8_t* data_pointer =
      static_cast<const uint8_t*>(named_data.get().data());
  unpacked_data_.push_back(std::move(named_data.get()));
  unpacked_data_to_name_[data_pointer] = name;
  return data_pointer;
}

void XNNWeightsCache::release_entry(void* packed_data_ptr) {
  packed_pointer_to_container_.erase(packed_data_ptr);
#ifndef _WIN32
  // Per-entry file-backed mmap region: munmap to release VM. The
  // packed_data_ptrs_ slot is nulled by the caller so existing offsets
  // stay valid.
  auto region_it = file_ptr_to_region_index_.find(packed_data_ptr);
  if (region_it != file_ptr_to_region_index_.end()) {
    MmapRegion& region = mmap_regions_[region_it->second];
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
      region.addr = nullptr;
      region.size = 0;
    }
    file_ptr_to_region_index_.erase(region_it);
  }
#endif
}

void XNNWeightsCache::full_unload() {
#ifndef _WIN32
  for (auto& region : mmap_regions_) {
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
      region.addr = nullptr;
      region.size = 0;
    }
  }
  mmap_regions_.clear();
  mmap_regions_synced_ = 0;
  packed_data_ptrs_.clear();
  ptr_to_file_offset_.clear();
  file_ptr_to_region_index_.clear();
  if (packed_file_fd_ >= 0) {
    close(packed_file_fd_);
    packed_file_fd_ = -1;
  }
#endif
}

Error XNNWeightsCache::delete_packed_data(
    const std::vector<std::string>& packed_data_names) {
  if (!is_finalized_) {
    ET_LOG(Error, "delete_packed_data called before finalize_for_runtime");
    return Error::InvalidArgument;
  }
  for (const std::string& name : packed_data_names) {
    auto entry = name_to_packed_data_metadata_.find(name);
    if (entry == name_to_packed_data_metadata_.end()) {
      ET_LOG(Error, "delete_packed_data: '%s' not found", name.c_str());
      return Error::InvalidArgument;
    }
    if (--entry->second.ref_count > 0) {
      continue;
    }
    // Keep from_load entries: their packed bytes live in the cache file
    // and stay valid until full unload. Erasing them would force the
    // next init to re-pack and append ~450 MB to the file per cycle.
    if (entry->second.from_load) {
      entry->second.in_current_runtime = false;
      continue;
    }
    release_entry(packed_data_ptrs_[entry->second.offset]);
    packed_data_ptrs_[entry->second.offset] = nullptr;
    name_to_packed_data_metadata_.erase(entry);
  }

  // Last entry gone: drop all in-memory state. File on disk is preserved
  // so the next process can load_packed_cache and skip re-packing. If
  // reserve_space after the last save corrupted the trailer, load will
  // fall through to fresh-write — same outcome as truncating here.
  if (name_to_packed_data_metadata_.empty()) {
    full_unload();
  }
  return Error::Ok;
}

size_t XNNWeightsCache::look_up(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key) {
  const void* unpacked_weights_ptr = cache_key->kernel;
  const void* unpacked_bias_ptr = cache_key->bias;
  context->last_lookup_seed_mismatch_ = false;
  auto entry = context->unpacked_data_to_name_.find(unpacked_weights_ptr);
  if (entry == context->unpacked_data_to_name_.end()) {
    context->last_lookup_unnamed_ = true;
    return SIZE_MAX;
  }
  context->last_lookup_unnamed_ = false;
  std::string weight_bias_name = entry->second;

  if (unpacked_bias_ptr != nullptr) {
    auto bias_entry = context->unpacked_data_to_name_.find(unpacked_bias_ptr);
    if (bias_entry != context->unpacked_data_to_name_.end()) {
      weight_bias_name.append(bias_entry->second);
    }
  }

  auto packed_weight_entry =
      context->name_to_packed_data_metadata_.find(weight_bias_name);
  if (packed_weight_entry == context->name_to_packed_data_metadata_.end()) {
    return SIZE_MAX;
  }
  // XNNPACK upgrade detection: a ukernel whose implementation changed
  // produces a different seed. Reject the cached entry so look_up_or_insert
  // falls through to re-pack with the current ukernel.
  if (packed_weight_entry->second.seed != cache_key->seed) {
    ET_LOG(
        Info,
        "look_up: seed mismatch for '%s' (cached=0x%08x, current=0x%08x); "
        "treating as miss for re-pack",
        weight_bias_name.c_str(),
        packed_weight_entry->second.seed,
        cache_key->seed);
    context->last_lookup_seed_mismatch_ = true;
    return SIZE_MAX;
  }
  packed_weight_entry->second.in_current_runtime = true;
  return packed_weight_entry->second.offset;
}

void* XNNWeightsCache::reserve_space(XNNWeightsCache* context, size_t n) {
#ifndef _WIN32
  // XNNPACK also re-packs for a different runtime context by calling
  // reserve_space with no preceding look_up, so consume the flag here:
  // only the re-pack directly following a seed-mismatch look_up may take
  // the file-backed path.
  const bool seed_mismatch_repack = context->last_lookup_seed_mismatch_;
  context->last_lookup_seed_mismatch_ = false;

  // Unnamed constants never reload by name, and an incidental re-pack of an
  // already-valid loaded cache needn't persist — both go to heap. A
  // seed-mismatch re-pack is different: the loaded entry is stale (XNNPACK
  // upgrade), so it must take the file-backed path below, letting
  // save_packed_index persist the refreshed seed so the next launch hits
  // instead of re-packing into heap (dirty memory) every time.
  if (context->last_lookup_unnamed_ ||
      (context->loaded_from_disk_ && !seed_mismatch_repack)) {
    context->record_heap_alloc(
        n,
        context->last_lookup_unnamed_ ? PackedCacheHeapReason::UnnamedConstant
                                      : PackedCacheHeapReason::RepackAfterLoad);
    return context->reserve_space_heap(n);
  }
  if (context->packed_file_fd_ >= 0) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t file_offset =
        (context->packed_file_used_ + page_size - 1) & ~(page_size - 1);
    size_t map_size = (n + page_size - 1) & ~(page_size - 1);

    if (ftruncate(context->packed_file_fd_, file_offset + map_size) != 0) {
      const int err = errno;
      ET_LOG(
          Error,
          "reserve_space ftruncate to %zu failed (errno=%d)",
          file_offset + map_size,
          err);
      context->record_cache_failure(PackedCacheFailure::GrowFailed, err);
      close(context->packed_file_fd_);
      context->packed_file_fd_ = -1;
      context->record_heap_alloc(n, PackedCacheHeapReason::GrowFailed);
      return context->reserve_space_heap(n);
    }

    void* ptr = mmap(
        nullptr,
        map_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        context->packed_file_fd_,
        file_offset);
    if (ptr == MAP_FAILED) {
      const int err = errno;
      ET_LOG(
          Error,
          "reserve_space mmap %zu bytes failed (errno=%d)",
          map_size,
          err);
      context->record_cache_failure(PackedCacheFailure::MmapFailed, err);
      close(context->packed_file_fd_);
      context->packed_file_fd_ = -1;
      context->record_heap_alloc(n, PackedCacheHeapReason::MmapFailed);
      return context->reserve_space_heap(n);
    }

    // mmap returns page-aligned (>= 4 KiB), which trivially satisfies the
    // 64-byte kPackedAllocationAlignment XNNPACK expects.
    ET_DCHECK_MSG(
        (reinterpret_cast<uintptr_t>(ptr) % kPackedAllocationAlignment) == 0,
        "mmap returned ptr not aligned to %zu bytes",
        kPackedAllocationAlignment);

    context->packed_file_used_ = file_offset + map_size;
    // n, not map_size: the heap side records the raw request too, and the
    // heap:mapped ratio is only meaningful if both measure the same thing.
    context->record_mapped_alloc(n);
    context->file_ptr_to_region_index_[ptr] = context->mmap_regions_.size();
    context->mmap_regions_.push_back({ptr, map_size});
    context->ptr_to_file_offset_[ptr] = file_offset;
    return ptr;
  }
#endif
  context->record_heap_alloc(n, PackedCacheHeapReason::NoFileBacking);
  return context->reserve_space_heap(n);
}

void* XNNWeightsCache::reserve_space_heap(size_t n) {
  try {
    size_t raw_allocation_size = n + kPackedAllocationAlignment - 1;
    std::unique_ptr<char[]> data_container(new char[raw_allocation_size]);

    void* maybe_aligned_space = data_container.get();
    void* aligned_space = std::align(
        kPackedAllocationAlignment,
        n,
        maybe_aligned_space,
        raw_allocation_size // Note that std::align mutates this value.
    );
    ET_CHECK_MSG(aligned_space != nullptr, "Memory alignment failed.");

    packed_pointer_to_container_[aligned_space] = std::move(data_container);
    return aligned_space;
  } catch (std::bad_alloc& e) {
    ET_LOG(
        Error,
        "XNN weight cache failed to allocate %zu bytes: %s.",
        n,
        e.what());
    return nullptr;
  }
}

size_t XNNWeightsCache::look_up_or_insert(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key,
    void* ptr,
    size_t size) {
  size_t offset = context->look_up(context, cache_key);

  // XNNPACK calls with ptr==nullptr after a cache hit (no packing
  // happened, nothing to validate against). Return the offset as-is.
  if (ptr == nullptr) {
    return offset;
  }

  if (offset != SIZE_MAX) {
    void* saved_ptr = context->offset_to_addr(context, offset);
    if (saved_ptr != nullptr && 0 == memcmp(ptr, saved_ptr, size)) {
      return offset;
    }
    // Cache out of date: name hits but packed bytes differ.
    return SIZE_MAX;
  }

  // Add to Cache if it is not finalized
  size_t next_offset = context->packed_data_ptrs_.size();
  auto entry = context->unpacked_data_to_name_.find(cache_key->kernel);

  // Check if weight_pointer has been cached
  if (entry != context->unpacked_data_to_name_.end()) {
    std::string weight_bias_name = entry->second;
    if (cache_key->bias != nullptr) {
      auto bias_entry = context->unpacked_data_to_name_.find(cache_key->bias);
      if (bias_entry != context->unpacked_data_to_name_.end()) {
        weight_bias_name.append(bias_entry->second);
      }
    }
    PackedDataMeta packed_data_metadata;
    packed_data_metadata.offset = next_offset;
    packed_data_metadata.data_size = size;
    packed_data_metadata.ref_count =
        0; // ref_count is only incremented after finalizing for runtime
    packed_data_metadata.in_current_runtime = true;
    packed_data_metadata.seed = cache_key->seed;
    context->name_to_packed_data_metadata_[weight_bias_name] =
        packed_data_metadata;
  } else {
    ET_LOG(
        Info,
        "Warning: Unpacked weight and bias were not registered with names, "
        "this will add new cache entries for packed data and may affect performance.");
  }
  context->packed_data_ptrs_.push_back(ptr);

  return next_offset;
}

bool XNNWeightsCache::is_finalized(XNNWeightsCache* context) {
  return context->is_finalized_;
}

void* XNNWeightsCache::offset_to_addr(XNNWeightsCache* context, size_t offset) {
  return context->packed_data_ptrs_[offset];
}

enum xnn_status XNNWeightsCache::delete_cache(XNNWeightsCache* context) {
  return xnn_status_success;
}

void XNNWeightsCache::set_packed_cache_path(const std::string& path) {
  packed_cache_path_ = path;
  // The refusal and the acceptance are both about the file at the old path.
  cache_file_protected_ = false;
  cache_file_accepted_ = false;
}

Error XNNWeightsCache::save_packed_index() {
#ifndef _WIN32
  if (packed_file_fd_ < 0) {
    return Error::Ok;
  }
  // Skip no-op saves: identical bytes would still bump mtime via
  // pwrite/fsync, making the cache file appear modified on every load.
  // The `mmap_regions_at_last_save_ > 0` guard is sufficient because a
  // successful save closes packed_file_fd_ before returning, so re-entry
  // past the `fd < 0` early-return above requires initialize_for_runtime
  // to reopen the fd, which only happens via load_packed_cache (or the
  // fresh-write path) that always populates at least one mmap region.
  if (mmap_regions_.size() == mmap_regions_at_last_save_ &&
      mmap_regions_at_last_save_ > 0) {
    return Error::Ok;
  }
  // No content check here. packed_file_fd_ only ever comes from
  // open_cache_for_write for a file this instance accepted (one that ends
  // with a cache trailer, or one the fresh write checked and truncated), and
  // writes through it reach that same file even if the path is replaced
  // later. Checking again would only look at weights this process packed,
  // which can happen to start like a program and must not block the save.

  size_t index_start = packed_file_used_;
  std::vector<uint8_t> buf;
  uint32_t entry_count = 0;

  // Index entry: [name_len:u32][name][file_offset:u64][data_size:u64][seed:u32]
  for (const auto& [name, meta] : name_to_packed_data_metadata_) {
    void* ptr = packed_data_ptrs_[meta.offset];
    auto it = ptr_to_file_offset_.find(ptr);
    if (it == ptr_to_file_offset_.end()) {
      continue;
    }
    entry_count++;
    append_le(buf, static_cast<uint32_t>(name.size()));
    buf.insert(buf.end(), name.begin(), name.end());
    append_le(buf, static_cast<uint64_t>(it->second));
    append_le(buf, static_cast<uint64_t>(meta.data_size));
    append_le(buf, meta.seed);
  }

  // Footer: [index_start:u64][entry_count:u32][magic:u32][version:u32]
  append_le(buf, static_cast<uint64_t>(index_start));
  append_le(buf, entry_count);
  append_le(buf, kCacheMagic);
  append_le(buf, kCacheVersion);

  if (ftruncate(packed_file_fd_, index_start + buf.size()) != 0) {
    ET_LOG(Error, "Failed to extend file for index (errno=%d)", errno);
    return Error::Internal;
  }
  ssize_t written =
      pwrite(packed_file_fd_, buf.data(), buf.size(), index_start);
  if (written != static_cast<ssize_t>(buf.size())) {
    ET_LOG(Error, "Failed to write index (errno=%d)", errno);
    return Error::Internal;
  }
  // Ensure trailer is on disk before we declare success.
  if (fsync(packed_file_fd_) != 0) {
    ET_LOG(Error, "fsync of packed cache failed (errno=%d)", errno);
    // Continue — data is in page cache; durability is best-effort.
  }
  // Log the final file size (= index_start + trailer) so production
  // logs surface unbounded growth from orphan packs: a same-name
  // re-pack leaves the old packed bytes in the file even though the
  // trailer drops the old entry. Monitoring file_bytes over time tells
  // us when GC or a size cap is needed.
  const size_t file_bytes = index_start + buf.size();
  file_bytes_.store(
      static_cast<int64_t>(file_bytes), std::memory_order_relaxed);
  ET_LOG(
      Info,
      "Saved packed weight index: %u entries at offset %zu, file_bytes=%zu",
      entry_count,
      index_start,
      file_bytes);

  // Promote freshly-packed entries to from_load now that they're durable
  // on disk, so delete_packed_data preserves them across unload/reload.
  for (auto& [name, meta] : name_to_packed_data_metadata_) {
    if (!meta.from_load &&
        ptr_to_file_offset_.find(packed_data_ptrs_[meta.offset]) !=
            ptr_to_file_offset_.end()) {
      meta.from_load = true;
    }
  }

  mmap_regions_at_last_save_ = mmap_regions_.size();
  // Advance packed_file_used_ PAST the trailer we just wrote. Without
  // this, the next reserve_space (e.g. another PTE's compile inheriting
  // this path) would write at index_start, overwriting the trailer we
  // just wrote and breaking cross-process cache reuse (next launch's
  // load_packed_cache reads garbage at file end → invalid magic →
  // ftruncate(0) → all data lost). The old trailer becomes orphan dead
  // bytes; the new save will write a new trailer at end of the new
  // packs, keeping the trailer always at file end.
  packed_file_used_ = index_start + buf.size();

  // Keep fd open. Subsequent finalize_for_runtime calls auto-trigger
  // save, which needs a live fd. Closing here would force every init to
  // re-enter load_packed_cache + open_locked round-trip — unnecessary
  // since the fd remains valid for both reads and writes through the
  // process lifetime.
#endif
  return Error::Ok;
}

bool XNNWeightsCache::load_packed_cache() {
#ifndef _WIN32
  int fd = open(packed_cache_path_.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }
  // Prevent racing with a concurrent writer
  if (flock(fd, LOCK_SH | LOCK_NB) != 0) {
    close(fd);
    return false;
  }
  // Use lseek instead of fstat: fstat is on Apple's Required Reason API
  // list (file-timestamp category), which would force every iOS consumer
  // of this header to declare a PrivacyInfo reason. lseek is not on the
  // list and yields the same byte count for a regular file.
  off_t end_off = lseek(fd, 0, SEEK_END);
  if (end_off < 0) {
    close(fd);
    return false;
  }
  size_t file_size = static_cast<size_t>(end_off);
  const std::optional<CacheFooter> parsed_footer =
      read_cache_footer(fd, file_size);
  if (!parsed_footer.has_value()) {
    close(fd);
    return false;
  }
  const uint64_t index_start = parsed_footer->index_start;
  const uint32_t entry_count = parsed_footer->entry_count;
  const size_t index_region_end = file_size - 20;

  void* map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  // The live MAP_SHARED mapping keeps this open file description — and with
  // it the flock — alive for as long as the mapping exists, so close() alone
  // does NOT drop the lock. Without the explicit LOCK_UN the O_RDWR reopen in
  // initialize_for_runtime always fails with EWOULDBLOCK, leaving
  // packed_file_fd_ == -1, which silently disables every write path (new
  // packs, seed-mismatch re-packs, save_packed_index) after a warm load.
  flock(fd, LOCK_UN);
  close(fd);
  if (map == MAP_FAILED) {
    return false;
  }
  mmap_regions_.push_back({map, file_size});

  // Bytes actually referenced by the index. Less than file_size whenever an
  // earlier run re-packed a name and orphaned its old bytes.
  size_t loaded_bytes = 0;
  const uint8_t* cursor = static_cast<const uint8_t*>(map) + index_start;
  const uint8_t* end = static_cast<const uint8_t*>(map) + index_region_end;

  for (uint32_t i = 0; i < entry_count && cursor + 4 <= end; ++i) {
    uint32_t name_len = read_le<uint32_t>(cursor);
    cursor += 4;
    // [file_offset:u64][data_size:u64][seed:u32] = 20 bytes
    if (cursor + name_len + 20 > end) {
      // Truncated entry header: trailer doesn't match the entry_count we
      // read from the footer, so the cache is corrupt. Apply the same
      // full rollback as the invalid-bounds branch below — otherwise the
      // entries inserted so far would be silently accepted as a partial
      // cache, and the next save_packed_index would rewrite a trailer
      // covering only that subset (permanently dropping the rest).
      ET_LOG(
          Error,
          "load_packed_cache: truncated entry header at index %u (entry_count=%u); aborting load",
          i,
          entry_count);
      munmap(map, file_size);
      mmap_regions_.pop_back();
      name_to_packed_data_metadata_.clear();
      packed_data_ptrs_.clear();
      ptr_to_file_offset_.clear();
      return false;
    }
    std::string name(reinterpret_cast<const char*>(cursor), name_len);
    cursor += name_len;
    uint64_t file_offset = read_le<uint64_t>(cursor);
    cursor += 8;
    uint64_t data_size = read_le<uint64_t>(cursor);
    cursor += 8;
    uint32_t seed = read_le<uint32_t>(cursor);
    cursor += 4;

    // Bounds check: the entry's bytes must lie entirely inside the
    // packed-data region.
    if (file_offset >= index_start || data_size > index_start - file_offset) {
      ET_LOG(
          Error,
          "load_packed_cache: entry '%s' has invalid bounds (file_offset=%llu, data_size=%llu, index_start=%llu); aborting load",
          name.c_str(),
          static_cast<unsigned long long>(file_offset),
          static_cast<unsigned long long>(data_size),
          static_cast<unsigned long long>(index_start));
      // Roll back any partial state.
      munmap(map, file_size);
      mmap_regions_.pop_back();
      name_to_packed_data_metadata_.clear();
      packed_data_ptrs_.clear();
      ptr_to_file_offset_.clear();
      return false;
    }

    size_t ptr_index = packed_data_ptrs_.size();
    void* entry_ptr = static_cast<char*>(map) + file_offset;
    packed_data_ptrs_.push_back(entry_ptr);
    // Tracked so a subsequent save_packed_index can rewrite the trailer
    // covering both loaded and newly-packed entries.
    ptr_to_file_offset_[entry_ptr] = file_offset;
    PackedDataMeta meta;
    meta.offset = ptr_index;
    meta.data_size = data_size;
    meta.ref_count = 0;
    meta.in_current_runtime = false;
    meta.from_load = true;
    meta.seed = seed;
    loaded_bytes += static_cast<size_t>(data_size);
    name_to_packed_data_metadata_[name] = meta;
  }

  // Start writing new packs AFTER the existing trailer (not at the
  // trailer's offset). Writing at index_start would overwrite the
  // valid trailer this PTE just loaded, breaking cross-process reuse
  // (next launch reads garbage at file end → magic invalid →
  // ftruncate(0) → all data lost). The old trailer becomes orphan
  // bytes in the middle of the file; save_packed_index writes a new
  // trailer at the new end including all entries (old + new).
  packed_file_used_ = file_size;
  // In-memory state matches the on-disk trailer; the next save would be
  // a no-op until new packs arrive. Initialize watermark so
  // save_packed_index short-circuits.
  mmap_regions_at_last_save_ = mmap_regions_.size();
  mmap_regions_synced_ = mmap_regions_.size();
  loaded_from_disk_ = true;
  // Success path only: the truncated-entry branch above munmaps and rolls
  // back, so counting at the mmap call would over-report.
  //
  // loaded_bytes, not file_size. The file is append-only, so a same-name
  // re-pack leaves the old bytes behind; file_size counts those orphans and
  // would inflate mapped_bytes against heap_bytes. Without this a warm launch
  // reports heap=0/mapped=0/file=0, identical to the feature being off.
  record_mapped_alloc(loaded_bytes);
  file_bytes_.store(static_cast<int64_t>(file_size), std::memory_order_relaxed);
  return true;
#else
  return false;
#endif
}

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
