#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace executorch::backends::xnnpack {
/// The key for the backend. This is used to register the backend, check
/// availability, and get/set options.
const char xnnpack_backend_key[] = "XnnpackBackend";

/// The key for the workspace sharing option. See the WorkspaceSharingMode enum
/// for a description of the associated functionality.
const char workspace_sharing_mode_option_key[] = "workspace_sharing_mode";

/// The key for the weight cache option. When enabled, packed weights are shared
// across delegate instances. Changes only affect subsequently loaded models.
const char weight_cache_option_key[] = "weight_cache_enabled";

/// Path for the packed weight file. When set, reserve_space() allocates from
/// a MAP_SHARED file instead of heap; msync makes pages clean on iOS.
// Must remain a C array (not const char*) so it can bind to the
// BackendOptions::set_option(const char (&)[N], ...) template overloads.
// @lint-ignore CLANGTIDY facebook-hte-CArray
const char packed_cache_path_option_key[] = "packed_cache_path";

/// EXPERIMENTAL — option name and semantics may change without notice.
///
/// Setting this to `true` triggers persisting the packed weight cache to disk
/// so a subsequent process load can mmap the same file and skip XNNPACK weight
/// repacking. The on-disk path is configured via
/// `packed_cache_path_option_key`. The disk write is a one-shot side effect
/// (the value is not stored): every `true` set fires another save.
// Must remain a C array for the BackendOptions template overloads.
// @lint-ignore CLANGTIDY facebook-hte-CArray
const char save_weight_cache_on_disk_option_key[] = "save_weight_cache_on_disk";

/// Workspace sharing mode. This is a backend option that can be set via the
/// set_option API to control memory sharing between CALL_DELEGATE instances.
/// This is useful for reducing memory consumption.
enum class WorkspaceSharingMode {
  /// No workspace sharing. Each CALL_DELEGATE instance will have its own
  /// workspace (memory arena).
  Disabled = 0,

  /// All CALL_DELEGATE instances in a given program will share a workspace.
  /// This reduces memory consumption
  /// for methods with multiple delegate calls, at the cost of only allowing one
  /// method to execute at a time.
  PerModel = 1,

  /// All CALL_DELEGATE instances accross all loaded methods will share a
  /// workspace. This reduces memory
  /// consumption by overlapping activation memory between methods but enforces
  /// synchronization between
  /// methods. If multiple methods are run concurrently, it may block as only
  /// one delegate call occur
  /// at a time. Additionally, the workspace does not shrink when a method is
  /// unloaded, so memory will
  /// only be reclaimed when all XNNPACK-delegated methods are unloaded.
  Global = 2,

  /// The number of workspace sharing modes. This is not a valid mode and is
  /// only used for tracking the
  // maximum enum value.
  Count,
};

/// Outcome of opening the packed-weight cache file.
enum class PackedCacheState : int32_t {
  /// No cache path configured — the caller never opted in.
  Disabled = 0,
  /// The cache file opened. Does NOT imply zero heap; see PackedCacheStats.
  FileBacked = 1,
  /// A path was configured but the file could not be used.
  HeapFallback = 2,
};

/** Why an individual allocation was served from heap. */
enum class PackedCacheHeapReason : int32_t {
  None = 0,
  /// The instance has no cache path: it never opted into file backing, so
  /// heap is the intended behaviour rather than a fallback. Bucketed
  /// separately and excluded from heap_bytes — a process that mixes an
  /// opted-in model with a non-opted-in one would otherwise report the
  /// latter's packed weights as if the former had fallen back.
  NotOptedIn = 1,
  /// Unnamed constant — can never be reloaded by name. By design.
  UnnamedConstant = 2,
  /// Incidental re-pack after a successful load. By design *if* the loaded
  /// cache is complete; a large volume here means it was not.
  RepackAfterLoad = 3,
  /// No usable file descriptor at allocation time.
  NoFileBacking = 4,
  /// ftruncate() to extend the file failed.
  GrowFailed = 5,
  /// mmap() of the grown region failed.
  MmapFailed = 6,
  /// Not a reason; bounds the per-reason counters. Matches the
  /// WorkspaceSharingMode convention in this header.
  Count,
};

/** Which step failed when a configured path still ended up on heap. */
enum class PackedCacheFailure : int32_t {
  None = 0,
  OpenFailed = 1,
  TruncateFailed = 2,
  GrowFailed = 3,
  MmapFailed = 4,
};

/**
 * Per-cache counters. `heap_bytes` against `mapped_bytes` is the signal;
 * `state` alone calls a partially-loaded cache healthy.
 */
struct PackedCacheStats {
  PackedCacheState state{PackedCacheState::Disabled};
  PackedCacheFailure failure{PackedCacheFailure::None};
  int32_t last_errno{0};
  /// Cache file size as of the last successful save.
  int64_t file_bytes{0};
  /// Packed bytes served from heap when the file was supposed to serve them.
  /// Excludes NotOptedIn, so this is only ever "bytes that should have been
  /// file-backed and were not".
  int64_t heap_bytes{0};
  /// Packed bytes served from the mmap'd file (clean, file-backed).
  int64_t mapped_bytes{0};
  /// Reason accounting for the largest share of heap_bytes. On the aggregate
  /// this is the argmax over per-reason bytes summed across caches, not the
  /// local reason of whichever cache happened to allocate the most.
  PackedCacheHeapReason heap_reason{PackedCacheHeapReason::None};
  /// Heap bytes split by reason, so callers can sum per reason rather than
  /// per cache. Index with PackedCacheHeapReason. Excludes nothing — the
  /// NotOptedIn slot is populated here but omitted from `heap_bytes`.
  std::array<int64_t, static_cast<std::size_t>(PackedCacheHeapReason::Count)>
      heap_bytes_by_reason{};
};

/** One live cache instance and its own counters. */
struct PackedCacheEntry {
  /// Cache file path. Empty for the shared heap-only instance handed to
  /// callers that never configured one.
  std::string path;
  PackedCacheStats stats;
};

/**
 * Aggregate plus the per-instance breakdown behind it.
 *
 * Both come from one pass, so the summary and the detail always describe the
 * same instant. The breakdown exists because the aggregate alone cannot be
 * attributed: a process running several models folds them into one number, so
 * a fallback in one model is indistinguishable from a fallback in another.
 * The manager already keys caches by path — this stops discarding that.
 *
 * Takes no per-instance lock: the counters are atomics, so this never waits
 * on a model compile.
 */
struct PackedCacheReport {
  /// Summed counters. `failure` / `last_errno` are left unset here; read them
  /// from the `dominant_fallback` entry so they stay tied to one cache.
  PackedCacheStats aggregate;
  /// Sorted by path, so repeated calls agree regardless of map iteration
  /// order.
  std::vector<PackedCacheEntry> per_cache;
  /// Index into `per_cache` of the cache that best explains a fallback: the
  /// largest heap contributor, or if none allocated, the first cache in
  /// HeapFallback. -1 when nothing fell back.
  int32_t dominant_fallback{-1};
};

PackedCacheReport get_packed_cache_report();

} // namespace executorch::backends::xnnpack
