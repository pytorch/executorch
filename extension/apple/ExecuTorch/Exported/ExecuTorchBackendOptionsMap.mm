/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchBackendOptionsMap.h"
#import "ExecuTorchBackendOptionsMap+Internal.h"

#import "ExecuTorchError.h"

#import <executorch/runtime/backend/options.h>

#import <climits>
#import <unordered_set>
#import <vector>

using executorch::runtime::BackendOption;
using executorch::runtime::Error;
using executorch::runtime::LoadBackendOptionsMap;
using executorch::runtime::Span;
using executorch::runtime::kMaxOptionKeyLength;
using executorch::runtime::kMaxOptionValueLength;

namespace {

// Translate an ObjC dictionary into the C++ map + the backing storage whose
// Spans the map references. On failure, returns a non-OK Error, writes a
// human-readable explanation to `*outReason`, and leaves the outputs in a
// partial (but destructible) state. Callers should construct the
// storage/map locally and only commit them to ivars on success.
//
// Lifetime note on `storage`: each `Span<BackendOption>` stored inside `map`
// points at the heap buffer owned by one of the inner `std::vector`s. Even
// if the OUTER `std::vector` reallocates, those inner vectors are
// move-constructed to their new home, which preserves their data() pointer
// (the heap buffer is moved, not copied). So the Spans remain valid across
// outer-vector growth. The `reserve(options.count)` below is belt-and-
// suspenders: we never need to grow beyond the reservation because the
// loop iterates exactly `options.count` times.
Error buildBackendOptionsMap(
    NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *options,
    std::vector<std::vector<BackendOption>> &storage,
    LoadBackendOptionsMap &map,
    NSString * __autoreleasing *outReason) {
  storage.reserve(options.count);
  for (NSString *backendId in options) {
    const char *backendIdCStr = backendId.UTF8String;
    if (backendIdCStr == nullptr) {
      *outReason = @"backend id is not valid UTF-8";
      return Error::InvalidArgument;
    }
    // Reject empty backend ids early so the error message names the caller
    // bug precisely; the C++ set_options below also rejects empty ids, but
    // via a generic InvalidArgument.
    if (backendIdCStr[0] == '\0') {
      *outReason = @"backend id is empty";
      return Error::InvalidArgument;
    }
    NSArray<ExecuTorchBackendOption *> *backendOptions = options[backendId];
    std::vector<BackendOption> opts;
    opts.reserve(backendOptions.count);
    // Reject duplicate keys within the same backend. The C++ runtime's
    // set_option path would otherwise silently resolve to last-write-wins
    // (or undefined depending on the backend), which is almost never what
    // the caller intended.
    std::unordered_set<std::string> seenKeys;
    seenKeys.reserve(backendOptions.count);
    for (ExecuTorchBackendOption *opt in backendOptions) {
      BackendOption bo;
      const char *keyCStr = opt.key.UTF8String;
      if (keyCStr == nullptr) {
        *outReason = [NSString stringWithFormat:
            @"option key for backend '%@' is not valid UTF-8", backendId];
        return Error::InvalidArgument;
      }
      if (keyCStr[0] == '\0') {
        *outReason = [NSString stringWithFormat:
            @"option key for backend '%@' is empty", backendId];
        return Error::InvalidArgument;
      }
      // The C++ runtime stores option keys in a fixed-size buffer of
      // kMaxOptionKeyLength (including the null terminator). Reject inputs
      // that would silently truncate.
      const size_t keyLen = strlen(keyCStr);
      if (keyLen >= kMaxOptionKeyLength) {
        *outReason = [NSString stringWithFormat:
            @"option key '%@' for backend '%@' is %zu bytes; limit is %zu",
            opt.key, backendId, keyLen, (size_t)(kMaxOptionKeyLength - 1)];
        return Error::InvalidArgument;
      }
      if (!seenKeys.insert(std::string(keyCStr)).second) {
        *outReason = [NSString stringWithFormat:
            @"duplicate option key '%@' for backend '%@'",
            opt.key, backendId];
        return Error::InvalidArgument;
      }
      strncpy(bo.key, keyCStr, kMaxOptionKeyLength - 1);
      bo.key[kMaxOptionKeyLength - 1] = '\0';
      switch (opt.type) {
        case ExecuTorchBackendOptionTypeBoolean:
          bo.value = (bool)opt.boolValue;
          break;
        case ExecuTorchBackendOptionTypeInteger:
          // The C++ runtime stores integer option values as 32-bit `int`.
          // Reject anything that would silently narrow. On Apple's current
          // 64-bit-only targets NSInteger is int64_t, so this check is
          // meaningful; on a hypothetical 32-bit build NSInteger would be
          // int32_t and the comparison would be tautological (still correct,
          // just never trips).
          if (opt.intValue < INT_MIN || opt.intValue > INT_MAX) {
            *outReason = [NSString stringWithFormat:
                @"option '%@' for backend '%@' is %ld; out of 32-bit int range",
                opt.key, backendId, (long)opt.intValue];
            return Error::InvalidArgument;
          }
          bo.value = (int)opt.intValue;
          break;
        case ExecuTorchBackendOptionTypeString: {
          const char *valCStr = opt.stringValue.UTF8String;
          if (valCStr == nullptr) {
            *outReason = [NSString stringWithFormat:
                @"option '%@' value for backend '%@' is not valid UTF-8",
                opt.key, backendId];
            return Error::InvalidArgument;
          }
          // Same fixed-buffer constraint as the key.
          const size_t valLen = strlen(valCStr);
          if (valLen >= kMaxOptionValueLength) {
            *outReason = [NSString stringWithFormat:
                @"option '%@' value for backend '%@' is %zu bytes; limit is %zu",
                opt.key, backendId, valLen, (size_t)(kMaxOptionValueLength - 1)];
            return Error::InvalidArgument;
          }
          std::array<char, kMaxOptionValueLength> arr{};
          strncpy(arr.data(), valCStr, kMaxOptionValueLength - 1);
          arr[kMaxOptionValueLength - 1] = '\0';
          bo.value = arr;
          break;
        }
      }
      opts.push_back(bo);
    }
    storage.push_back(std::move(opts));
    auto &backOpts = storage.back();
    // C++ set_options enforces backend-id length (kMaxBackendIdLength = 64).
    // We pass through its Error code unchanged but surface a targeted reason.
    const auto err = map.set_options(
        backendIdCStr,
        Span<BackendOption>(backOpts.data(), backOpts.size()));
    if (err != Error::Ok) {
      *outReason = [NSString stringWithFormat:
          @"failed to install options for backend '%@' (C++ Error %d; backend id may exceed 63 bytes or map is full)",
          backendId, (int)err];
      return err;
    }
  }
  return Error::Ok;
}

} // namespace

@implementation ExecuTorchBackendOptionsMap {
  // Backing storage for the Spans referenced by _map. Must outlive _map.
  std::vector<std::vector<BackendOption>> _storage;
  LoadBackendOptionsMap _map;
  // Cached snapshot of the original input, for the public -options accessor.
  // Built once at init; immutable.
  NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *_snapshot;
}

- (nullable instancetype)initWithOptions:(NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *)options
                                   error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }
  // Build into local temporaries so a partial failure leaves the ivars in a
  // pristine (default-constructed) state. Commit only on full success.
  std::vector<std::vector<BackendOption>> storage;
  LoadBackendOptionsMap map;
  NSString *reason = nil;
  const auto buildError = buildBackendOptionsMap(options, storage, map, &reason);
  if (buildError != Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCodeAndDescription(
          (ExecuTorchErrorCode)buildError, reason);
    }
    return nil;
  }
  _storage = std::move(storage);
  // Move-assignment order matters: `_storage` is moved first so the Spans
  // inside `map` (which point at each inner vector's heap buffer) survive
  // into `_storage`. std::vector's move preserves data() pointers, so the
  // Spans remain valid. This relies on `LoadBackendOptionsMap`'s move
  // being a shallow member-wise move that does not recompute span
  // pointers; if that ever changes, the move order here would need to be
  // revisited. The end-to-end CoreML-delegated test exercises this path.
  _map = std::move(map);
  // Snapshot the input as a shallow-immutable dictionary, then also copy
  // each value array immutably. Combined with ExecuTorchBackendOption
  // itself being immutable, this guarantees the public -options accessor
  // returns a consistent view even if the caller passes mutable container
  // subclasses and mutates them later.
  NSMutableDictionary *snapshot = [NSMutableDictionary dictionaryWithCapacity:options.count];
  for (NSString *key in options) {
    snapshot[key] = [options[key] copy];
  }
  _snapshot = [snapshot copy];
  return self;
}

+ (nullable instancetype)mapWithOptions:(NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *)options
                                  error:(NSError **)error {
  return [[self alloc] initWithOptions:options error:error];
}

- (NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *)options {
  return _snapshot;
}

- (const LoadBackendOptionsMap *)cppMap {
  return &_map;
}

#pragma mark - NSObject

- (NSString *)description {
  // Compact one-line format. The default NSDictionary formatter is multi-
  // line and hard to read in `po`. Backends are listed in the dict's
  // enumeration order (insertion order is not guaranteed by NSDictionary,
  // but in practice this is good enough for debugging).
  if (_snapshot.count == 0) {
    return [NSString stringWithFormat:@"<%@ (empty)>",
            NSStringFromClass([self class])];
  }
  NSMutableArray<NSString *> *backendStrings =
      [NSMutableArray arrayWithCapacity:_snapshot.count];
  for (NSString *backendId in _snapshot) {
    NSArray<ExecuTorchBackendOption *> *opts = _snapshot[backendId];
    NSMutableArray<NSString *> *optStrings =
        [NSMutableArray arrayWithCapacity:opts.count];
    for (ExecuTorchBackendOption *opt in opts) {
      [optStrings addObject:opt.description];
    }
    [backendStrings addObject:
        [NSString stringWithFormat:@"%@=[%@]", backendId,
            [optStrings componentsJoinedByString:@", "]]];
  }
  return [NSString stringWithFormat:@"<%@ %@>",
          NSStringFromClass([self class]),
          [backendStrings componentsJoinedByString:@", "]];
}

@end
