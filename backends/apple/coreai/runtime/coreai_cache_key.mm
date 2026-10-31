/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_cache_key.h"

#include <CommonCrypto/CommonDigest.h>
#include <limits.h>
#import "coreai_manifest.h"
#import "coreai_validation.h"

namespace executorch::backends::coreai {
namespace {
using runtime::Result;

constexpr NSUInteger kHashLength = 2 * CC_SHA256_DIGEST_LENGTH;
constexpr NSUInteger kFixedLength = 2 * (kHashLength + 1);
constexpr NSUInteger kMaxKeyLength =
    NAME_MAX - (sizeof(kBookmarkExtension) - 1);

static_assert(kCacheEntryLockStripeCount > 0);
static_assert(kCacheEntryLockStripeCount <= 256);

bool valid_device(NSString* device) {
  if (![device isKindOfClass:NSString.class] ||
      device.length > kMaxKeyLength - kFixedLength) {
    return false;
  }
  const NSUInteger separator = [device rangeOfString:@"_"].location;
  if (separator == NSNotFound || separator == 0 ||
      separator + 1 == device.length) {
    return false;
  }
  for (NSUInteger i = 0; i < device.length; ++i) {
    const unichar c = [device characterAtIndex:i];
    if (i == separator || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9')) {
      continue;
    }
    if (i < separator || (c != '_' && c != '-')) {
      return false;
    }
  }
  return true;
}

Result<NSData*> encode_json(id value) {
  NSError* error = nil;
  NSData* data = [NSJSONSerialization dataWithJSONObject:value
                                                 options:0
                                                   error:&error];
  ET_CHECK_OR_RETURN_ERROR(
      data != nil,
      InvalidExternalData,
      "Cannot encode Core AI cache identity: %s",
      error.localizedDescription.UTF8String);
  return data;
}

Result<NSString*> hash_data(NSData* data) {
  ET_CHECK_OR_RETURN_ERROR(
      data != nil && data.length <= UINT_MAX,
      InvalidExternalData,
      "Cannot hash Core AI cache identity");
  unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), bytes);
  NSMutableString* result = [NSMutableString stringWithCapacity:kHashLength];
  for (unsigned char byte : bytes) {
    [result appendFormat:@"%02x", byte];
  }
  return [result copy];
}

Result<NSString*> compute_bundle_identity(
    NSString* digest,
    NSString* bundle_name,
    bool aot_compiled) {
  // Bumping the leading version makes every existing cache entry unreachable.
  auto encoded = encode_json(
      @[ @"1", digest, aot_compiled ? @"aot" : @"source", bundle_name ]);
  if (!encoded.ok()) {
    return encoded.error();
  }
  return hash_data(encoded.get());
}

Result<NSData*> utf8_data(NSString* value) {
  ET_CHECK_OR_RETURN_ERROR(
      is_nonempty_string(value),
      DelegateInvalidCompatibility,
      "Missing Core AI cache specialization");
  NSData* data = [value dataUsingEncoding:NSUTF8StringEncoding];
  ET_CHECK_OR_RETURN_ERROR(
      data != nil, InvalidExternalData, "Cannot encode Core AI specialization");
  return data;
}
} // namespace

bool is_cache_key(NSString* key) {
  if (![key isKindOfClass:NSString.class] || key.length <= kFixedLength ||
      key.length > kMaxKeyLength) {
    return false;
  }
  const NSUInteger options_start = key.length - kHashLength;
  if ([key characterAtIndex:kHashLength] != '-' ||
      [key characterAtIndex:options_start - 1] != '-') {
    return false;
  }
  NSString* model = [key substringToIndex:kHashLength];
  NSString* options = [key substringFromIndex:options_start];
  if (!is_lowercase_hex(model, kHashLength) ||
      !is_lowercase_hex(options, kHashLength)) {
    return false;
  }
  return valid_device(
      [key substringWithRange:NSMakeRange(
                                  kHashLength + 1, key.length - kFixedLength)]);
}

NSString* cache_key_bundle_identity(NSString* key) {
  return is_cache_key(key) ? [key substringToIndex:kHashLength] : nil;
}

Result<NSString*> bundle_identity(
    const Manifest& manifest,
    NSString* bundle_name) {
  NSString* digest = manifest.bundle_digests[bundle_name];
  ET_CHECK_OR_RETURN_ERROR(
      digest != nil, InvalidArgument, "Unknown Core AI bundle name");
  return compute_bundle_identity(digest, bundle_name, manifest.aot_compiled);
}

Result<NSString*> bundle_identity(const SelectedBundle& bundle) {
  return compute_bundle_identity(
      bundle.digest, bundle.bundle_name, bundle.aot_compiled);
}

Result<NSString*> cache_variant_key(
    const SelectedBundle& bundle,
    NSString* platform,
    NSString* architecture,
    NSString* specialization_identity) {
  auto model = bundle_identity(bundle);
  if (!model.ok()) {
    return model.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      is_nonempty_string(platform) && is_nonempty_string(architecture) &&
          ![platform containsString:@"_"],
      DelegateInvalidCompatibility,
      "Invalid Core AI cache context");
  NSString* device =
      [NSString stringWithFormat:@"%@_%@", platform, architecture];
  ET_CHECK_OR_RETURN_ERROR(
      valid_device(device),
      DelegateInvalidCompatibility,
      "Invalid Core AI cache device");
  auto specialization = utf8_data(specialization_identity);
  if (!specialization.ok()) {
    return specialization.error();
  }
  auto options = hash_data(specialization.get());
  if (!options.ok()) {
    return options.error();
  }
  return [NSString
      stringWithFormat:@"%@-%@-%@", model.get(), device, options.get()];
}

Result<NSString*> cache_entry_lock_name(NSString* key) {
  ET_CHECK_OR_RETURN_ERROR(
      is_cache_key(key), InvalidArgument, "Invalid Core AI cache key");
  NSData* encoded = [key dataUsingEncoding:NSUTF8StringEncoding];
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(encoded.bytes, static_cast<CC_LONG>(encoded.length), digest);
  const size_t stripe = digest[0] % kCacheEntryLockStripeCount;
  // A different stripe scheme needs a new prefix so its files never alias
  // these.
  return [NSString stringWithFormat:@"stripe-v1-%02zx.lock", stripe];
}

} // namespace executorch::backends::coreai
