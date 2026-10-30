/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_manifest.h"
#include <cstdint>
#include <limits>
#import "coreai_validation.h"

namespace executorch::backends::coreai {

Manifest::Manifest(
    NSString* export_hash,
    NSString* function,
    NSArray<NSString*>* inputs,
    NSArray<NSString*>* outputs,
    NSDictionary<NSString*, NSNumber*>* files,
    NSDictionary<NSString*, NSString*>* bundle_digests,
    bool aot_compiled,
    NSString* platform,
    NSDictionary<NSString*, NSString*>* architectures,
    NSOperatingSystemVersion minimum_version)
    : export_hash(export_hash),
      function(function),
      inputs(inputs),
      outputs(outputs),
      files(files),
      bundle_digests(bundle_digests),
      aot_compiled(aot_compiled),
      platform(platform),
      architectures(architectures),
      minimum_version(minimum_version) {}

SelectedBundle::SelectedBundle(
    const Manifest& manifest,
    NSString* path,
    NSDictionary<NSString*, NSNumber*>* files)
    : export_hash(manifest.export_hash),
      path(path),
      bundle_name(path.lastPathComponent),
      digest(manifest.bundle_digests[path.lastPathComponent]),
      function(manifest.function),
      inputs(manifest.inputs),
      outputs(manifest.outputs),
      files([files copy]),
      aot_compiled(manifest.aot_compiled) {}

namespace {
using runtime::Error;
using runtime::Result;

bool names_array(id value) {
  if (![value isKindOfClass:NSArray.class]) {
    return false;
  }
  NSMutableSet<NSString*>* seen = [NSMutableSet set];
  for (id name in value) {
    if (!is_nonempty_string(name) || [seen containsObject:name]) {
      return false;
    }
    [seen addObject:name];
  }
  return true;
}

bool file_size(id value) {
  return [value isKindOfClass:NSNumber.class] &&
      CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
      !CFNumberIsFloatType((__bridge CFNumberRef)value) &&
      [value compare:@0] != NSOrderedAscending &&
      [value compare:@(std::numeric_limits<int64_t>::max())] !=
      NSOrderedDescending &&
      [value isEqualToNumber:@([value unsignedLongLongValue])];
}

Error validate_files(
    NSDictionary<NSString*, NSNumber*>* filenames,
    NSSet<NSString*>* bundles) {
  ET_CHECK_OR_RETURN_ERROR(
      [filenames isKindOfClass:NSDictionary.class] && filenames.count > 0,
      InvalidProgram,
      "Core AI manifest requires asset file sizes");
  NSSet<NSString*>* files = [NSSet setWithArray:filenames.allKeys];
  NSMutableSet<NSString*>* populated = [NSMutableSet set];
  NSMutableDictionary<NSString*, NSString*>* spellings =
      [NSMutableDictionary dictionary];
  NSLocale* locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
  for (NSString* file in filenames) {
    ET_CHECK_OR_RETURN_ERROR(
        is_relative_path(file) && file_size(filenames[file]) &&
            [bundles containsObject:[file componentsSeparatedByString:@"/"]
                                        .firstObject] &&
            [file containsString:@"/"] &&
            [file canBeConvertedToEncoding:NSUTF8StringEncoding],
        InvalidProgram,
        "Invalid Core AI asset path or size");
    [populated addObject:[file componentsSeparatedByString:@"/"].firstObject];
    // Aliased parent directories also merge otherwise distinct file paths.
    NSString* path = file;
    while (path.length > 0) {
      NSString* identity =
          [path stringByFoldingWithOptions:NSCaseInsensitiveSearch
                                    locale:locale]
              .decomposedStringWithCanonicalMapping;
      NSString* previous = spellings[identity];
      ET_CHECK_OR_RETURN_ERROR(
          previous == nil || [previous isEqualToString:path],
          InvalidProgram,
          "Case or Unicode alias in Core AI asset paths");
      spellings[identity] = path;
      path = path.stringByDeletingLastPathComponent;
      ET_CHECK_OR_RETURN_ERROR(
          ![files containsObject:path],
          InvalidProgram,
          "Conflicting Core AI asset paths");
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      [populated isEqualToSet:bundles],
      InvalidProgram,
      "Core AI manifest requires files for every declared bundle");
  return Error::Ok;
}

bool parse_version(id value, NSOperatingSystemVersion& version) {
  // A null floor means the exporter used the SDK's save_asset default.
  if (value == NSNull.null) {
    version = {};
    return true;
  }
  if (!is_nonempty_string(value)) {
    return false;
  }
  NSArray<NSString*>* parts = [value componentsSeparatedByString:@"."];
  if (parts.count > 3) {
    return false;
  }
  NSInteger numbers[3] = {0, 0, 0};
  NSCharacterSet* invalid = [[NSCharacterSet
      characterSetWithCharactersInString:@"0123456789"] invertedSet];
  for (NSUInteger i = 0; i < parts.count; ++i) {
    NSString* part = parts[i];
    if (part.length == 0 || part.length > 6 ||
        [part rangeOfCharacterFromSet:invalid].location != NSNotFound) {
      return false;
    }
    numbers[i] = part.integerValue;
  }
  version = {numbers[0], numbers[1], numbers[2]};
  return numbers[0] > 0;
}

} // namespace

Result<Manifest> parse_manifest(NSData* data) {
  ET_CHECK_OR_RETURN_ERROR(
      data != nil, InvalidProgram, "Missing Core AI manifest JSON");
  NSError* error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data
                                              options:0
                                                error:&error];
  ET_CHECK_OR_RETURN_ERROR(
      [object isKindOfClass:NSDictionary.class],
      InvalidProgram,
      "Invalid Core AI manifest JSON");
  NSDictionary* dict = object;
  NSString* packaging = dict[@"packaging"];
  ET_CHECK_OR_RETURN_ERROR(
      [packaging isKindOfClass:NSString.class],
      InvalidProgram,
      "Missing Core AI packaging");
  ET_CHECK_OR_RETURN_ERROR(
      [packaging isEqualToString:@"inline"] ||
          [packaging isEqualToString:@"aot_compiled_inline"],
      NotSupported,
      "Unsupported Core AI packaging; expected .aimodel or AOT .aimodelc");
  const bool aot_compiled = [packaging hasPrefix:@"aot_compiled_"];
  NSString* export_hash = dict[@"export_hash"];
  NSString* function = dict[@"function"];
  NSArray<NSString*>* inputs = dict[@"input_names"];
  NSArray<NSString*>* outputs = dict[@"output_names"];
  NSString* platform = nil;
  NSDictionary<NSString*, NSString*>* architectures = nil;
  ET_CHECK_OR_RETURN_ERROR(
      is_lowercase_hex(export_hash),
      InvalidProgram,
      "Invalid Core AI export hash");
  NSMutableSet<NSString*>* bundles = [NSMutableSet set];
  NSString* source_path = nil;
  if (aot_compiled) {
    platform = dict[@"platform"];
    architectures = dict[@"archs"];
    ET_CHECK_OR_RETURN_ERROR(
        is_nonempty_string(platform),
        InvalidProgram,
        "Core AI AOT manifest requires a target platform");
    ET_CHECK_OR_RETURN_ERROR(
        [architectures isKindOfClass:NSDictionary.class] &&
            architectures.count > 0,
        InvalidProgram,
        "Core AI AOT manifest requires a nonempty architecture map");
    for (id arch in architectures) {
      ET_CHECK_OR_RETURN_ERROR(
          is_relative_path(arch) && ![arch containsString:@"/"] &&
              [arch canBeConvertedToEncoding:NSUTF8StringEncoding],
          InvalidProgram,
          "Invalid Core AI AOT architecture name");
      NSString* bundle = [NSString stringWithFormat:@"model.%@.aimodelc", arch];
      id path = architectures[arch];
      ET_CHECK_OR_RETURN_ERROR(
          is_relative_path(path) &&
              [path isEqualToString:[export_hash
                                        stringByAppendingPathComponent:bundle]],
          InvalidProgram,
          "Invalid Core AI AOT bundle path for architecture %s",
          [arch UTF8String]);
      [bundles addObject:bundle];
    }
  } else {
    source_path = dict[@"path"];
    ET_CHECK_OR_RETURN_ERROR(
        is_relative_path(source_path) &&
            [source_path
                isEqualToString:[export_hash
                                    stringByAppendingString:@"/model.aimodel"]],
        InvalidProgram,
        "Invalid Core AI source bundle path");
    [bundles addObject:@"model.aimodel"];
  }
  ET_CHECK_OR_RETURN_ERROR(
      is_nonempty_string(function) && names_array(inputs) &&
          names_array(outputs),
      InvalidProgram,
      "Core AI manifest requires ordered input/output names");
  NSOperatingSystemVersion minimum_version{};
  ET_CHECK_OR_RETURN_ERROR(
      parse_version(dict[@"min_deployment_version"], minimum_version),
      InvalidProgram,
      "Invalid Core AI minimum deployment version");
  NSDictionary<NSString*, NSNumber*>* files = dict[@"files"];
  ET_CHECK_OK_OR_RETURN_ERROR(validate_files(files, bundles));
  NSDictionary<NSString*, NSString*>* bundle_digests = dict[@"bundle_digests"];
  ET_CHECK_OR_RETURN_ERROR(
      [bundle_digests isKindOfClass:NSDictionary.class] &&
          [[NSSet setWithArray:bundle_digests.allKeys] isEqualToSet:bundles],
      InvalidProgram,
      "Core AI bundle digests must exactly cover declared bundles");
  for (NSString* bundle in bundles) {
    id digest = bundle_digests[bundle];
    ET_CHECK_OR_RETURN_ERROR(
        is_lowercase_hex(digest, 64),
        InvalidProgram,
        "Invalid Core AI exported bundle digest");
  }
  return Manifest(
      export_hash,
      function,
      inputs,
      outputs,
      files,
      bundle_digests,
      aot_compiled,
      platform,
      architectures,
      minimum_version);
}

Result<Manifest> parse_manifest(const void* data, size_t size) {
  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr, InvalidProgram, "Missing Core AI manifest JSON");
  return parse_manifest([NSData dataWithBytes:data length:size]);
}

Result<SelectedBundle> select_bundle(
    const Manifest& manifest,
    NSString* device_architecture,
    NSString* platform) {
  if (!manifest.aot_compiled) {
    return SelectedBundle(
        manifest,
        [manifest.export_hash stringByAppendingString:@"/model.aimodel"],
        manifest.files);
  }
  ET_CHECK_OR_RETURN_ERROR(
      [manifest.platform isEqualToString:platform],
      DelegateInvalidCompatibility,
      "Core AI AOT target platform %s does not match runtime platform %s; "
      "re-export the model for this platform",
      manifest.platform.UTF8String,
      platform == nil ? "nil" : platform.UTF8String);
  NSArray<NSString*>* architectures = [manifest.architectures.allKeys
      sortedArrayUsingComparator:^NSComparisonResult(NSString* a, NSString* b) {
        return [a compare:b options:NSLiteralSearch];
      }];
  NSString* available = [architectures componentsJoinedByString:@", "];
  ET_CHECK_OR_RETURN_ERROR(
      is_nonempty_string(device_architecture),
      DelegateInvalidCompatibility,
      "Core AI SDK device architecture is unavailable (%s); available "
      "architectures: [%s]; re-export the model as .aimodel or for this device",
      device_architecture == nil ? "nil" : device_architecture.UTF8String,
      available.UTF8String);
  NSString* path = manifest.architectures[device_architecture];
  ET_CHECK_OR_RETURN_ERROR(
      path != nil,
      DelegateInvalidCompatibility,
      "Core AI AOT has no bundle for device architecture %s; available "
      "architectures: [%s]; re-export the model for this device architecture",
      device_architecture.UTF8String,
      available.UTF8String);
  NSString* prefix = [path.lastPathComponent stringByAppendingString:@"/"];
  NSMutableDictionary<NSString*, NSNumber*>* selected_files =
      [NSMutableDictionary dictionary];
  for (NSString* file in manifest.files) {
    if ([file hasPrefix:prefix]) {
      selected_files[file] = manifest.files[file];
    }
  }
  return SelectedBundle(manifest, path, selected_files);
}

} // namespace executorch::backends::coreai
