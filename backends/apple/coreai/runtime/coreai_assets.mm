/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_assets.h"
#include <cstdint>
#include <limits>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

bool nonempty_string(id value) {
  return [value isKindOfClass:NSString.class] && [value length] > 0 &&
      [value rangeOfString:@"\0"].location == NSNotFound;
}

bool names_array(id value) {
  if (![value isKindOfClass:NSArray.class]) {
    return false;
  }
  NSMutableSet<NSString*>* seen = [NSMutableSet set];
  for (id name in value) {
    if (!nonempty_string(name) || [seen containsObject:name]) {
      return false;
    }
    [seen addObject:name];
  }
  return true;
}

bool relative_path(NSString* path) {
  if (!nonempty_string(path) || path.isAbsolutePath ||
      [path containsString:@"\\"]) {
    return false;
  }
  for (NSString* component in [path componentsSeparatedByString:@"/"]) {
    if (component.length == 0 || [component isEqualToString:@"."] ||
        [component isEqualToString:@".."]) {
      return false;
    }
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
  for (NSString* file in filenames) {
    ET_CHECK_OR_RETURN_ERROR(
        relative_path(file) && file_size(filenames[file]) &&
            [bundles containsObject:[file componentsSeparatedByString:@"/"]
                                        .firstObject] &&
            [file containsString:@"/"] &&
            [file canBeConvertedToEncoding:NSUTF8StringEncoding],
        InvalidProgram,
        "Invalid Core AI asset path or size");
    [populated addObject:[file componentsSeparatedByString:@"/"].firstObject];
    NSString* parent = file.stringByDeletingLastPathComponent;
    while (parent.length > 0) {
      ET_CHECK_OR_RETURN_ERROR(
          ![files containsObject:parent],
          InvalidProgram,
          "Conflicting Core AI asset paths");
      parent = parent.stringByDeletingLastPathComponent;
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
  if (!nonempty_string(value)) {
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
  Manifest manifest;
  manifest.aot_compiled = [packaging hasPrefix:@"aot_compiled_"];
  manifest.hash = dict[@"hash"];
  manifest.path = dict[@"path"];
  manifest.function = dict[@"function"];
  manifest.inputs = dict[@"input_names"];
  manifest.outputs = dict[@"output_names"];
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(manifest.hash) &&
          [manifest.hash
              rangeOfCharacterFromSet:[[NSCharacterSet
                                          characterSetWithCharactersInString:
                                              @"0123456789abcdef"] invertedSet]]
                  .location == NSNotFound,
      InvalidProgram,
      "Invalid Core AI model hash");
  NSMutableSet<NSString*>* bundles = [NSMutableSet set];
  if (manifest.aot_compiled) {
    manifest.platform = dict[@"platform"];
    manifest.archs = dict[@"archs"];
    ET_CHECK_OR_RETURN_ERROR(
        nonempty_string(manifest.platform),
        InvalidProgram,
        "Core AI AOT manifest requires a target platform; re-export the model");
    ET_CHECK_OR_RETURN_ERROR(
        [manifest.archs isKindOfClass:NSDictionary.class] &&
            manifest.archs.count > 0,
        InvalidProgram,
        "Core AI AOT manifest requires a nonempty architecture map; "
        "re-export the model");
    for (id arch in manifest.archs) {
      ET_CHECK_OR_RETURN_ERROR(
          relative_path(arch) && ![arch containsString:@"/"] &&
              [arch canBeConvertedToEncoding:NSUTF8StringEncoding],
          InvalidProgram,
          "Invalid Core AI AOT architecture name");
      NSString* bundle = [NSString stringWithFormat:@"model.%@.aimodelc", arch];
      id path = manifest.archs[arch];
      ET_CHECK_OR_RETURN_ERROR(
          relative_path(path) &&
              [path isEqualToString:[manifest.hash
                                        stringByAppendingPathComponent:bundle]],
          InvalidProgram,
          "Invalid Core AI AOT bundle path for architecture %s",
          [arch UTF8String]);
      [bundles addObject:bundle];
    }
    manifest.path = nil;
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        relative_path(manifest.path) &&
            [manifest.path
                isEqualToString:[manifest.hash
                                    stringByAppendingString:@"/model.aimodel"]],
        InvalidProgram,
        "Invalid Core AI source bundle path");
    [bundles addObject:@"model.aimodel"];
  }
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(manifest.function) && names_array(manifest.inputs) &&
          names_array(manifest.outputs),
      InvalidProgram,
      "Core AI manifest requires ordered input/output names; re-export the "
      "model");
  ET_CHECK_OR_RETURN_ERROR(
      parse_version(dict[@"min_deployment_version"], manifest.minimum_version),
      InvalidProgram,
      "Invalid Core AI minimum deployment version");
  manifest.files = dict[@"files"];
  ET_CHECK_OK_OR_RETURN_ERROR(validate_files(manifest.files, bundles));
  manifest.bundle_digests = dict[@"bundle_digests"];
  ET_CHECK_OR_RETURN_ERROR(
      [manifest.bundle_digests isKindOfClass:NSDictionary.class] &&
          [[NSSet setWithArray:manifest.bundle_digests.allKeys]
              isEqualToSet:bundles],
      InvalidProgram,
      "Core AI bundle digests must exactly cover declared bundles");
  for (NSString* bundle in bundles) {
    id digest = manifest.bundle_digests[bundle];
    ET_CHECK_OR_RETURN_ERROR(
        nonempty_string(digest) && [digest length] == 64 &&
            [digest
                rangeOfCharacterFromSet:
                    [[NSCharacterSet
                        characterSetWithCharactersInString:@"0123456789abcdef"]
                        invertedSet]]
                    .location == NSNotFound,
        InvalidProgram,
        "Invalid Core AI exported bundle digest");
  }
  return manifest;
}

Result<Manifest> select_assets(
    const Manifest& manifest,
    NSString* device_architecture,
    NSString* platform) {
  if (!manifest.aot_compiled) {
    return manifest;
  }
  ET_CHECK_OR_RETURN_ERROR(
      [manifest.platform isEqualToString:platform],
      DelegateInvalidCompatibility,
      "Core AI AOT target platform %s does not match runtime platform %s; "
      "re-export the model for this platform",
      manifest.platform.UTF8String,
      platform.UTF8String);
  NSArray<NSString*>* architectures = [manifest.archs.allKeys
      sortedArrayUsingComparator:^NSComparisonResult(NSString* a, NSString* b) {
        return [a compare:b options:NSLiteralSearch];
      }];
  NSString* available = [architectures componentsJoinedByString:@", "];
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(device_architecture),
      DelegateInvalidCompatibility,
      "Core AI SDK device architecture is unavailable (%s); available "
      "architectures: [%s]; re-export the model as .aimodel or for this device",
      device_architecture == nil ? "nil" : device_architecture.UTF8String,
      available.UTF8String);
  NSString* path = manifest.archs[device_architecture];
  ET_CHECK_OR_RETURN_ERROR(
      path != nil,
      DelegateInvalidCompatibility,
      "Core AI AOT has no bundle for device architecture %s; available "
      "architectures: [%s]; re-export the model for this device architecture",
      device_architecture.UTF8String,
      available.UTF8String);
  Manifest selected = manifest;
  selected.path = path;
  NSString* prefix = [path.lastPathComponent stringByAppendingString:@"/"];
  NSMutableDictionary<NSString*, NSNumber*>* files =
      [NSMutableDictionary dictionary];
  for (NSString* file in manifest.files) {
    if ([file hasPrefix:prefix]) {
      files[file] = manifest.files[file];
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      files.count > 0,
      InvalidProgram,
      "Core AI manifest has no files for selected device architecture %s; "
      "re-export the model",
      device_architecture.UTF8String);
  selected.files = files;
  selected.bundle_digests = @{
    path.lastPathComponent : manifest.bundle_digests[path.lastPathComponent]
  };
  return selected;
}

} // namespace executorch::backends::coreai
