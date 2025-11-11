/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "DynamicTestCase.h"

#import <objc/runtime.h>
#import <sys/utsname.h>

#if TARGET_OS_IOS
#import <UIKit/UIDevice.h>
#endif

static NSString *deviceInfoString(void) {
  static NSString *deviceInfo;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    struct utsname systemInfo;
    uname(&systemInfo);
#if TARGET_OS_IOS
    UIDevice *device = UIDevice.currentDevice;
    deviceInfo = [NSString stringWithFormat:@"%@_%@_%@",
                                            device.systemName,
                                            device.systemVersion,
                                            @(systemInfo.machine)];
#elif TARGET_OS_MAC
        NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
        deviceInfo = [NSString stringWithFormat:@"macOS_%ld_%ld_%ld_%@",
                      (long)version.majorVersion,
                      (long)version.minorVersion, (long)version.patchVersion, @(systemInfo.machine)];
#endif // TARGET_OS_IOS
    deviceInfo = [[deviceInfo
        componentsSeparatedByCharactersInSet:[NSCharacterSet
                                                 punctuationCharacterSet]]
        componentsJoinedByString:@"_"];
  });
  return deviceInfo;
}

@implementation DynamicTestCase

+ (void)initialize {
  if (self != [DynamicTestCase class]) {
    NSString *deviceInfo = deviceInfoString();
    [[self dynamicTests]
        enumerateKeysAndObjectsUsingBlock:^(NSString *testName,
                                            void (^testCase)(XCTestCase *),
                                            BOOL __unused *stop) {
          NSString *methodName =
              [NSString stringWithFormat:@"test_%@_%@", testName, deviceInfo];
          class_addMethod(self,
                          NSSelectorFromString(methodName),
                          imp_implementationWithBlock(testCase),
                          "v@:");
        }];
  }
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTests {
  return @{};
}

@end
