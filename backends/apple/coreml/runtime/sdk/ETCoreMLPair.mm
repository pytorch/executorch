//
// ETCoreMLPair.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLPair.h"

@implementation ETCoreMLPair

- (instancetype)initWithFirst:(id)first second:(id)second {
    self = [super init];
    if (self) {
        _first = first;
        _second = second;
    }
    
    return self;
}

- (instancetype)copyWithZone:(NSZone *)zone {
    return [[ETCoreMLPair allocWithZone:zone] initWithFirst:self.first second:self.second];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    }
    
    if (![other isKindOfClass:self.class]) {
        return NO;
    }
    
    return [self.first isEqual:((ETCoreMLPair *)other).first] && [self.second isEqual:((ETCoreMLPair *)other).second];
}

@end
