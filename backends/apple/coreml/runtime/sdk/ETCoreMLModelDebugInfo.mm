//
//  ETCoreMLModelDebugInfo.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import "ETCoreMLModelDebugInfo.h"

#import "ETCoreMLStrings.h"
#import "ETCoreMLModelStructurePath.h"
#import "objc_safe_cast.h"


@implementation ETCoreMLModelDebugInfo

- (instancetype)initWithPathToDebugSymbolMap:(NSDictionary<ETCoreMLModelStructurePath *, NSString *> *)pathToDebugSymbolMap
                       pathToDebugHandlesMap:(NSDictionary<ETCoreMLModelStructurePath *, NSArray<NSString *> *> *)pathToDebugHandlesMap {
    self = [super init];
    if (self) {
        _pathToDebugSymbolMap = [pathToDebugSymbolMap copy];
        _pathToDebugHandlesMap = [pathToDebugHandlesMap copy];
    }

    return self;
}

+ (nullable instancetype)modelDebugInfoFromData:(NSData *)data error:(NSError * __autoreleasing *)error {
    id object = [NSJSONSerialization JSONObjectWithData:data options:(NSJSONReadingOptions)0 error:error];
    if (!object) {
        return nil;
    }

    NSDictionary<NSString *, id> *jsonDict = SAFE_CAST(object, NSDictionary);
    // Construct operation path to debug symbol map.
    NSDictionary<NSString *, NSArray<id> *> *debugSymbolToPathMap = SAFE_CAST(jsonDict[ETCoreMLStrings.debugSymbolToOperationPathKeyName], NSDictionary);
    NSMutableDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToDebugSymbolMap = [NSMutableDictionary dictionaryWithCapacity:debugSymbolToPathMap.count];
    for (NSString *symbolName in debugSymbolToPathMap) {
        NSArray<NSDictionary<NSString *, id> *> *components = SAFE_CAST(debugSymbolToPathMap[symbolName], NSArray);
        if (components.count == 0) {
            continue;
        }
        ETCoreMLModelStructurePath *path = [[ETCoreMLModelStructurePath alloc] initWithComponents:components];
        pathToDebugSymbolMap[path] = symbolName;

    }
    // Construct operation path to debug handles map.
    NSDictionary<NSString *, NSArray<NSString *> *> *debugSymbolToHandles = SAFE_CAST(jsonDict[ETCoreMLStrings.debugSymbolToHandlesKeyName], NSDictionary);
    NSMutableDictionary<ETCoreMLModelStructurePath *, NSArray<NSString *> *> *pathToDebugHandlesMap = [NSMutableDictionary dictionaryWithCapacity:debugSymbolToHandles.count];
    for (NSString *debugSymbol in debugSymbolToHandles) {
        NSArray<id> *components = debugSymbolToPathMap[debugSymbol];
        if (components.count == 0) {
            continue;
        }

        NSArray<NSString *> *debugHandles = debugSymbolToHandles[debugSymbol];
        if (debugHandles.count == 0) {
            continue;
        }

        ETCoreMLModelStructurePath *path = [[ETCoreMLModelStructurePath alloc] initWithComponents:components];
        pathToDebugHandlesMap[path] = debugHandles;
    }

    return [[ETCoreMLModelDebugInfo alloc] initWithPathToDebugSymbolMap:pathToDebugSymbolMap
                                                  pathToDebugHandlesMap:pathToDebugHandlesMap];

}

@end
