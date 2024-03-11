// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const LLaMARunnerErrorDomain;

NS_SWIFT_NAME(Runner)
@interface LLaMARunner : NSObject

- (instancetype)initWithModelPath:(NSString*)filePath
                    tokenizerPath:(NSString*)tokenizerPath;
- (BOOL)isloaded;
- (BOOL)loadWithError:(NSError**)error;
- (BOOL)generate:(NSString*)prompt
       sequenceLength:(NSInteger)seq_len
    withTokenCallback:(nullable void (^)(NSString*))callback
                error:(NSError**)error;
- (void)stop;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
