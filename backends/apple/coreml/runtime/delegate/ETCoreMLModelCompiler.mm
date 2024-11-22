//
// ETCoreMLModelCompiler.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLModelCompiler.h>
#import <ETCoreMLLogging.h>
#import <TargetConditionals.h>

@implementation ETCoreMLModelCompiler

+ (nullable NSURL *)compileModelAtURL:(NSURL *)modelURL
                 maxWaitTimeInSeconds:(NSTimeInterval)maxWaitTimeInSeconds
                                error:(NSError* __autoreleasing *)error {
#if TARGET_OS_WATCH
    (void)modelURL;
    (void)maxWaitTimeInSeconds;
    (void)error;
    ETCoreMLLogErrorAndSetNSError(error,
                                  ETCoreMLErrorModelCompilationNotSupported,
                                  "%@: Model compilation is not supported on the target, please make sure to export a compiled model.",
                                  NSStringFromClass(ETCoreMLModelCompiler.class));
    return nil;
#else
    __block NSError *localError = nil;
    __block NSURL *result = nil;

    if (@available(iOS 16, macOS 13, watchOS 9, tvOS 16, *)) {
        dispatch_semaphore_t sema = dispatch_semaphore_create(0);
        [MLModel compileModelAtURL:modelURL completionHandler:^(NSURL * _Nullable tempURL, NSError * _Nullable compilationError) {
            result = [tempURL copy];
            localError = compilationError;
            dispatch_semaphore_signal(sema);
        }];

        long status = dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(maxWaitTimeInSeconds * NSEC_PER_SEC)));
        if (status != 0) {
            ETCoreMLLogErrorAndSetNSError(error,
                                        ETCoreMLErrorCompilationFailed,
                                        "%@: Failed to compile model in %f seconds.",
                                        NSStringFromClass(ETCoreMLModelCompiler.class),
                                        maxWaitTimeInSeconds);
            return nil;
        }
    } else {
        result = [MLModel compileModelAtURL:modelURL error:&localError];
    }

    if (localError) {
        ETCoreMLLogErrorAndSetNSError(error,
                                    ETCoreMLErrorCompilationFailed,
                                    "%@: Failed to compile model, error: %@",
                                    NSStringFromClass(ETCoreMLModelCompiler.class),
                                    localError);
        return nil;
    } else {
        return result;
    }
#endif
}

@end
