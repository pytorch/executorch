//
// ETCoreMLLogging.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>
#import <os/log.h>

#import <executorch/runtime/platform/log.h>

NS_ASSUME_NONNULL_BEGIN

/// The error domain for the delegate error.
extern NSErrorDomain const ETCoreMLErrorDomain;

/// The error codes that are exposed publicly.
typedef NS_ERROR_ENUM(ETCoreMLErrorDomain, ETCoreMLError) {
    ETCoreMLErrorCorruptedData = 1, // AOT blob can't be parsed.
    ETCoreMLErrorCorruptedMetadata = 2, // AOT blob has incorrect or missing metadata.
    ETCoreMLErrorCorruptedModel = 3, // AOT blob has incorrect or missing CoreML model.
    ETCoreMLErrorBrokenModel = 4, // CoreML model doesn't match the input and output specification.
    ETCoreMLErrorCompilationFailed = 5, // CoreML model failed to compile.
    ETCoreMLErrorModelCompilationNotSupported = 6, // CoreML model compilation is not supported by the target.
    ETCoreMLErrorModelProfilingNotSupported = 7, // Model profiling is not supported by the target.
    ETCoreMLErrorModelSaveFailed = 8, // Failed to save CoreML model to disk.
    ETCoreMLErrorModelCacheCreationFailed = 9, // Failed to create model cache.
    ETCoreMLErrorInternalError = 10, // Internal error.
};

@interface ETCoreMLErrorUtils : NSObject

+ (NSError*)errorWithCode:(ETCoreMLError)code
          underlyingError:(nullable NSError*)underlyingError
                   format:(nullable NSString*)description, ... NS_FORMAT_FUNCTION(3, 4);

+ (NSError*)errorWithIntegerCode:(NSInteger)code
                 underlyingError:(nullable NSError*)underlyingError
                          format:(nullable NSString*)format
                            args:(va_list)args;

@property (class, strong, readonly, nonatomic, nullable) os_log_t loggingChannel;

@end

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#if ET_LOG_ENABLED
#define ETCoreMLLogError(error, formatString, ...)                                                      \
    do {                                                                                                \
        NSString* message = error.localizedDescription;                                                 \
        message = [NSString stringWithFormat:@"[Core ML] " formatString " %@", ##__VA_ARGS__, message]; \
        ET_LOG(Error, "%s", message.UTF8String);                                                        \
    } while (0)
#else
#define ETCoreMLLogError(error, formatString, ...) \
    os_log_error(ETCoreMLErrorUtils.loggingChannel, formatString " %@", ##__VA_ARGS__, error.localizedDescription)
#endif

#if ET_LOG_ENABLED
#define ETCoreMLLogInfo(formatString, ...) \
    ET_LOG(Info, "%s", [NSString stringWithFormat:@formatString, ##__VA_ARGS__].UTF8String)
#else
#define ETCoreMLLogInfo(formatString, ...) os_log_info(ETCoreMLErrorUtils.loggingChannel, formatString, ##__VA_ARGS__)
#endif

/// Record the error with `os_log_error` and fills `*errorOut` with `NSError`.
#define ETCoreMLLogErrorAndSetNSError(errorOut, errorCode, formatString, ...)                                 \
    do {                                                                                                      \
        NSDictionary* userInfo =                                                                              \
            @{ NSLocalizedDescriptionKey : [NSString stringWithFormat:@formatString, ##__VA_ARGS__] };        \
        NSError* localError = [NSError errorWithDomain:ETCoreMLErrorDomain code:errorCode userInfo:userInfo]; \
        ETCoreMLLogError(localError, "");                                                                     \
        if (errorOut) {                                                                                       \
            *errorOut = localError;                                                                           \
        }                                                                                                     \
    } while (0)

/// Record the error and its underlying error with `os_log_error` and fills `*errorOut` with `NSError`.
#define ETCoreMLLogUnderlyingErrorAndSetNSError(errorOut, errorCode, underlyingNSError, formatString, ...) \
    do {                                                                                                   \
        ETCoreMLLogError(underlyingNSError, formatString, ##__VA_ARGS__);                                  \
        if (errorOut) {                                                                                    \
            *errorOut = [ETCoreMLErrorUtils errorWithCode:errorCode                                        \
                                          underlyingError:underlyingNSError                                \
                                                   format:@formatString, ##__VA_ARGS__];                   \
        }                                                                                                  \
    } while (0)


#pragma clang diagnostic pop

NS_ASSUME_NONNULL_END
