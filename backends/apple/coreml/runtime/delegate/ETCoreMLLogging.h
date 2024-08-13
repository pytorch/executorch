//
// ETCoreMLLogging.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import <executorch/runtime/platform/log.h>
#import <os/log.h>

NS_ASSUME_NONNULL_BEGIN

/// The error domain for the delegate error.
extern NSErrorDomain const ETCoreMLErrorDomain;

/// The error codes that are exposed publicly.
typedef NS_ERROR_ENUM(ETCoreMLErrorDomain, ETCoreMLError) {
    ETCoreMLErrorCorruptedData = 1, // AOT blob can't be parsed.
    ETCoreMLErrorCorruptedMetadata, // AOT blob has incorrect or missing metadata.
    ETCoreMLErrorCorruptedModel, // AOT blob has incorrect or missing CoreML model.
    ETCoreMLErrorBrokenModel, // CoreML model doesn't match the input and output specification.
    ETCoreMLErrorCompilationFailed, // CoreML model failed to compile.
    ETCoreMLErrorModelCompilationNotSupported, // CoreML model compilation is not supported by the target.
    ETCoreMLErrorModelProfilingNotSupported, // Model profiling is not supported by the target.
    ETCoreMLErrorModelSaveFailed, // Failed to save CoreML model to disk.
    ETCoreMLErrorModelCacheCreationFailed, // Failed to create model cache.
    ETCoreMLErrorInternalError, // Internal error.
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

/// Record the error with `os_log_error` and fills `*errorOut` with `NSError`.
#define ETCoreMLLogErrorAndSetNSError(errorOut, errorCode, formatString, ...)                                        \
    if (ET_LOG_ENABLED) {                                                                                            \
        ET_LOG(Error, "%s", [NSString stringWithFormat:@formatString, ##__VA_ARGS__].UTF8String);                    \
    } else {                                                                                                         \
        os_log_error(ETCoreMLErrorUtils.loggingChannel, formatString, ##__VA_ARGS__);                                \
    }                                                                                                                \
    if (errorOut) {                                                                                                  \
        *errorOut =                                                                                                  \
            [NSError errorWithDomain:ETCoreMLErrorDomain                                                             \
                                code:errorCode                                                                       \
                            userInfo:@{                                                                              \
                                NSLocalizedDescriptionKey : [NSString stringWithFormat:@formatString, ##__VA_ARGS__] \
                            }];                                                                                      \
    }

/// Record the error and its underlying error with `os_log_error` and fills `*errorOut` with `NSError`.
#define ETCoreMLLogUnderlyingErrorAndSetNSError(errorOut, errorCode, underlyingNSError, formatString, ...) \
    if (ET_LOG_ENABLED) {                                                                                  \
        ET_LOG(Error, "%s", [NSString stringWithFormat:@formatString, ##__VA_ARGS__].UTF8String);          \
    } else {                                                                                               \
        os_log_error(ETCoreMLErrorUtils.loggingChannel,                                                    \
                     formatString ", with underlying error= %@.",                                          \
                     ##__VA_ARGS__,                                                                        \
                     (underlyingNSError).localizedDescription);                                            \
    }                                                                                                      \
    if (errorOut) {                                                                                        \
        *errorOut = [ETCoreMLErrorUtils errorWithCode:errorCode                                            \
                                      underlyingError:underlyingNSError                                    \
                                               format:@formatString, ##__VA_ARGS__];                       \
    }

#define ETCoreMLLogError(error, formatString, ...)                                                \
    if (ET_LOG_ENABLED) {                                                                         \
        ET_LOG(Error, "%s", [NSString stringWithFormat:@formatString, ##__VA_ARGS__].UTF8String); \
    } else {                                                                                      \
        os_log_error(ETCoreMLErrorUtils.loggingChannel,                                           \
                     formatString ", with error= %@.",                                            \
                     ##__VA_ARGS__,                                                               \
                     (error).localizedDescription);                                               \
    }


#pragma clang diagnostic pop

NS_ASSUME_NONNULL_END
