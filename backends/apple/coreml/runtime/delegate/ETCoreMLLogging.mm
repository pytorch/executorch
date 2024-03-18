//
// ETCoreMLLogging.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLLogging.h>

#import <ETCoreMLStrings.h>

const NSErrorDomain ETCoreMLErrorDomain = @"com.apple.executorchcoreml";

static os_log_t LoggingChannel() {
    static dispatch_once_t onceToken;
    static os_log_t coreChannel;
    dispatch_once(&onceToken, ^{
        coreChannel = os_log_create(ETCoreMLStrings.productIdentifier.UTF8String, ETCoreMLStrings.productName.UTF8String);
        if (!coreChannel) {
            os_log_error(OS_LOG_DEFAULT, "Failed to create os_log_t coreChannel");
        }
    });
    return coreChannel;
}


@implementation ETCoreMLErrorUtils

+ (NSError *)errorWithIntegerCode:(NSInteger)code
                  underlyingError:(nullable NSError *)underlyingError
                           format:(nullable NSString *)format
                             args:(va_list)args {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#pragma clang diagnostic ignored "-Wnullable-to-nonnull-conversion"
    NSString *description = format ? [[NSString alloc] initWithFormat:format arguments:args] : nil;
#pragma clang diagnostic pop
    
    NSMutableDictionary *userInfo = [[NSMutableDictionary alloc] init];
    if (description) {
        userInfo[NSLocalizedDescriptionKey] =  description;
    }
    if (underlyingError) {
        userInfo[NSUnderlyingErrorKey] = underlyingError;
    }
    
    return [NSError errorWithDomain:ETCoreMLErrorDomain
                               code:code
                           userInfo:userInfo];
}

+ (NSError *)errorWithCode:(ETCoreMLError)code
           underlyingError:(nullable NSError *)underlyingError
                    format:(nullable NSString *)format
                      args:(va_list)args {
    return [self errorWithIntegerCode:code underlyingError:underlyingError format:format args:args];
}

+ (NSError *)errorWithCode:(ETCoreMLError)type
           underlyingError:(nullable NSError *)underlyingError
                    format:(nullable NSString *)description, ... {
    va_list args;
    va_start(args, description);
    NSError *error = [self errorWithCode:type underlyingError:underlyingError format:description args:args];
    va_end(args);
    return error;
}

+ (os_log_t)loggingChannel {
    return LoggingChannel();
}

@end
