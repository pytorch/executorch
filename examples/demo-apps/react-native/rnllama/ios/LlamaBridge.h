#ifndef LLaMABridge_h
#define LLaMABridge_h

#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>
#import "LLaMARunner.h"

NS_ASSUME_NONNULL_BEGIN

@interface LLaMABridge : RCTEventEmitter <RCTBridgeModule>
@property (nonatomic, strong) LLaMARunner *runner;
@end

NS_ASSUME_NONNULL_END

#endif /* LLaMABridge_h */
