#ifndef LLaMABridge_h
#define LLaMABridge_h

#import <LLaMARunner/LLaMARunner.h>
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

NS_ASSUME_NONNULL_BEGIN

@interface LLaMABridge : RCTEventEmitter <RCTBridgeModule>
@property (nonatomic, strong) LLaMARunner *runner;
@end

NS_ASSUME_NONNULL_END

#endif /* LLaMABridge_h */
