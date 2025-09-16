#ifndef LLaMABridge_h
#define LLaMABridge_h

#import <ExecuTorchLLM/ExecuTorchLLM.h>
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

NS_ASSUME_NONNULL_BEGIN

@interface LLaMABridge : RCTEventEmitter <RCTBridgeModule>
@property (nonatomic, strong) ExecuTorchLLMTextRunner *runner;
@end

NS_ASSUME_NONNULL_END

#endif /* LLaMABridge_h */
