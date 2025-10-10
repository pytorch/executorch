#import "LLaMABridge.h"

@implementation LLaMABridge

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

- (NSArray<NSString *> *)supportedEvents {
  return @[@"onToken", @"onError"];
}

RCT_EXPORT_METHOD(initialize:(NSString *)modelPath
                  tokenizerPath:(NSString *)tokenizerPath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    self.runner = [[ExecuTorchLLMTextRunner alloc] initWithModelPath:modelPath
                                                       tokenizerPath:tokenizerPath];

    NSError *error = nil;
    if (![self.runner loadWithError:&error]) {
      reject(@"load_error", error.localizedDescription, error);
      return;
    }
    
    resolve(@YES);
  });
}

RCT_EXPORT_METHOD(generate:(NSString *)prompt
                  sequenceLength:(nonnull NSNumber *)seqLen
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSError *error = nil;
    BOOL success = [self.runner generate:prompt
                          sequenceLength:[seqLen integerValue]
                       withTokenCallback:^(NSString *token) {
      [self sendEventWithName:@"onToken" body:token];
    } error:&error];
    
    if (!success) {
      reject(@"generation_error", error.localizedDescription, error);
      return;
    }
    
    resolve(@YES);
  });
}

RCT_EXPORT_METHOD(stop) {
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    [self.runner stop];
  });
}

@end
