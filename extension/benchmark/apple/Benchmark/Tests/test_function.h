/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#import <CoreML/CoreML.h>
#import "model_no_kv.h"

BOOL prefill_no_kv_cache(void) {
    model_no_kv *model = [[model_no_kv alloc] init];
    
    MLMultiArray *tokens =  [[MLMultiArray alloc] initWithShape:(NSArray*)(@[@1, @512])
                                              dataType:(MLMultiArrayDataType)MLMultiArrayDataTypeInt32
                                                 error:nil] ;
    
    for (int i = 0; i < 512; i++) {
        tokens[i] = @2;
    }
    model_no_kvInput *inputs = [[model_no_kvInput alloc] initWithTokens:tokens];
    
    for (int i = 0; i < 100; i++) {
        model_no_kvOutput *output = [model predictionFromFeatures:inputs error:nil];
    }
    
    return YES;
}
