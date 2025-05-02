/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation

public enum Constants {
    public static let qwen3PromptTemplate = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
%@<|im_end|>
<|im_start|>assistant
<think>

</think>



"""
    
    public static let llama3PromptTemplate = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>%@<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
public static let phi4PromptTemplate = "<|user|>%@<|end|><|assistant|>"
}
