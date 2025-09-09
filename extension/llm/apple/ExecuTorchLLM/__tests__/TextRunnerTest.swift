/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorchLLM
import XCTest

struct SpecialTokens {
  static let kSpecialTokensSize = 256

  static func defaultSpecialTokens() -> [String] {
    var tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|finetune_right_pad_id|>",
      "<|step_id|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|eom_id|>",
      "<|eot_id|>",
      "<|python_tag|>"
    ]
    var reservedIndex = 2
    while tokens.count < kSpecialTokensSize {
      tokens.append("<|reserved_special_token_\(reservedIndex)|>")
      reservedIndex += 1
    }
    return tokens
  }
}

class TextRunnerTest: XCTestCase {
  func test() {
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "llama3_2-1B", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "tokenizer", ofType: "model") else {
      XCTFail("Couldn't find model or tokenizer files")
      return
    }
    let runner = TextRunner(modelPath: modelPath, tokenizerPath: tokenizerPath, specialTokens: SpecialTokens.defaultSpecialTokens())
    var text = ""

    do {
      try runner.generate("hello", sequenceLength: 2) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertEqual("hello,", text.lowercased())
  }
}
