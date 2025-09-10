/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorchLLM
import XCTest

class MultimodalRunnerTest: XCTestCase {
  func test() {
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "llava", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "tokenizer", ofType: "bin") else {
      XCTFail("Couldn't find model or tokenizer files")
      return
    }
    return
    let runner = MultimodalRunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
    var text = ""

    do {
      try runner.generate([MultimodalInput("hello")], sequenceLength: 2) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertEqual("hello,", text.lowercased())
  }
}
