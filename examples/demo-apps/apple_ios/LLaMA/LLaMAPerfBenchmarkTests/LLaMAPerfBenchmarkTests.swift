/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import XCTest
import LLaMARunner

final class LLaMAPerfBenchmarkTests: XCTestCase {
  func testLlama2() throws {
    guard
      let modelPath = Bundle(for: type(of: self))
        .path(forResource: "llama2", ofType: "pte")
    else {
      XCTFail("Failed to get model path")
      return
    }

    guard
      let tokenizerPath = Bundle(for: type(of: self))
        .path(forResource: "tokenizer", ofType: "bin")
    else {
      XCTFail("Failed to get tokenizer path")
      return
    }

    let runner = Runner(modelPath: modelPath, tokenizerPath: tokenizerPath)
    do {
      try runner.load()
    } catch let loadError {
      XCTFail("Failed to load the model: \(loadError)")
    }
    XCTAssertTrue(runner.isloaded())

    let seq_len = 128
    var tokens: [String] = []
    try runner.generate("How do you do! I'm testing llama2 on mobile device", sequenceLength: seq_len) { token in
      tokens.append(token)
    }
  }
}
