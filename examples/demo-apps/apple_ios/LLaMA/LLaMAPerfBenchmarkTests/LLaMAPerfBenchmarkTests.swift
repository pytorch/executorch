/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import LLaMARunner
import XCTest

final class LLaMAPerfBenchmarkTests: XCTestCase {
  func testLlama2() throws {
    guard
      let modelPath = Bundle.main.path(
        forResource: "llama2",
        ofType: "pte",
        inDirectory: "aatp/data"
      )
    else {
      XCTFail("Failed to get model path")
      return
    }

    guard
      let tokenizerPath = Bundle.main.path(
        forResource: "tokenizer",
        ofType: "bin",
        inDirectory: "aatp/data"
      )
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
