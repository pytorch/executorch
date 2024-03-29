/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@testable import LLaMARunner

import XCTest

final class RunnerTest: XCTestCase {

  func test() {
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "xnnpack_dq_llama2", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "flores200sacrebleuspm", ofType: "bin") else {
      XCTFail("Couldn't find model or tokenizer files")
      return
    }
    let runner = Runner(modelPath: modelPath, tokenizerPath: tokenizerPath)
    var text = ""

    do {
      try runner.generate("fr hello", sequenceLength: 128) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertTrue(["bonjour", "salut", "coucou"].map { $0.lowercased() }.contains { text.lowercased().contains($0) })
  }
}
