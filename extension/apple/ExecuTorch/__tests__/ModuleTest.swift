/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@testable import ExecuTorch

import XCTest

class ModuleTest: XCTestCase {
  func test() throws {
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
  }
}
