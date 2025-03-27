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
  var resourceBundle: Bundle {
#if SWIFT_PACKAGE
    return Bundle.module
#else
    return Bundle(for: type(of: self))
#endif
  }

  func testLoad() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    XCTAssertNoThrow(try module.load())
    XCTAssertTrue(module.isLoaded())
  }
}
