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

  func testLoadMethod() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    XCTAssertNoThrow(try module.load("forward"))
    XCTAssertTrue(module.isLoaded("forward"))
  }

  func testMethodNames() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    var methodNames: Set<String>?
    XCTAssertNoThrow(methodNames = try module.methodNames())
    XCTAssertEqual(methodNames, Set(["forward"]))
  }

  func testExecute() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let inputs = [Tensor([1], dataType: .float), Tensor([1], dataType: .float)]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor, Tensor([2], dataType: .float, shapeDynamism: .static))

    let inputs2 = [Tensor([2], dataType: .float), Tensor([3], dataType: .float)]
    var outputs2: [Value]?
    XCTAssertNoThrow(outputs2 = try module.forward(inputs2))
    XCTAssertEqual(outputs2?.first?.tensor, Tensor([5], dataType: .float, shapeDynamism: .static))

    let inputs3 = [Tensor([13.25], dataType: .float), Tensor([29.25], dataType: .float)]
    var outputs3: [Value]?
    XCTAssertNoThrow(outputs3 = try module.forward(inputs3))
    XCTAssertEqual(outputs3?.first?.tensor, Tensor([42.5], dataType: .float, shapeDynamism: .static))
  }
}
