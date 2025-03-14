/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@testable import ExecutorchRuntime
import ExecutorchRuntimeValueSupport
import XCTest

class ExecutorchRuntimeTests: XCTestCase {
  func testRuntimeWithAddPTE() throws {
    let bundle = Bundle(for: type(of: self))
    let modelPath = try XCTUnwrap(bundle.path(forResource: "add", ofType: "pte"))
    let runtime = try XCTUnwrap(ExecutorchRuntime(modelPath: modelPath, modelMethodName: "forward"))

    let tensorInput = try XCTUnwrap(runtime.getModelTensorFactory().createFloatTensor(value: [2.0], shape: [1]))
    let input = try runtime.getModelValueFactory().createTensor(value: tensorInput)

    let output = try XCTUnwrap(runtime.infer(input: [input, input]))

    let tensorOutput = try output.first?.tensorValue().floatRepresentation()
    XCTAssertEqual(tensorOutput?.floatArray.count, 1)
    XCTAssertEqual(tensorOutput?.shape.count, 1)
    XCTAssertEqual(tensorOutput?.shape.first, 1)
    XCTAssertEqual(tensorOutput?.floatArray.first, 4.0)
  }
}
