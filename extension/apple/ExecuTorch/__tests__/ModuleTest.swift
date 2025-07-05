/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
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
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))

    let inputs2: [Tensor<Float>] = [Tensor([2]), Tensor([3])]
    var outputs2: [Value]?
    XCTAssertNoThrow(outputs2 = try module.forward(inputs2))
    XCTAssertEqual(outputs2?.first?.tensor(), Tensor([Float(5)]))

    let inputs3: [Tensor<Float>] = [Tensor([13.25]), Tensor([29.25])]
    var outputs3: [Value]?
    XCTAssertNoThrow(outputs3 = try module.forward(inputs3))
    XCTAssertEqual(outputs3?.first?.tensor(), Tensor([Float(42.5)]))

    let lhsScalar: Float = 2
    let rhsScalar: Float = 3
    let lhsTensor = Tensor([lhsScalar])
    let rhsTensor = Tensor([rhsScalar])
    let lhsValue = Value(lhsTensor)
    let rhsValue = Value(rhsTensor)
    var outputs4: [Value]?
    XCTAssertNoThrow(outputs4 = try module.forward([lhsValue, rhsValue]))
    XCTAssertEqual(outputs4?.first?.tensor(), Tensor([Float(5)]))
  }

  func testmethodMetadata() throws {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let methodMetadata = try module.methodMetadata("forward")
    XCTAssertEqual(methodMetadata.name, "forward")
    XCTAssertEqual(methodMetadata.inputValueTags.count, 2)
    XCTAssertEqual(methodMetadata.outputValueTags.count, 1)

    XCTAssertEqual(methodMetadata.inputValueTags[0], .tensor)
    let inputTensorMetadata1 = methodMetadata.inputTensorMetadata[0]
    XCTAssertEqual(inputTensorMetadata1?.shape, [1])
    XCTAssertEqual(inputTensorMetadata1?.dimensionOrder, [0])
    XCTAssertEqual(inputTensorMetadata1?.dataType, .float)
    XCTAssertEqual(inputTensorMetadata1?.isMemoryPlanned, true)
    XCTAssertEqual(inputTensorMetadata1?.name, "")

    XCTAssertEqual(methodMetadata.inputValueTags[1], .tensor)
    let inputTensorMetadata2 = methodMetadata.inputTensorMetadata[1]
    XCTAssertEqual(inputTensorMetadata2?.shape, [1])
    XCTAssertEqual(inputTensorMetadata2?.dimensionOrder, [0])
    XCTAssertEqual(inputTensorMetadata2?.dataType, .float)
    XCTAssertEqual(inputTensorMetadata2?.isMemoryPlanned, true)
    XCTAssertEqual(inputTensorMetadata2?.name, "")

    XCTAssertEqual(methodMetadata.outputValueTags[0], .tensor)
    let outputTensorMetadata = methodMetadata.outputTensorMetadata[0]
    XCTAssertEqual(outputTensorMetadata?.shape, [1])
    XCTAssertEqual(outputTensorMetadata?.dimensionOrder, [0])
    XCTAssertEqual(outputTensorMetadata?.dataType, .float)
    XCTAssertEqual(outputTensorMetadata?.isMemoryPlanned, true)
    XCTAssertEqual(outputTensorMetadata?.name, "")

    XCTAssertEqual(methodMetadata.attributeTensorMetadata.count, 0)
    XCTAssertEqual(methodMetadata.memoryPlannedBufferSizes.count, 1)
    XCTAssertEqual(methodMetadata.memoryPlannedBufferSizes[0], 48)
    XCTAssertEqual(methodMetadata.backendNames.count, 0)
    XCTAssertEqual(methodMetadata.instructionCount, 1)
  }
}
