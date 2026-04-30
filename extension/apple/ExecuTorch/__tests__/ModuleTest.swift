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

  func testInvalidModuleLoad() {
    let module = Module(filePath: "invalid/path")
    XCTAssertThrowsError(try module.load())
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

  func testForwardReturnConversion() throws {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]

    let outputValues: [Value] = try module.forward(inputs)
    XCTAssertEqual(outputValues, [Value(Tensor<Float>([2]))])

    let outputValue: Value = try module.forward(inputs)
    XCTAssertEqual(outputValue, Value(Tensor<Float>([2])))

    let outputTensors: [Tensor<Float>] = try module.forward(inputs)
    XCTAssertEqual(outputTensors, [Tensor([2])])

    let outputTensor: Tensor<Float> = try module.forward(Tensor<Float>([1]), Tensor<Float>([1]))
    XCTAssertEqual(outputTensor, Tensor([2]))

    let scalars = (try module.forward(Tensor<Float>([1]), Tensor<Float>([1])) as Tensor<Float>).scalars()
    XCTAssertEqual(scalars, [2])

    let scalars2 = try Tensor<Float>(module.forward(Tensor<Float>([1]), Tensor<Float>([1]))).scalars()
    XCTAssertEqual(scalars2, [2])
  }

  func testMethodMetadata() throws {
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

  func testSetInputs() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)

    XCTAssertNoThrow(try module.setInput(Tensor<Float>([2]), at: 1))
    XCTAssertNoThrow(try module.setInput(Tensor<Float>([1])))
    XCTAssertEqual(try module.forward(), Tensor<Float>([3]))

    XCTAssertNoThrow(try module.setInputs(Tensor<Float>([3]), Tensor<Float>([4])))
    XCTAssertEqual(try module.forward(), Tensor<Float>([7]))

    XCTAssertThrowsError(try module.setInputs(Tensor<Float>([1])))
  }

  func testUnloadMethod() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    XCTAssertNoThrow(try module.load("forward"))
    XCTAssertTrue(module.isLoaded("forward"))

    XCTAssertNoThrow(try module.setInputs(Tensor<Float>([1]), Tensor<Float>([2])))
    XCTAssertEqual(try module.forward(), Tensor<Float>([3]))

    XCTAssertTrue(module.unload("forward"))
    XCTAssertFalse(module.isLoaded("forward"))
    XCTAssertFalse(module.unload("forward"))

    XCTAssertThrowsError(try module.forward())
    XCTAssertTrue(module.isLoaded("forward"))
    XCTAssertNoThrow(try module.setInputs(Tensor<Float>([2]), Tensor<Float>([3])))
    XCTAssertEqual(try module.forward(), Tensor<Float>([5]))
  }

  func testBackendOptionCreation() {
    let boolOption = BackendOption("use_cache", true)
    XCTAssertEqual(boolOption.key, "use_cache")
    XCTAssertEqual(boolOption.type, .boolean)
    XCTAssertTrue(boolOption.boolValue)

    let intOption = BackendOption("num_threads", 4)
    XCTAssertEqual(intOption.key, "num_threads")
    XCTAssertEqual(intOption.type, .integer)
    XCTAssertEqual(intOption.intValue, 4)

    let stringOption = BackendOption("compute_unit", "cpu_and_gpu")
    XCTAssertEqual(stringOption.key, "compute_unit")
    XCTAssertEqual(stringOption.type, .string)
    XCTAssertEqual(stringOption.stringValue, "cpu_and_gpu")
  }

  func testLoadWithBackendOptions() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let backendOptions: [String: [BackendOption]] = [
      "SomeBackend": [
        BackendOption("num_threads", 4),
        BackendOption("use_cache", true),
      ]
    ]
    XCTAssertNoThrow(try module.load(backendOptions: backendOptions))
    XCTAssertTrue(module.isLoaded())
  }

  func testLoadWithEmptyBackendOptions() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let backendOptions: [String: [BackendOption]] = [:]
    XCTAssertNoThrow(try module.load(backendOptions: backendOptions))
    XCTAssertTrue(module.isLoaded())
  }

  func testLoadMethodWithBackendOptions() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let backendOptions: [String: [BackendOption]] = [
      "SomeBackend": [
        BackendOption("compute_unit", "cpu_and_gpu"),
      ]
    ]
    XCTAssertNoThrow(try module.load("forward", backendOptions: backendOptions))
    XCTAssertTrue(module.isLoaded("forward"))
  }

  func testLoadWithBackendOptionsThenExecute() {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let backendOptions: [String: [BackendOption]] = [
      "SomeBackend": [
        BackendOption("num_threads", 4),
      ]
    ]
    XCTAssertNoThrow(try module.load(backendOptions: backendOptions))

    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))
  }

  // Regression test: when load(backendOptions:) is followed by a lazy
  // load_method (triggered by forward without an explicit load("forward")),
  // the LoadBackendOptionsMap built by the ObjC wrapper must outlive the
  // wrapper call. The previous implementation built the map on the stack and
  // passed its address into Module::load (which only borrows it per the C++
  // contract); the per-delegate loop in Method::init then dereferenced a
  // dangling pointer in strcmp and crashed with EXC_BAD_ACCESS.
  //
  // The plain add.pte fixture does NOT trigger this because it has zero
  // delegates, so the per-delegate loop never executes. We use a
  // CoreML-delegated add model (add_coreml.pte, generated at CI time by
  // resources/generate_coreml_test_models.py) which has at least one
  // delegate.
  func testLoadWithBackendOptionsThenExecuteOnCoreMLDelegatedModel() throws {
    guard let modelPath = resourceBundle.path(forResource: "add_coreml", ofType: "pte") else {
      throw XCTSkip("add_coreml.pte not bundled — run extension/apple/ExecuTorch/__tests__/resources/generate_coreml_test_models.py to generate it.")
    }
    let module = Module(filePath: modelPath)
    let backendOptions: [String: [BackendOption]] = [
      "CoreMLBackend": [
        BackendOption("compute_unit", "cpu_and_gpu"),
        BackendOption("_use_new_cache", true),
      ]
    ]
    XCTAssertNoThrow(try module.load(backendOptions: backendOptions))
    // No explicit load("forward") here — exercise the lazy load_method path
    // that previously dereferenced a dangling LoadBackendOptionsMap.
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))
  }

  // Regression test: calling load(backendOptions:) twice on the same Module
  // must remain safe. The C++ Module stores a pointer to the ObjC wrapper's
  // ivar map; the wrapper rebuilds the map in place on each call rather than
  // reallocating a new object. This test locks in the contract that the
  // ivar's address is stable across replacements, so a lazy load_method
  // triggered after the second load call dereferences valid (replaced)
  // contents rather than freed memory.
  func testRepeatedLoadWithBackendOptionsThenExecuteOnCoreMLDelegatedModel() throws {
    guard let modelPath = resourceBundle.path(forResource: "add_coreml", ofType: "pte") else {
      throw XCTSkip("add_coreml.pte not bundled — run extension/apple/ExecuTorch/__tests__/resources/generate_coreml_test_models.py to generate it.")
    }
    let module = Module(filePath: modelPath)

    let firstOptions: [String: [BackendOption]] = [
      "CoreMLBackend": [
        BackendOption("compute_unit", "cpu_only"),
      ]
    ]
    XCTAssertNoThrow(try module.load(backendOptions: firstOptions))

    // Replace the options before any method is loaded. The previous storage
    // (and any Spans into it) must be safely torn down and rebuilt without
    // invalidating the stored pointer in the underlying C++ Module.
    let secondOptions: [String: [BackendOption]] = [
      "CoreMLBackend": [
        BackendOption("compute_unit", "cpu_and_gpu"),
        BackendOption("_use_new_cache", true),
      ]
    ]
    XCTAssertNoThrow(try module.load(backendOptions: secondOptions))

    // Lazy load_method via forward() should now see the second options.
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))
  }
}
