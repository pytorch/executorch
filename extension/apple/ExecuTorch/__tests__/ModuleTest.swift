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

  /// Resolves a fixture by name. In CI (the `CI` env var is set, regardless
  /// of value — matches the convention used by GitHub Actions / Sandcastle /
  /// most CI systems), absence is a hard failure (a thrown non-`XCTSkip`
  /// error → the test is reported as failed, not skipped). Locally, absence
  /// is a soft skip — convenient on dev machines without the CoreML python
  /// deps.
  private func requireFixture(_ name: String, ofType type: String) throws -> String {
    if let path = resourceBundle.path(forResource: name, ofType: type) {
      return path
    }
    let message = "\(name).\(type) not bundled — run extension/apple/ExecuTorch/__tests__/resources/generate_coreml_test_models.py to generate it."
    if ProcessInfo.processInfo.environment["CI"] != nil {
      // Throw a plain Error (NOT XCTSkip) so the test is reported as failed
      // rather than skipped. The thrown error's localizedDescription is the
      // single failure artifact recorded for the test.
      throw NSError(
        domain: "ModuleTest.FixtureMissing",
        code: -1,
        userInfo: [NSLocalizedDescriptionKey: "[CI] \(message)"]
      )
    }
    throw XCTSkip(message)
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

  func testBackendOptionEqualityHashAndDescription() {
    // Equality and hash agree on equal contents, differ on any field.
    XCTAssertEqual(BackendOption("k", true), BackendOption("k", true))
    XCTAssertEqual(BackendOption("k", 4), BackendOption("k", 4))
    XCTAssertEqual(BackendOption("k", "v"), BackendOption("k", "v"))

    XCTAssertNotEqual(BackendOption("k", true), BackendOption("k", false))
    XCTAssertNotEqual(BackendOption("k", 4), BackendOption("k", 5))
    XCTAssertNotEqual(BackendOption("k", "v"), BackendOption("k", "w"))
    XCTAssertNotEqual(BackendOption("k1", 4), BackendOption("k2", 4))
    // Different types with same key are not equal.
    XCTAssertNotEqual(BackendOption("k", 1), BackendOption("k", true))

    XCTAssertEqual(
      BackendOption("k", true).hashValue,
      BackendOption("k", true).hashValue
    )
    // Set membership works.
    let set: Set<BackendOption> = [BackendOption("k", 1), BackendOption("k", 1)]
    XCTAssertEqual(set.count, 1)

    // Description is human-readable, not a pointer.
    let desc = BackendOption("compute_unit", "cpu_and_gpu").description
    XCTAssertTrue(desc.contains("compute_unit"))
    XCTAssertTrue(desc.contains("cpu_and_gpu"))
    XCTAssertFalse(desc.contains("0x"), "description should not include a pointer: \(desc)")
  }

  func testLoadWithBackendOptions() throws {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let options = try BackendOptionsMap(options: [
      "SomeBackend": [
        BackendOption("num_threads", 4),
        BackendOption("use_cache", true),
      ]
    ])
    XCTAssertNoThrow(try module.load(options))
    XCTAssertTrue(module.isLoaded())
  }

  func testLoadWithEmptyBackendOptions() throws {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let options = try BackendOptionsMap(options: [:])
    XCTAssertNoThrow(try module.load(options))
    XCTAssertTrue(module.isLoaded())
  }

  func testLoadMethodWithBackendOptions() throws {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let options = try BackendOptionsMap(options: [
      "SomeBackend": [
        BackendOption("compute_unit", "cpu_and_gpu"),
      ]
    ])
    XCTAssertNoThrow(try module.load("forward", options: options))
    XCTAssertTrue(module.isLoaded("forward"))
  }

  func testLoadWithBackendOptionsThenExecute() throws {
    guard let modelPath = resourceBundle.path(forResource: "add", ofType: "pte") else {
      XCTFail("Couldn't find the model file")
      return
    }
    let module = Module(filePath: modelPath)
    let options = try BackendOptionsMap(options: [
      "SomeBackend": [
        BackendOption("num_threads", 4),
      ]
    ])
    XCTAssertNoThrow(try module.load(options))

    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))
  }

  // Regression test: when load(_:BackendOptionsMap) is followed by a lazy
  // load_method (triggered by forward without an explicit load("forward")),
  // the C++ LoadBackendOptionsMap held inside the BackendOptionsMap must
  // outlive the wrapper call. The Module retains the BackendOptionsMap via
  // ARC for exactly that reason. The per-delegate loop in Method::init
  // would otherwise dereference a dangling pointer in strcmp and crash
  // with EXC_BAD_ACCESS.
  //
  // The plain add.pte fixture does NOT trigger this because it has zero
  // delegates, so the per-delegate loop never executes. We use a
  // CoreML-delegated add model (add_coreml.pte, generated at CI time by
  // resources/generate_coreml_test_models.py) which has at least one
  // delegate.
  func testLoadWithBackendOptionsThenExecuteOnCoreMLDelegatedModel() throws {
    let modelPath = try requireFixture("add_coreml", ofType: "pte")
    let module = Module(filePath: modelPath)
    let options = try BackendOptionsMap(options: [
      "CoreMLBackend": [
        BackendOption("compute_unit", "cpu_and_gpu"),
        BackendOption("_use_new_cache", true),
      ]
    ])
    XCTAssertNoThrow(try module.load(options))
    // No explicit load("forward") here — exercise the lazy load_method path
    // that previously dereferenced a dangling LoadBackendOptionsMap.
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))
  }

  // Regression test: calling load(_:BackendOptionsMap) twice on the same
  // Module must remain safe. The Module retains the most recently passed
  // map via ARC; the previous one is released only after the new one is
  // installed, so the C++ pointer it stored is always valid.
  func testRepeatedLoadWithBackendOptionsThenExecuteOnCoreMLDelegatedModel() throws {
    let modelPath = try requireFixture("add_coreml", ofType: "pte")
    let module = Module(filePath: modelPath)

    let firstOptions = try BackendOptionsMap(options: [
      "CoreMLBackend": [
        BackendOption("compute_unit", "cpu_only"),
      ]
    ])
    XCTAssertNoThrow(try module.load(firstOptions))

    let secondOptions = try BackendOptionsMap(options: [
      "CoreMLBackend": [
        BackendOption("compute_unit", "cpu_and_gpu"),
        BackendOption("_use_new_cache", true),
      ]
    ])
    XCTAssertNoThrow(try module.load(secondOptions))

    // Lazy load_method via forward() should now see the second options.
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    var outputs: [Value]?
    XCTAssertNoThrow(outputs = try module.forward(inputs))
    XCTAssertEqual(outputs?.first?.tensor(), Tensor([Float(2)]))
  }

  // Validation happens at BackendOptionsMap construction time. The current
  // C++ runtime stores integer option values as 32-bit `int` and stores
  // string keys/values in fixed-size buffers, so values outside the
  // representable range or strings exceeding the buffer must surface as a
  // thrown error rather than silently truncating.
  func testBackendOptionsMapDescription() throws {
    // Empty map renders without a trailing space artifact.
    let empty = try BackendOptionsMap(options: [:])
    XCTAssertEqual(empty.description, "<ExecuTorchBackendOptionsMap (empty)>")

    // Populated map renders compactly and includes both backend id and option keys.
    let populated = try BackendOptionsMap(options: [
      "CoreMLBackend": [BackendOption("compute_unit", "cpu_only")]
    ])
    let desc = populated.description
    XCTAssertTrue(desc.contains("CoreMLBackend"), desc)
    XCTAssertTrue(desc.contains("compute_unit"), desc)
    XCTAssertTrue(desc.contains("cpu_only"), desc)
    XCTAssertFalse(desc.contains("\n"), "description should be a single line: \(desc)")
  }

  func testBackendOptionsMapValidation() {
    // Integer overflow.
    XCTAssertThrowsError(try BackendOptionsMap(options: [
      "AnyBackend": [BackendOption("too_big", Int(Int32.max) + 1)]
    ]))
    XCTAssertThrowsError(try BackendOptionsMap(options: [
      "AnyBackend": [BackendOption("too_small", Int(Int32.min) - 1)]
    ]))
    // Oversized key. The C++ kMaxOptionKeyLength is small (32 chars at the
    // time of writing); 256 bytes is well over any plausible bound.
    let longKey = String(repeating: "k", count: 256)
    XCTAssertThrowsError(try BackendOptionsMap(options: [
      "AnyBackend": [BackendOption(longKey, 1)]
    ]))
    // Oversized string value. kMaxOptionValueLength is also a small fixed
    // buffer; 4096 bytes will exceed it on every supported runtime.
    let longValue = String(repeating: "v", count: 4096)
    XCTAssertThrowsError(try BackendOptionsMap(options: [
      "AnyBackend": [BackendOption("compute_unit", longValue)]
    ]))
  }

  // A single BackendOptionsMap can be reused across multiple Module
  // instances without copying. Each Module retains it independently via ARC.
  func testBackendOptionsMapReusedAcrossModules() throws {
    let modelPath = try requireFixture("add_coreml", ofType: "pte")
    let options = try BackendOptionsMap(options: [
      "CoreMLBackend": [BackendOption("compute_unit", "cpu_only")]
    ])

    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    for _ in 0..<2 {
      let module = Module(filePath: modelPath)
      try module.load(options)
      let outs: [Value] = try module.forward(inputs)
      XCTAssertEqual(outs.first?.tensor(), Tensor([Float(2)]))
    }
  }
}
