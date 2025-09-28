/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_exported import ExecuTorch

public extension TensorMetadata {
  /// The size of each dimension.
  var shape: [Int] { __shape.map(\.intValue) }

  /// The layout order of each dimension.
  var dimensionOrder: [Int] { __dimensionOrder.map(\.intValue) }
}

public extension MethodMetadata {
  /// The declared input tags.
  var inputValueTags: [ValueTag] {
    __inputValueTags.map { ValueTag(rawValue: $0.uint32Value)! }
  }

  /// The declared output tags.
  var outputValueTags: [ValueTag] {
    __outputValueTags.map { ValueTag(rawValue: $0.uint32Value)! }
  }

  /// A dictionary mapping each input index to its `TensorMetadata`.
  var inputTensorMetadata: [Int: TensorMetadata] {
    Dictionary(uniqueKeysWithValues:
      __inputTensorMetadata.map { (key, value) in (key.intValue, value) }
    )
  }

  /// A dictionary mapping each output index to its `TensorMetadata`.
  var outputTensorMetadata: [Int: TensorMetadata] {
    Dictionary(uniqueKeysWithValues:
      __outputTensorMetadata.map { (key, value) in (key.intValue, value) }
    )
  }

  /// The sizes of all memory-planned buffers as a native Swift array of `Int`.
  var memoryPlannedBufferSizes: [Int] {
    __memoryPlannedBufferSizes.map(\.intValue)
  }
}

public extension Module {
  /// Executes a specific method with the provided input values.
  /// The method is loaded on demand if not already loaded.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - inputs: An array of `ValueConvertible` types representing the inputs.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if method execution fails.
  func execute(_ method: String, _ inputs: [ValueConvertible]) throws -> [Value] {
    try __executeMethod(method, withInputs: inputs.map { $0.asValue() } )
  }

  /// Executes a specific method with variadic inputs.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - inputs: A variadic list of `ValueConvertible` inputs.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if loading or execution fails.
  func execute(_ method: String, _ inputs: ValueConvertible...) throws -> [Value] {
    try execute(method, inputs)
  }

  /// Executes the "forward" method with the provided input values.
  /// The method is loaded on demand if not already loaded.
  ///
  /// - Parameter inputs: An array of `ValueConvertible` types representing the inputs.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if method execution fails.
  func forward(_ inputs: [ValueConvertible]) throws -> [Value] {
    try __executeMethod("forward", withInputs: inputs.map { $0.asValue() })
  }

  /// Executes the "forward" method with variadic inputs.
  ///
  /// - Parameter inputs: A variadic list of `ValueConvertible` inputs.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if loading or execution fails.
  func forward(_ inputs: ValueConvertible...) throws -> [Value] {
    try forward(inputs)
  }
}

public extension Module {
  /// Executes a specific method and decodes the outputs into `Output` generic type.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - inputs: An array of `ValueConvertible` inputs.
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func execute<Output: ValueSequenceConstructible>(_ method: String, _ inputs: [ValueConvertible]) throws -> Output {
    try Output(__executeMethod(method, withInputs: inputs.map { $0.asValue() }))
  }

  /// Executes a specific method with variadic inputs and decodes into `Output` generic type.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - inputs: A variadic list of `ValueConvertible` inputs.
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func execute<Output: ValueSequenceConstructible>(_ method: String, _ inputs: ValueConvertible...) throws -> Output {
    try execute(method, inputs)
  }

  /// Executes a specific method with a single input and decodes into `Output` generic type.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - input: A single `ValueConvertible` input.
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func execute<Output: ValueSequenceConstructible>(_ method: String, _ input: ValueConvertible) throws -> Output {
    try execute(method, [input])
  }

  /// Executes a specific method with no inputs and decodes into `Output` generic type.
  ///
  /// - Parameter method: The name of the method to execute.
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func execute<Output: ValueSequenceConstructible>(_ method: String) throws -> Output {
    try execute(method, [])
  }

  /// Executes the "forward" method and decodes into `Output` generic type.
  ///
  /// - Parameters:
  ///   - inputs: An array of `ValueConvertible` inputs to pass to "forward".
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func forward<Output: ValueSequenceConstructible>(_ inputs: [ValueConvertible]) throws -> Output {
    try execute("forward", inputs)
  }

  /// Executes the "forward" method with variadic inputs and decodes into `Output` generic type.
  ///
  /// - Parameters:
  ///   - inputs: A variadic list of `ValueConvertible` inputs.
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func forward<Output: ValueSequenceConstructible>(_ inputs: ValueConvertible...) throws -> Output {
    try forward(inputs)
  }

  /// Executes the "forward" method with a single input and decodes into `Output` generic type.
  ///
  /// - Parameters:
  ///   - input: A single `ValueConvertible` to pass to "forward".
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func forward<Output: ValueSequenceConstructible>(_ input: ValueConvertible) throws -> Output {
    try forward([input])
  }

  /// Executes the "forward" method with no inputs and decodes into `Output` generic type.
  ///
  /// - Returns: An instance of `Output` decoded from the returned `[Value]`, or `nil` on mismatch.
  /// - Throws: An error if loading, execution or result conversion fails.
  func forward<Output: ValueSequenceConstructible>() throws -> Output {
    try execute("forward")
  }
}

public extension Module {
  /// Sets a single input value for a method at the specified index.
  ///
  /// - Parameters:
  ///   - value: The input as a `ValueConvertible`.
  ///   - method: The method name.
  ///   - index: Zero-based input index.
  /// - Throws: If setting the input fails.
  func setInput(_ value: ValueConvertible, for method: String, at index: Int) throws {
    try __setInput(value.asValue(), forMethod: method, at: index)
  }

  /// Sets a single input value for a method at index 0.
  ///
  /// - Parameters:
  ///   - value: The input as a `ValueConvertible`.
  ///   - method: The method name.
  /// - Throws: If setting the input fails.
  func setInput(_ value: ValueConvertible, for method: String) throws {
    try setInput(value, for: method, at: 0)
  }

  /// Sets a single input value for the "forward" method at the specified index.
  ///
  /// - Parameters:
  ///   - value: The input as a `ValueConvertible`.
  ///   - index: Zero-based input index.
  /// - Throws: If setting the input fails.
  func setInput(_ value: ValueConvertible, at index: Int) throws {
    try setInput(value, for: "forward", at: index)
  }

  /// Sets the first input value (index 0) for the "forward" method.
  ///
  /// - Parameter value: The input as a `ValueConvertible`.
  /// - Throws: If setting the input fails.
  func setInput(_ value: ValueConvertible) throws {
    try setInput(value, for: "forward", at: 0)
  }

  /// Sets all input values for a method.
  ///
  /// - Parameters:
  ///   - values: The inputs as an array of `ValueConvertible`.
  ///   - method: The method name.
  /// - Throws: If setting the inputs fails.
  func setInputs(_ values: [ValueConvertible], for method: String) throws {
    try __setInputs(values.map { $0.asValue() }, forMethod: method)
  }

  /// Sets all input values for the "forward" method.
  ///
  /// - Parameter values: The inputs as an array of `ValueConvertible`.
  /// - Throws: If setting the inputs fails.
  func setInputs(_ values: [ValueConvertible]) throws {
    try setInputs(values, for: "forward")
  }

  /// Sets all input values for a method using variadic arguments.
  ///
  /// - Parameters:
  ///   - values: The inputs as a variadic list of `ValueConvertible`.
  ///   - method: The method name.
  /// - Throws: If setting the inputs fails.
  func setInputs(_ values: ValueConvertible..., for method: String) throws {
    try setInputs(values, for: method)
  }

  /// Sets all input values for the "forward" method using variadic arguments.
  ///
  /// - Parameter values: The inputs as a variadic list of `ValueConvertible`.
  /// - Throws: If setting the inputs fails.
  func setInputs(_ values: ValueConvertible...) throws {
    try setInputs(values, for: "forward")
  }

  /// Sets the output location for a method at the specified index.
  ///
  /// Only tensor outputs are supported. The provided value must wrap a tensor
  /// with compatible shape and data type for the method’s output slot.
  ///
  /// - Parameters:
  ///   - value: The output buffer as a `ValueConvertible` (tensor).
  ///   - method: The method name.
  ///   - index: Zero-based output index.
  /// - Throws: If setting the output fails.
  func setOutput(_ value: ValueConvertible, for method: String, at index: Int) throws {
    try __setOutput(value.asValue(), forMethod: method, at: index)
  }

  /// Sets the output location for a method at index 0.
  ///
  /// - Parameters:
  ///   - value: The output buffer as a `ValueConvertible` (tensor).
  ///   - method: The method name.
  /// - Throws: If setting the output fails.
  func setOutput(_ value: ValueConvertible, for method: String) throws {
    try setOutput(value, for: method, at: 0)
  }

  /// Sets the output location for the "forward" method at the specified index.
  ///
  /// - Parameters:
  ///   - value: The output buffer as a `ValueConvertible` (tensor).
  ///   - index: Zero-based output index.
  /// - Throws: If setting the output fails.
  func setOutput(_ value: ValueConvertible, at index: Int) throws {
    try setOutput(value, for: "forward", at: index)
  }

  /// Sets the first output location (index 0) for the "forward" method.
  ///
  /// - Parameter value: The output buffer as a `ValueConvertible` (tensor).
  /// - Throws: If setting the output fails.
  func setOutput(_ value: ValueConvertible) throws {
    try setOutput(value, for: "forward", at: 0)
  }
}
