/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_exported import ExecuTorch

/// A protocol that types conform to in order to be used as tensor element types.
/// Provides the mapping from the Swift type to the underlying `DataType`.
@available(*, deprecated, message: "This API is experimental.")
public protocol Scalar {
  /// The `DataType` corresponding to this scalar type.
  static var dataType: DataType { get }
  /// Converts the scalar to an `NSNumber`.
  func asNSNumber() -> NSNumber
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt8: Scalar {
  /// The `DataType` corresponding to `UInt8`, which is `.byte`.
  public static var dataType: DataType { .byte }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int8: Scalar {
  /// The `DataType` corresponding to `Int8`, which is `.char`.
  public static var dataType: DataType { .char }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int16: Scalar {
  /// The `DataType` corresponding to `Int16`, which is `.short`.
  public static var dataType: DataType { .short }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int32: Scalar {
  /// The `DataType` corresponding to `Int32`, which is `.int`.
  public static var dataType: DataType { .int }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int64: Scalar {
  /// The `DataType` corresponding to `Int64`, which is `.long`.
  public static var dataType: DataType { .long }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int: Scalar {
  /// The `DataType` corresponding to `Int`, which is `.long`.
  public static var dataType: DataType { .long }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Float: Scalar {
  /// The `DataType` corresponding to `Float`, which is `.float`.
  public static var dataType: DataType { .float }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Double: Scalar {
  /// The `DataType` corresponding to `Double`, which is `.double`.
  public static var dataType: DataType { .double }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Bool: Scalar {
  /// The `DataType` corresponding to `Bool`, which is `.bool`.
  public static var dataType: DataType { .bool }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt16: Scalar {
  /// The `DataType` corresponding to `UInt16`.
  public static var dataType: DataType { .uInt16 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt32: Scalar {
  /// The `DataType` corresponding to `UInt32`.
  public static var dataType: DataType { .uInt32 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt64: Scalar {
  /// The `DataType` corresponding to `UInt64`.
  public static var dataType: DataType { .uInt64 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt: Scalar {
  /// The `DataType` corresponding to `UInt`.
  public static var dataType: DataType { .uInt64 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

/// A tensor class for ExecuTorch operations.
///
/// This class encapsulates a native `ExecuTorchTensor` instance and provides a variety of
/// initializers and utility methods to work with tensor data.
@available(*, deprecated, message: "This API is experimental.")
public class Tensor<T: Scalar>: Equatable {
  /// The data type of the tensor's elements.
  public var dataType: DataType { objcTensor.dataType }

  /// The shape of the tensor.
  public var shape: [Int] { objcTensor.shape.map(\.intValue) }

  /// The strides of the tensor.
  public var strides: [Int] { objcTensor.strides.map(\.intValue) }

  /// The order of dimensions in the tensor.
  public var dimensionOrder: [Int] { objcTensor.dimensionOrder.map(\.intValue) }

  /// The dynamism of the tensor's shape.
  public var shapeDynamism: ShapeDynamism { objcTensor.shapeDynamism }

  /// The total number of elements in the tensor.
  public var count: Int { objcTensor.count }

  /// Initializes a tensor with an `ExecuTorchTensor` instance.
  ///
  /// - Parameter tensor: An `ExecuTorchTensor` instance.
  public init(_ tensor: __ExecuTorchTensor) {
    precondition(tensor.dataType == T.dataType)
    objcTensor = tensor
  }

  /// Creates a new tensor that shares the underlying data storage with the
  /// given tensor. This new tensor is a view and does not own the data.
  ///
  /// - Parameter tensor: The tensor to create a view of.
  public convenience init(_ tensor: Tensor<T>) {
    self.init(__ExecuTorchTensor(tensor.objcTensor))
  }

  /// Initializes a tensor without copying the provided data.
  ///
  /// - Parameters:
  ///   - pointer: A pointer to the data buffer.
  ///   - shape: An array of integers representing the tensor's shape.
  ///   - strides: An array of integers representing the tensor's strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating whether the shape is static or dynamic.
  public convenience init(
    bytesNoCopy pointer: UnsafeMutableRawPointer,
    shape: [Int],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    self.init(__ExecuTorchTensor(
      bytesNoCopy: pointer,
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Initializes a tensor by copying bytes from the provided pointer.
  ///
  /// - Parameters:
  ///   - pointer: A pointer to the source data buffer.
  ///   - shape: An array of integers representing the tensor's shape.
  ///   - strides: An array of integers representing the tensor's strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating the shape dynamism.
  public convenience init(
    bytes pointer: UnsafeRawPointer,
    shape: [Int],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    self.init(__ExecuTorchTensor(
      bytes: pointer,
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Initializes a tensor using a `Data` object. The tensor holds a reference
  /// to the `Data` object to ensure its buffer remains alive. The data is not copied.
  ///
  /// - Parameters:
  ///   - data: A `Data` object containing the tensor data.
  ///   - shape: An array of integers representing the tensor's shape.
  ///   - strides: An array of integers representing the tensor's strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating the shape dynamism.
  public convenience init(
    data: Data,
    shape: [Int],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    self.init(__ExecuTorchTensor(
      data: data,
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Initializes a tensor with an array of scalar values.
  ///
  /// - Parameters:
  ///   - scalars: An array of scalar values.
  ///   - shape: An array of integers representing the desired tensor shape. If empty, the shape is inferred as `[scalars.count]`.
  ///   - strides: An array of integers representing the tensor strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating the shape dynamism.
  public convenience init(
    _ scalars: [T],
    shape: [Int] = [],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    let nsShape = (shape.isEmpty ? [scalars.count] : shape).map(NSNumber.init)
    precondition(scalars.count == elementCount(ofShape: nsShape))
    self.init(scalars.withUnsafeBufferPointer { buffer in
      __ExecuTorchTensor(
        bytes: buffer.baseAddress!,
        shape: nsShape,
        strides: strides.map(NSNumber.init),
        dimensionOrder: dimensionOrder.map(NSNumber.init),
        dataType: T.dataType,
        shapeDynamism: shapeDynamism
      )
    })
  }

  /// Initializes a tensor with a single scalar value.
  ///
  /// - Parameter scalar: A scalar value.
  public convenience init(_ scalar: T) {
    self.init(__ExecuTorchTensor(scalar.asNSNumber(), dataType: T.dataType))
  }

  /// Returns a copy of the tensor.
  ///
  /// - Returns: A new `Tensor` instance that is a duplicate of the current tensor.
  public func copy() -> Tensor<T> {
    Tensor<T>(objcTensor.copy())
  }

  /// Calls the closure with a typed, immutable buffer pointer over the tensor’s elements.
  ///
  /// - Parameter body: A closure that receives an `UnsafeBufferPointer<T>` bound to the tensor’s data.
  /// - Returns: The value returned by `body`.
  /// - Throws: Any error thrown by `body`.
  public func withUnsafeBytes<R>(_ body: (UnsafeBufferPointer<T>) throws -> R) throws -> R {
    var result: Result<R, Error>?
    objcTensor.bytes { pointer, count, _ in
      result = Result { try body(
        UnsafeBufferPointer(
          start: pointer.assumingMemoryBound(to: T.self),
          count: count
        )
      ) }
    }
    return try result!.get()
  }

  /// Calls the closure with a typed, mutable buffer pointer over the tensor’s elements.
  ///
  /// - Parameter body: A closure that receives an `UnsafeMutableBufferPointer<T>` bound to the tensor’s data.
  /// - Returns: The value returned by `body`.
  /// - Throws: Any error thrown by `body`.
  public func withUnsafeMutableBytes<R>(_ body: (UnsafeMutableBufferPointer<T>) throws -> R) throws -> R {
    var result: Result<R, Error>?
    objcTensor.mutableBytes { pointer, count, _ in
      result = Result { try body(
        UnsafeMutableBufferPointer(
          start: pointer.assumingMemoryBound(to: T.self),
          count: count
        )
      ) }
    }
    return try result!.get()
  }

  /// Resizes the tensor to a new shape.
  ///
  /// - Parameter shape: An array of `Int` representing the desired new shape.
  /// - Throws: An error if the resize operation fails.
  public func resize(to shape: [Int]) throws {
    try objcTensor.resize(to: shape.map(NSNumber.init))
  }

  // MARK: Equatable

  /// Determines whether the current tensor is equal to another tensor.
  ///
  /// - Parameters:
  ///   - lhs: The left-hand side tensor.
  ///   - rhs: The right-hand side tensor.
  /// - Returns: `true` if the tensors have the same type, shape, strides, and data; otherwise, `false`.
  public static func == (lhs: Tensor<T>, rhs: Tensor<T>) -> Bool {
    lhs.objcTensor.isEqual(to: rhs.objcTensor)
  }

  // MARK: Internal

  let objcTensor: __ExecuTorchTensor
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Returns the tensor's elements as an array of scalars.
  ///
  /// - Returns: An array of scalars of type `T`.
  /// - Throws: An error if the underlying data cannot be accessed.
  func scalars() throws -> [T] {
    try withUnsafeBytes(Array.init)
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates an empty tensor with the specified properties.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new, empty `Tensor` instance.
  static func empty(
    shape: [Int],
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.empty(
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates an empty tensor with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new, empty `Tensor` instance.
  static func empty(
    like: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.empty(
      like: like.objcTensor,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates a tensor filled with the specified scalar value.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - scalar: The value to fill the tensor with.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with the scalar value.
  static func full(
    shape: [Int],
    scalar: T,
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.full(
      shape: shape.map(NSNumber.init),
      scalar: scalar.asNSNumber(),
      strides: strides.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates a tensor filled with a scalar value, with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - scalar: The value to fill the tensor with.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with the scalar value.
  static func full(
    like: Tensor<T>,
    scalar: T,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.full(
      like: like.objcTensor,
      scalar: scalar.asNSNumber(),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates a tensor filled with ones.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with ones.
  static func ones(
    shape: [Int],
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.ones(
      shape: shape.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates a tensor of ones with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with ones.
  static func ones(
    like: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.ones(
      like: like.objcTensor,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates a tensor filled with zeros.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with zeros.
  static func zeros(
    shape: [Int],
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.zeros(
      shape: shape.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates a tensor of zeros with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with zeros.
  static func zeros(
    like: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.zeros(
      like: like.objcTensor,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates a tensor with random values uniformly distributed in `[0, 1)`.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with random values.
  static func rand(
    shape: [Int],
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.rand(
      shape: shape.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates a tensor with random values with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with random values.
  static func rand(
    like: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.rand(
      like: like.objcTensor,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates a tensor with random values from a normal distribution with mean `0` and variance `1`.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with values from a normal distribution.
  static func randn(
    shape: [Int],
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.randn(
      shape: shape.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates a tensor with random normal values with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with values from a normal distribution.
  static func randn(
    like: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.randn(
      like: like.objcTensor,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Creates a tensor with random integers from `low` (inclusive) to `high` (exclusive).
  ///
  /// - Parameters:
  ///   - low: The inclusive lower bound of the random integer range.
  ///   - high: The exclusive upper bound of the random integer range.
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with random integer values.
  static func randint(
    low: Int,
    high: Int,
    shape: [Int],
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.randint(
      low: low,
      high: high,
      shape: shape.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Creates a tensor with random integers with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `Tensor` instance whose shape and strides are used.
  ///   - low: The inclusive lower bound of the random integer range.
  ///   - high: The exclusive upper bound of the random integer range.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `Tensor` instance filled with random integer values.
  static func randint(
    like: Tensor<T>,
    low: Int,
    high: Int,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(__ExecuTorchTensor.randint(
      like: like.objcTensor,
      low: low,
      high: high,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }
}
