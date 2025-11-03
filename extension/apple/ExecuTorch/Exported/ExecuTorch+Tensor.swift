/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_exported import ExecuTorch

/// Computes the total number of elements in a tensor based on its shape.
///
/// - Parameter shape: An array of integers, where each element represents a dimension size.
/// - Returns: An integer equal to the product of the sizes of all dimensions.
public func elementCount(ofShape shape: [Int]) -> Int {
  __ExecuTorchElementCountOfShape(shape.map(NSNumber.init))
}

/// A protocol that types conform to in order to be used as tensor element types.
/// Provides the mapping from the Swift type to the underlying `DataType`.
public protocol Scalar {
  /// The `DataType` corresponding to this scalar type.
  static var dataType: DataType { get }
  /// Converts the scalar to an `NSNumber`.
  func asNSNumber() -> NSNumber
}

extension UInt8: Scalar {
  /// The `DataType` corresponding to `UInt8`, which is `.byte`.
  public static var dataType: DataType { .byte }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Int8: Scalar {
  /// The `DataType` corresponding to `Int8`, which is `.char`.
  public static var dataType: DataType { .char }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Int16: Scalar {
  /// The `DataType` corresponding to `Int16`, which is `.short`.
  public static var dataType: DataType { .short }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Int32: Scalar {
  /// The `DataType` corresponding to `Int32`, which is `.int`.
  public static var dataType: DataType { .int }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Int64: Scalar {
  /// The `DataType` corresponding to `Int64`, which is `.long`.
  public static var dataType: DataType { .long }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Int: Scalar {
  /// The `DataType` corresponding to `Int`, which is `.long`.
  public static var dataType: DataType { .long }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Float: Scalar {
  /// The `DataType` corresponding to `Float`, which is `.float`.
  public static var dataType: DataType { .float }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Double: Scalar {
  /// The `DataType` corresponding to `Double`, which is `.double`.
  public static var dataType: DataType { .double }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension Bool: Scalar {
  /// The `DataType` corresponding to `Bool`, which is `.bool`.
  public static var dataType: DataType { .bool }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension UInt16: Scalar {
  /// The `DataType` corresponding to `UInt16`.
  public static var dataType: DataType { .uInt16 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension UInt32: Scalar {
  /// The `DataType` corresponding to `UInt32`.
  public static var dataType: DataType { .uInt32 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension UInt64: Scalar {
  /// The `DataType` corresponding to `UInt64`.
  public static var dataType: DataType { .uInt64 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

extension UInt: Scalar {
  /// The `DataType` corresponding to `UInt`.
  public static var dataType: DataType { .uInt64 }
  /// Returns the value as an `NSNumber`.
  public func asNSNumber() -> NSNumber { NSNumber(value: self) }
}

/// A type-erasing tensor class for ExecuTorch operations.
public extension AnyTensor {
  /// The shape of the tensor.
  var shape: [Int] { __shape.map(\.intValue) }

  /// The strides of the tensor.
  var strides: [Int] { __strides.map(\.intValue) }

  /// The order of dimensions in the tensor.
  var dimensionOrder: [Int] { __dimensionOrder.map(\.intValue) }

  /// The total number of elements in the tensor.
  var count: Int { __count }

  /// Creates a new tensor that shares the underlying data storage with the
  /// given tensor, with metadata overrides. An empty array for
  /// a parameter signifies that it should be inherited or derived.
  ///
  /// - Parameters:
  ///   - tensor: The tensor instance to create a view of.
  ///   - shape: An override for the tensor's shape.
  ///   - dimensionOrder: An override for the tensor's dimension order.
  ///   - strides: An override for the tensor's strides.
  convenience init(
    _ tensor: AnyTensor,
    shape: [Int] = [],
    dimensionOrder: [Int] = [],
    strides: [Int] = []
  ) {
    self.init(
      __tensor: tensor,
      shape: shape.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      strides: strides.map(NSNumber.init)
    )
  }

  /// Initializes a tensor without copying the provided data.
  ///
  /// - Parameters:
  ///   - pointer: A pointer to the data buffer.
  ///   - shape: An array of integers representing the tensor's shape.
  ///   - strides: An array of integers representing the tensor's strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating whether the shape is static or dynamic.
  convenience init(
    bytesNoCopy pointer: UnsafeMutableRawPointer,
    shape: [Int],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    self.init(
      __bytesNoCopy:    pointer,
      shape:          shape.map(NSNumber.init),
      strides:        strides.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      dataType:       dataType,
      shapeDynamism:  shapeDynamism
    )
  }

  /// Initializes a tensor by copying bytes from the provided pointer.
  ///
  /// - Parameters:
  ///   - pointer: A pointer to the source data buffer.
  ///   - shape: An array of integers representing the tensor's shape.
  ///   - strides: An array of integers representing the tensor's strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating the shape dynamism.
  convenience init(
    bytes pointer: UnsafeRawPointer,
    shape: [Int],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    self.init(
      __bytes: pointer,
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Initializes a tensor using a `Data` object. The tensor holds a reference
  /// to the `Data` object to ensure its buffer remains alive. The data is not copied.
  ///
  /// - Parameters:
  ///   - data: A `Data` object containing the tensor data.
  ///   - shape: An array of integers representing the tensor's shape.
  ///   - strides: An array of integers representing the tensor's strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating the shape dynamism.
  convenience init(
    data: Data,
    shape: [Int],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    self.init(
      __data: data,
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dimensionOrder: dimensionOrder.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Resizes the tensor to a new shape.
  ///
  /// - Parameter shape: An array of `Int` representing the desired new shape.
  /// - Throws: An error if the resize operation fails.
  func resize(to shape: [Int]) throws {
    try __resize(toShape: shape.map(NSNumber.init))
  }

  // MARK: Equatable

  /// Determines whether the current tensor is equal to another tensor.
  ///
  /// - Parameters:
  ///   - lhs: The left-hand side tensor.
  ///   - rhs: The right-hand side tensor.
  /// - Returns: `true` if the tensors have the same type, shape, strides, and data; otherwise, `false`.
  static func == (lhs: AnyTensor, rhs: AnyTensor) -> Bool {
    lhs.__isEqual(to: rhs)
  }

  /// Attempts to convert this type-erased `AnyTensor` into a strongly-typed `Tensor<T>`.
  ///
  /// - Returns: A `Tensor<T>` if the runtime data type matches, otherwise `nil`.
  func asTensor<T: Scalar>() -> Tensor<T>? {
    guard dataType == T.dataType else { return nil }
    return Tensor<T>(self)
  }
}

public extension AnyTensor {
  /// Creates an empty tensor with the specified properties.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new, empty `AnyTensor` instance.
  static func empty(
    shape: [Int],
    strides: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __empty(
      withShape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates an empty tensor with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new, empty `AnyTensor` instance.
  static func empty(
    like tensor: AnyTensor,
    dataType: DataType = .undefined,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __emptyTensorLike(
      tensor,
      dataType: dataType == .undefined ? tensor.dataType : dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

public extension AnyTensor {
  /// Creates a tensor filled with the specified scalar value.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - scalar: The value to fill the tensor with.
  ///   - strides: An array of integers representing the desired strides.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with the scalar value.
  static func full<T: Scalar>(
    shape: [Int],
    scalar: T,
    strides: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __fullTensor(
      withShape: shape.map(NSNumber.init),
      scalar: scalar.asNSNumber(),
      strides: strides.map(NSNumber.init),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates a tensor filled with a scalar value, with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - scalar: The value to fill the tensor with.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with the scalar value.
  static func full<T: Scalar>(
    like tensor: AnyTensor,
    scalar: T,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __fullTensorLike(
      tensor,
      scalar: scalar.asNSNumber(),
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

public extension AnyTensor {
  /// Creates a tensor filled with ones.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with ones.
  static func ones(
    shape: [Int],
    strides: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __onesTensor(
      withShape: shape.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates a tensor of ones with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with ones.
  static func ones(
    like tensor: AnyTensor,
    dataType: DataType = .undefined,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __onesTensorLike(
      tensor,
      dataType: dataType == .undefined ? tensor.dataType : dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

public extension AnyTensor {
  /// Creates a tensor filled with zeros.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with zeros.
  static func zeros(
    shape: [Int],
    strides: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __zerosTensor(
      withShape: shape.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates a tensor of zeros with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with zeros.
  static func zeros(
    like tensor: AnyTensor,
    dataType: DataType = .undefined,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __zerosTensorLike(
      tensor,
      dataType: dataType == .undefined ? tensor.dataType : dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

public extension AnyTensor {
  /// Creates a tensor with random values uniformly distributed in `[0, 1)`.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with random values.
  static func rand(
    shape: [Int],
    strides: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __randomTensor(
      withShape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates a tensor with random values with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with random values.
  static func rand(
    like tensor: AnyTensor,
    dataType: DataType = .undefined,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __randomTensorLike(
      tensor,
      dataType: dataType == .undefined ? tensor.dataType : dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

public extension AnyTensor {
  /// Creates a tensor with random values from a normal distribution with mean `0` and variance `1`.
  ///
  /// - Parameters:
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with values from a normal distribution.
  static func randn(
    shape: [Int],
    strides: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __randomNormalTensor(
      withShape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates a tensor with random normal values with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with values from a normal distribution.
  static func randn(
    like tensor: AnyTensor,
    dataType: DataType = .undefined,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __randomNormalTensorLike(
      tensor,
      dataType: dataType == .undefined ? tensor.dataType : dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

public extension AnyTensor {
  /// Creates a tensor with random integers from `low` (inclusive) to `high` (exclusive).
  ///
  /// - Parameters:
  ///   - low: The inclusive lower bound of the random integer range.
  ///   - high: The exclusive upper bound of the random integer range.
  ///   - shape: An array of integers representing the desired shape.
  ///   - strides: An array of integers representing the desired strides.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with random integer values.
  static func randint(
    low: Int,
    high: Int,
    shape: [Int],
    strides: [Int] = [],
    dataType: DataType,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __randomIntegerTensor(
      withLow: low,
      high: high,
      shape: shape.map(NSNumber.init),
      strides: strides.map(NSNumber.init),
      dataType: dataType,
      shapeDynamism: shapeDynamism
    )
  }

  /// Creates a tensor with random integers with the same properties as a given tensor.
  ///
  /// - Parameters:
  ///   - like: An existing `AnyTensor` instance whose shape and strides are used.
  ///   - low: The inclusive lower bound of the random integer range.
  ///   - high: The exclusive upper bound of the random integer range.
  ///   - dataType: A `DataType` value specifying the element type.
  ///   - shapeDynamism: A value specifying whether the shape is static or dynamic.
  /// - Returns: A new `AnyTensor` instance filled with random integer values.
  static func randint(
    like tensor: AnyTensor,
    low: Int,
    high: Int,
    dataType: DataType = .undefined,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> AnyTensor {
    __randomIntegerTensorLike(
      tensor,
      low: low,
      high: high,
      dataType: dataType == .undefined ? tensor.dataType : dataType,
      shapeDynamism: shapeDynamism
    )
  }
}

/// A generic tensor class for ExecuTorch operations.
///
/// This class encapsulates a type-erasing `AnyTensor` instance and provides a variety of
/// initializers and utility methods to work with tensor data.
public final class Tensor<T: Scalar>: Equatable {
  /// The data type of the tensor's elements.
  public var dataType: DataType { anyTensor.dataType }

  /// The shape of the tensor.
  public var shape: [Int] { anyTensor.shape }

  /// The strides of the tensor.
  public var strides: [Int] { anyTensor.strides }

  /// The order of dimensions in the tensor.
  public var dimensionOrder: [Int] { anyTensor.dimensionOrder }

  /// The dynamism of the tensor's shape.
  public var shapeDynamism: ShapeDynamism { anyTensor.shapeDynamism }

  /// The total number of elements in the tensor.
  public var count: Int { anyTensor.count }

  /// Initializes a tensor with an `AnyTensor` instance.
  ///
  /// - Parameter tensor: An `AnyTensor` instance.
  public init(_ tensor: AnyTensor) {
    precondition(tensor.dataType == T.dataType)
    anyTensor = tensor
  }

  /// Creates a new tensor that shares the underlying data storage with the
  /// given tensor, with optional metadata overrides. An empty array for
  /// a parameter signifies that it should be inherited or derived.
  ///
  /// - Parameters:
  ///   - tensor: The tensor to create a view of.
  ///   - shape: An override for the tensor's shape.
  ///   - dimensionOrder: An override for the tensor's dimension order.
  ///   - strides: An override for the tensor's strides.
  public convenience init(
    _ tensor: Tensor<T>,
    shape: [Int] = [],
    dimensionOrder: [Int] = [],
    strides: [Int] = []
  ) {
    self.init(
      AnyTensor(
        tensor.anyTensor,
        shape: shape,
        dimensionOrder: dimensionOrder,
        strides: strides
      )
    )
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
    self.init(AnyTensor(
      bytesNoCopy: pointer,
      shape: shape,
      strides: strides,
      dimensionOrder: dimensionOrder,
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
    self.init(AnyTensor(
      bytes: pointer,
      shape: shape,
      strides: strides,
      dimensionOrder: dimensionOrder,
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
    self.init(AnyTensor(
      data: data,
      shape: shape,
      strides: strides,
      dimensionOrder: dimensionOrder,
      dataType: T.dataType,
      shapeDynamism: shapeDynamism
    ))
  }

  /// Initializes a tensor without copying the data from an existing array.
  ///
  /// - Parameters:
  ///   - scalars: An `inout` array of scalar values to share memory with.
  ///   - shape: An array of integers representing the desired tensor shape. If empty, the shape is inferred as `[scalars.count]`.
  ///   - strides: An array of integers representing the tensor strides.
  ///   - dimensionOrder: An array of integers indicating the order of dimensions.
  ///   - shapeDynamism: A `ShapeDynamism` value indicating the shape dynamism.
  public convenience init(
    _ scalars: inout [T],
    shape: [Int] = [],
    strides: [Int] = [],
    dimensionOrder: [Int] = [],
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) {
    let newShape = shape.isEmpty ? [scalars.count] : shape
    precondition(scalars.count == elementCount(ofShape: newShape))
    self.init(scalars.withUnsafeMutableBufferPointer {
      AnyTensor(
        bytesNoCopy: $0.baseAddress!,
        shape: newShape,
        strides: strides,
        dimensionOrder: dimensionOrder,
        dataType: T.dataType,
        shapeDynamism: shapeDynamism
      )
    })
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
    let newShape = shape.isEmpty ? [scalars.count] : shape
    precondition(scalars.count == elementCount(ofShape: newShape))
    self.init(scalars.withUnsafeBufferPointer {
      AnyTensor(
        bytes: $0.baseAddress!,
        shape: newShape,
        strides: strides,
        dimensionOrder: dimensionOrder,
        dataType: T.dataType,
        shapeDynamism: shapeDynamism
      )
    })
  }

  /// Initializes a tensor with a single scalar value.
  ///
  /// - Parameter scalar: A scalar value.
  public convenience init(_ scalar: T) {
    self.init(AnyTensor(__scalar: scalar.asNSNumber(), dataType: T.dataType))
  }

  /// Returns a copy of the tensor.
  ///
  /// - Returns: A new `Tensor` instance that is a duplicate of the current tensor.
  public func copy() -> Tensor<T> {
    Tensor<T>(anyTensor.copy())
  }

  /// Returns a copy of the tensor, converted to the specified scalar type.
  ///
  /// - Parameter dataType: The target scalar type.
  /// - Returns: A new tensor with the same shape and metadata but converted elements.
  public func copy<U: Scalar>(to dataType: U.Type) -> Tensor<U> {
    Tensor<U>(anyTensor.copy(to: U.dataType))
  }

  /// Calls the closure with a typed, immutable buffer pointer over the tensor’s elements.
  ///
  /// - Parameter body: A closure that receives an `UnsafeBufferPointer<T>` bound to the tensor’s data.
  /// - Returns: The value returned by `body`.
  /// - Throws: Any error thrown by `body`.
  public func withUnsafeBytes<R>(_ body: (UnsafeBufferPointer<T>) throws -> R) rethrows -> R {
    try withoutActuallyEscaping(body) { body in
      var result: Result<R, Error>?
      anyTensor.bytes { pointer, count, _ in
        result = Result { try body(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: T.self), count: count)) }
      }
      return try result!.get()
    }
  }

  /// Calls the closure with a typed, mutable buffer pointer over the tensor’s elements.
  ///
  /// - Parameter body: A closure that receives an `UnsafeMutableBufferPointer<T>` bound to the tensor’s data.
  /// - Returns: The value returned by `body`.
  /// - Throws: Any error thrown by `body`.
  public func withUnsafeMutableBytes<R>(_ body: (UnsafeMutableBufferPointer<T>) throws -> R) rethrows -> R {
    try withoutActuallyEscaping(body) { body in
      var result: Result<R, Error>?
      anyTensor.mutableBytes { pointer, count, _ in
        result = Result { try body(UnsafeMutableBufferPointer(start: pointer.assumingMemoryBound(to: T.self), count: count)) }
      }
      return try result!.get()
    }
  }

  /// Resizes the tensor to a new shape.
  ///
  /// - Parameter shape: An array of `Int` representing the desired new shape.
  /// - Throws: An error if the resize operation fails.
  public func resize(to shape: [Int]) throws {
    try anyTensor.resize(to: shape)
  }

  // MARK: Equatable

  /// Determines whether the current tensor is equal to another tensor.
  ///
  /// - Parameters:
  ///   - lhs: The left-hand side tensor.
  ///   - rhs: The right-hand side tensor.
  /// - Returns: `true` if the tensors have the same type, shape, strides, and data; otherwise, `false`.
  public static func == (lhs: Tensor<T>, rhs: Tensor<T>) -> Bool {
    lhs.anyTensor == rhs.anyTensor
  }

  // Wrapped AnyTensor instance.
  public let anyTensor: AnyTensor
}

public extension Tensor {
  /// Returns the tensor's elements as an array of scalars.
  ///
  /// - Returns: An array of scalars of type `T`.
  func scalars() -> [T] {
    withUnsafeBytes { Array($0) }
  }
}

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
    Tensor<T>(AnyTensor.empty(
      shape: shape,
      strides: strides,
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
    like tensor: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.empty(
      like: tensor.anyTensor,
      shapeDynamism: shapeDynamism
    ))
  }
}

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
    Tensor<T>(AnyTensor.full(
      shape: shape,
      scalar: scalar,
      strides: strides,
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
    like tensor: Tensor<T>,
    scalar: T,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.full(
      like: tensor.anyTensor,
      scalar: scalar,
      shapeDynamism: shapeDynamism
    ))
  }
}

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
    Tensor<T>(AnyTensor.ones(
      shape: shape,
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
    like tensor: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.ones(
      like: tensor.anyTensor,
      shapeDynamism: shapeDynamism
    ))
  }
}

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
    Tensor<T>(AnyTensor.zeros(
      shape: shape,
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
    like tensor: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.zeros(
      like: tensor.anyTensor,
      shapeDynamism: shapeDynamism
    ))
  }
}

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
    Tensor<T>(AnyTensor.rand(
      shape: shape,
      strides: strides,
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
    like tensor: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.rand(
      like: tensor.anyTensor,
      shapeDynamism: shapeDynamism
    ))
  }
}

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
    Tensor<T>(AnyTensor.randn(
      shape: shape,
      strides: strides,
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
    like tensor: Tensor<T>,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.randn(
      like: tensor.anyTensor,
      shapeDynamism: shapeDynamism
    ))
  }
}

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
    Tensor<T>(AnyTensor.randint(
      low: low,
      high: high,
      shape: shape,
      strides: strides,
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
    like tensor: Tensor<T>,
    low: Int,
    high: Int,
    shapeDynamism: ShapeDynamism = .dynamicBound
  ) -> Tensor<T> {
    Tensor<T>(AnyTensor.randint(
      like: tensor.anyTensor,
      low: low,
      high: high,
      shapeDynamism: shapeDynamism
    ))
  }
}

extension Tensor: CustomStringConvertible {
  public var description: String {
    self.anyTensor.description
  }
}
