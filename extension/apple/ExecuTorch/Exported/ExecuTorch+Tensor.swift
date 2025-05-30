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
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt8: Scalar { public static var dataType: DataType { .byte } }
@available(*, deprecated, message: "This API is experimental.")
extension Int8: Scalar { public static var dataType: DataType { .char } }
@available(*, deprecated, message: "This API is experimental.")
extension Int16: Scalar { public static var dataType: DataType { .short } }
@available(*, deprecated, message: "This API is experimental.")
extension Int32: Scalar { public static var dataType: DataType { .int } }
@available(*, deprecated, message: "This API is experimental.")
extension Int64: Scalar { public static var dataType: DataType { .long } }
@available(*, deprecated, message: "This API is experimental.")
extension Int: Scalar { public static var dataType: DataType { .long } }
@available(*, deprecated, message: "This API is experimental.")
extension Float: Scalar { public static var dataType: DataType { .float } }
@available(*, deprecated, message: "This API is experimental.")
extension Double: Scalar { public static var dataType: DataType { .double } }
@available(*, deprecated, message: "This API is experimental.")
extension Bool: Scalar { public static var dataType: DataType { .bool } }
@available(*, deprecated, message: "This API is experimental.")
extension UInt16: Scalar { public static var dataType: DataType { .uInt16 } }
@available(*, deprecated, message: "This API is experimental.")
extension UInt32: Scalar { public static var dataType: DataType { .uInt32 } }
@available(*, deprecated, message: "This API is experimental.")
extension UInt64: Scalar { public static var dataType: DataType { .uInt64 } }
@available(*, deprecated, message: "This API is experimental.")
extension UInt: Scalar { public static var dataType: DataType { .uInt64 } }

@available(*, deprecated, message: "This API is experimental.")
public extension Tensor {
  /// Calls the closure with a typed, immutable buffer pointer over the tensor’s elements.
  ///
  /// - Parameter body: A closure that receives an `UnsafeBufferPointer<T>` bound to the tensor’s data.
  /// - Returns: The value returned by `body`.
  /// - Throws: `Error(code: .invalidArgument)` if `T.dataType` doesn’t match the tensor’s `dataType`,
  ///           or any error thrown by `body`.
  func withUnsafeBytes<T: Scalar, R>(_ body: (UnsafeBufferPointer<T>) throws -> R) throws -> R {
    guard dataType == T.dataType else { throw Error(code: .invalidArgument) }
    var result: Result<R, Error>?
    __bytes { pointer, count, _ in
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
  /// - Throws: `Error(code: .invalidArgument)` if `T.dataType` doesn’t match the tensor’s `dataType`,
  ///           or any error thrown by `body`.
  func withUnsafeMutableBytes<T: Scalar, R>(_ body: (UnsafeMutableBufferPointer<T>) throws -> R) throws -> R {
    guard dataType == T.dataType else { throw Error(code: .invalidArgument) }
    var result: Result<R, Error>?
    __mutableBytes { pointer, count, _ in
      result = Result { try body(
        UnsafeMutableBufferPointer(
          start: pointer.assumingMemoryBound(to: T.self),
          count: count
        )
      ) }
    }
    return try result!.get()
  }
}
