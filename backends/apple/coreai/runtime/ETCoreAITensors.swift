// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import CoreAI
import Foundation

@available(macOS 27.0, iOS 27.0, *)
struct CoreAIScalarInfo: Sendable {
  let sdkType: NDArray.ScalarType
  let bridgeType: ETCoreAIScalarType
  let byteWidth: Int

  // Only byte-addressable scalar types belong here; packed types need a separate layout policy.
  static let supported: [Self] = [
    Self(sdkType: .float16, bridgeType: .float16, byteWidth: 2),
    Self(sdkType: .float32, bridgeType: .float32, byteWidth: 4),
  ]

  static func from(sdkType: NDArray.ScalarType) throws -> Self {
    guard let info = supported.first(where: { $0.sdkType == sdkType }) else {
      throw bridgeError(.unsupported, "Unsupported Core AI scalar type: \(sdkType)")
    }
    return info
  }

  static func from(bridgeType: ETCoreAIScalarType) throws -> Self {
    guard let info = supported.first(where: { $0.bridgeType == bridgeType }) else {
      throw bridgeError(.unsupported, "Unsupported bridge scalar type: \(bridgeType.rawValue)")
    }
    return info
  }

  func byteCount(shape: [Int], errorCode: ETCoreAIErrorCode = .invalidArgument) throws -> Int {
    guard shape.allSatisfy({ $0 >= 0 }) else {
      throw bridgeError(errorCode, "Negative tensor dimension")
    }
    // Check suffix products too: SDK dense-stride construction must not overflow on empty tensors.
    var count = byteWidth
    for dimension in shape.reversed() {
      let (next, overflow) = count.multipliedReportingOverflow(by: max(dimension, 1))
      guard !overflow else {
        throw bridgeError(errorCode, "Tensor byte count overflows Int")
      }
      count = next
    }
    return shape.contains(0) ? 0 : count
  }
}

// ExecuTorch waits synchronously and keeps input storage live through output copying.
// This snapshot transfers only metadata and a pointer to the task; it does not own storage.
@available(macOS 27.0, iOS 27.0, *)
struct CoreAITensorPayload: @unchecked Sendable {
  let bytes: UnsafeRawPointer?
  let byteCount: Int
  let shape: [Int]
  let scalarType: ETCoreAIScalarType

  init(_ tensor: ETCoreAIInputTensor) throws {
    shape = try tensor.shape.map { number in
      guard let dimension = number as? Int, dimension >= 0 else {
        throw bridgeError(.invalidArgument, "Tensor dimensions must be nonnegative integers")
      }
      return dimension
    }
    scalarType = tensor.scalarType
    byteCount = try Self.byteCount(shape: shape, scalarType: scalarType)
    guard tensor.byteCount == byteCount else {
      throw bridgeError(
        .invalidArgument, "Tensor payload byte count does not match shape and dtype")
    }
    bytes = tensor.bytes
    guard byteCount == 0 || bytes != nil else {
      throw bridgeError(.invalidArgument, "Nonempty tensor has null data")
    }
  }

  static func byteCount(
    shape: [Int], scalarType: ETCoreAIScalarType,
    errorCode: ETCoreAIErrorCode = .invalidArgument
  ) throws -> Int {
    try CoreAIScalarInfo.from(bridgeType: scalarType).byteCount(shape: shape, errorCode: errorCode)
  }
}

@available(macOS 27.0, iOS 27.0, *)
func tensorDescriptor(_ value: InferenceValue.Descriptor?) throws -> NDArrayDescriptor {
  guard let value else { throw bridgeError(.invalidModel, "Missing tensor descriptor") }
  guard case .ndArray(let descriptor) = value else {
    throw bridgeError(.unsupported, "Image values are not supported")
  }
  _ = try CoreAIScalarInfo.from(sdkType: descriptor.scalarType)
  guard descriptor.interleaveLayout == nil else {
    throw bridgeError(.unsupported, "Interleaved tensor descriptors are not supported")
  }
  guard descriptor.shape.allSatisfy({ $0 >= -1 }) else {
    throw bridgeError(.invalidModel, "Invalid descriptor dimension")
  }
  return descriptor
}

@available(macOS 27.0, iOS 27.0, *)
func validateShape(
  _ shape: [Int], descriptor: NDArrayDescriptor,
  errorCode: ETCoreAIErrorCode = .invalidArgument
) throws {
  guard shape.count == descriptor.shape.count,
    zip(shape, descriptor.shape).allSatisfy({ actual, expected in
      actual >= 0 && (expected == -1 || actual == expected)
    })
  else {
    throw bridgeError(errorCode, "Tensor shape does not match the function descriptor")
  }
}

// Offsets are in elements, including padded or transposed SDK layouts.
struct CoreAITensorLayout {
  let shape: [Int]
  let strides: [Int]
  let count: Int
  // Row-major contiguous, ignoring strides of size-1 dimensions.
  let isDense: Bool

  init(shape: [Int], strides: [Int], width: Int, storageByteCount: Int) throws {
    guard shape.count == strides.count, shape.allSatisfy({ $0 >= 0 }),
      strides.allSatisfy({ $0 >= 0 }), width > 0, storageByteCount >= 0
    else { throw bridgeError(.unsupported, "Invalid or negative NDArray strides") }
    self.shape = shape
    self.strides = strides
    var count = 1
    var maximumOffset = 0
    for (dimension, stride) in zip(shape, strides) {
      let (next, countOverflow) = count.multipliedReportingOverflow(by: dimension)
      let (extent, extentOverflow) = max(dimension - 1, 0).multipliedReportingOverflow(by: stride)
      let (offset, offsetOverflow) = maximumOffset.addingReportingOverflow(extent)
      guard !countOverflow, !extentOverflow, !offsetOverflow else {
        throw bridgeError(.runtime, "NDArray layout overflows Int")
      }
      count = next
      maximumOffset = offset
    }
    self.count = count
    var expectedStride = 1
    var isDense = true
    for axis in shape.indices.reversed() where shape[axis] != 1 {
      isDense = isDense && strides[axis] == expectedStride
      expectedStride = expectedStride &* shape[axis]
    }
    self.isDense = isDense
    if count > 0 {
      let (span, spanOverflow) = maximumOffset.addingReportingOverflow(1)
      let (bytes, byteOverflow) = span.multipliedReportingOverflow(by: width)
      guard !spanOverflow, !byteOverflow, bytes <= storageByteCount else {
        throw bridgeError(.runtime, "NDArray layout exceeds its storage")
      }
    }
  }

  func offset(for linearIndex: Int) -> Int {
    var remainder = linearIndex
    var offset = 0
    for axis in shape.indices.reversed() {
      offset += (remainder % shape[axis]) * strides[axis]
      remainder /= shape[axis]
    }
    return offset
  }
}

@available(macOS 27.0, iOS 27.0, *)
func prepareOutput(_ array: NDArray) throws -> ETCoreAITensor {
  let type = try CoreAIScalarInfo.from(sdkType: array.scalarType)
  let byteCount = try type.byteCount(shape: array.shape, errorCode: .runtime)
  guard array.interleaveLayout == nil else {
    throw bridgeError(.unsupported, "Interleaved output storage is not supported")
  }
  let view = array.rawView()
  let layout = try CoreAITensorLayout(
    shape: array.shape, strides: array.strides, width: type.byteWidth,
    storageByteCount: view.bytes.byteCount)
  let shape = view.shape
  let strides = view.strides
  guard view.scalarType == type.sdkType, view.interleaveLayout == nil,
    shape.count == layout.shape.count, strides.count == layout.strides.count
  else {
    throw bridgeError(.runtime, "Output view layout changed")
  }
  for axis in layout.shape.indices {
    guard shape[axis] == layout.shape[axis], strides[axis] == layout.strides[axis] else {
      throw bridgeError(.runtime, "Output view layout changed")
    }
  }
  // Retain the immutable array, not its scoped pointer. All fallible work is complete.
  return ETCoreAITensor(
    byteCount: UInt(byteCount), shape: layout.shape.map(NSNumber.init(value:)),
    scalarType: type.bridgeType
  ) { destination in
    let view = array.rawView()
    view.bytes.withUnsafeBytes { source in
      if layout.isDense {
        destination.copyMemory(from: source.baseAddress!, byteCount: byteCount)
        return
      }
      for index in 0..<layout.count {
        destination.advanced(by: index * type.byteWidth).copyMemory(
          from: source.baseAddress!.advanced(by: layout.offset(for: index) * type.byteWidth),
          byteCount: type.byteWidth)
      }
    }
  }
}
