/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import CoreVideo
import ExecuTorch
import XCTest

// These tests cover the ObjC/Swift binding layer only: config field forwarding,
// the CVPixelBuffer entry point, the reuse (process-into) path, the
// letterbox-padding bridge, and the nil guard. Image-processing correctness
// (color conversion, resize/letterbox math, normalization, CPU/GPU
// equivalence, format support) is owned by the C++ suite
// (extension/image/test/image_processor_test.cpp and
// image_processor_apple_test.cpp) and is intentionally not re-tested here.
class ImageProcessorTest: XCTestCase {

  // MARK: - Helper: Create BGRA CVPixelBuffer

  private func makeBGRAPixelBuffer(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> CVPixelBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32BGRA,
      nil,
      &pixelBuffer
    )
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    if let base = CVPixelBufferGetBaseAddress(buffer) {
      let stride = CVPixelBufferGetBytesPerRow(buffer)
      let ptr = base.assumingMemoryBound(to: UInt8.self)
      for row in 0..<height {
        for col in 0..<width {
          let offset = row * stride + col * 4
          ptr[offset] = b       // B
          ptr[offset + 1] = g   // G
          ptr[offset + 2] = r   // R
          ptr[offset + 3] = 255 // A
        }
      }
    }

    return buffer
  }

  // MARK: - Reuse path (process into caller-provided tensor)

  func testProcessIntoMatchesAllocatingPath() throws {
    // Exercises the binding-only glue in process(_:into:): the Swift wrapper
    // unwraps `tensor.anyTensor` and the .mm reinterpret_casts its native
    // instance. The C++ process_pixelbuffer_into is covered separately; here
    // we just verify the reuse path fills a caller tensor with the same result
    // as the allocating process(_:). This also smoke-tests process(_:) and the
    // CVPixelBuffer -> C++ -> Tensor<Float> bridge end to end.
    let config = ImageProcessorConfig(targetWidth: 4, targetHeight: 4)
    let processor = ImageProcessor(config: config)

    guard let pixelBuffer = makeBGRAPixelBuffer(width: 8, height: 6, r: 200, g: 100, b: 50) else {
      XCTFail("Failed to create BGRA pixel buffer")
      return
    }

    let output = Tensor<Float>.zeros(shape: [1, 3, 4, 4])
    try processor.process(pixelBuffer, into: output)

    let expected: Tensor<Float> = try processor.process(pixelBuffer)
    XCTAssertEqual(output.shape, [1, 3, 4, 4])
    let outData = output.scalars()
    let expData = expected.scalars()
    XCTAssertEqual(outData.count, expData.count)
    for i in 0..<outData.count {
      XCTAssertEqual(outData[i], expData[i], accuracy: 1e-5, "into-path mismatch at \(i)")
    }
  }

  // MARK: - computeLetterboxPadding

  func testComputeLetterboxPadding() throws {
    // Exercises the binding-only glue: the .mm packs the C++ result into a
    // CGPoint and the Swift wrapper unpacks it to (x: Int, y: Int). The C++
    // compute_letterbox_padding is covered separately.
    let config = ImageProcessorConfig(
      targetWidth: 8,
      targetHeight: 8,
      resizeMode: .letterbox,
      letterboxAnchor: .center,
      padValue: 0.0,
      normalization: .zeroToOne(),
      gpuMinInputPixels: ImageProcessorConfig.alwaysCPU
    )
    let processor = ImageProcessor(config: config)

    // 8x4 source into an 8x8 target: width fits exactly, height is padded.
    // Resized content is 8x4, leaving 2px of pad on top and bottom.
    let padding = processor.computeLetterboxPadding(inputWidth: 8, inputHeight: 4)
    XCTAssertEqual(padding.x, 0)
    XCTAssertEqual(padding.y, 2)
  }

  // MARK: - Error handling tests

  func testProcessNilPixelBufferReturnsError() {
    let config = ImageProcessorConfig(targetWidth: 4, targetHeight: 4)
    let processor = ImageProcessor(config: config)

    // The nil guard lives in the ObjC wrapper (the C++ layer never sees nil),
    // so this path is binding-specific.
    XCTAssertThrowsError(try processor.processPixelBuffer(nil)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, ErrorDomain)
      XCTAssertEqual(nsError.code, ErrorCode.invalidArgument.rawValue)
    }
  }

  func testProcessIntoWrongShapeReturnsError() throws {
    // The into: path validates the caller-provided tensor in C++
    // (check_out_tensor) before any write. A wrong-shape Tensor<Float> must
    // surface .invalidArgument through the binding; this is the binding-specific
    // behavior the into: path exists for.
    let config = ImageProcessorConfig(targetWidth: 4, targetHeight: 4)
    let processor = ImageProcessor(config: config)

    guard let pixelBuffer = makeBGRAPixelBuffer(width: 8, height: 6, r: 200, g: 100, b: 50) else {
      XCTFail("Failed to create BGRA pixel buffer")
      return
    }

    // Config expects [1, 3, 4, 4]; pass a mismatched output tensor.
    let wrongShape = Tensor<Float>.zeros(shape: [1, 3, 8, 8])
    XCTAssertThrowsError(try processor.process(pixelBuffer, into: wrongShape)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, ErrorDomain)
      XCTAssertEqual(nsError.code, ErrorCode.invalidArgument.rawValue)
    }
  }

  // MARK: - Config round-trip tests

  func testConfigPropertyRoundTrip() throws {
    // Construct config with non-default values and verify they round-trip
    // through the processor. This catches dropped/misforwarded fields in
    // initWithConfig and nativeConfig.
    let config = ImageProcessorConfig(
      targetWidth: 224,
      targetHeight: 224,
      resizeMode: .letterbox,
      letterboxAnchor: .topLeft,
      padValue: 0.5,
      normalization: .imagenet(),
      gpuMinInputPixels: ImageProcessorConfig.alwaysCPU
    )
    let processor = ImageProcessor(config: config)

    // Verify all fields round-trip correctly
    XCTAssertEqual(processor.config.targetWidth, 224)
    XCTAssertEqual(processor.config.targetHeight, 224)
    XCTAssertEqual(processor.config.resizeMode, .letterbox)
    XCTAssertEqual(processor.config.letterboxAnchor, .topLeft)
    XCTAssertEqual(processor.config.padValue, 0.5, accuracy: 1e-6)
    XCTAssertEqual(processor.config.gpuMinInputPixels, ImageProcessorConfig.alwaysCPU)
    // Normalization is a reference type, so we check it's the same instance
    XCTAssertTrue(processor.config.normalization === config.normalization)
  }

  func testDefaultInitializerUsesDefaultThreshold() throws {
    // The convenience init inherits the C++ config's default gpuMinInputPixels.
    let config = ImageProcessorConfig(targetWidth: 4, targetHeight: 4)
    let processor = ImageProcessor(config: config)

    XCTAssertEqual(
      processor.config.gpuMinInputPixels,
      ImageProcessorConfig.defaultGpuMinInputPixels)
  }

  // MARK: - Custom normalization

  func testCustomNormalizationApplied() throws {
    // Verifies a custom ImageNormalization (scale/mean/std) actually flows
    // through the binding into the C++ pipeline. zeroToOne yields pixel/255;
    // with the same scale but mean 0.5 / std 0.5 the result is
    // (pixel/255 - 0.5) / 0.5 == 2 * zeroToOne - 1, channel-wise.
    guard let pixelBuffer = makeBGRAPixelBuffer(width: 8, height: 6, r: 200, g: 100, b: 50) else {
      XCTFail("Failed to create BGRA pixel buffer")
      return
    }

    let baseConfig = ImageProcessorConfig(targetWidth: 4, targetHeight: 4)
    let baseOutput = try ImageProcessor(config: baseConfig).process(pixelBuffer)

    let custom = ImageNormalization(
      scaleFactor: 1.0 / 255.0,
      mean: [0.5, 0.5, 0.5],
      standardDeviation: [0.5, 0.5, 0.5])
    let customConfig = ImageProcessorConfig(
      targetWidth: 4,
      targetHeight: 4,
      normalization: custom)
    let customOutput = try ImageProcessor(config: customConfig).process(pixelBuffer)

    let base = baseOutput.scalars()
    let got = customOutput.scalars()
    XCTAssertEqual(base.count, got.count)
    for i in 0..<got.count {
      XCTAssertEqual(got[i], 2.0 * base[i] - 1.0, accuracy: 1e-5, "custom-normalization mismatch at \(i)")
    }
  }
}
