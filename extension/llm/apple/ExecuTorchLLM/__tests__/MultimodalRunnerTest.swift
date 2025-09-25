/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorchLLM
import XCTest

extension UIImage {
  func asImage() -> Image {
    let targetWidth = 336
    let scaledHeight = Int((Double(targetWidth) * Double(size.height) / Double(size.width)).rounded())
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    let resizedImage = UIGraphicsImageRenderer(size: CGSize(width: targetWidth, height: scaledHeight), format: format).image { _ in
      draw(in: CGRect(origin: .zero, size: CGSize(width: targetWidth, height: scaledHeight)))
    }
    let resizedCGImage = resizedImage.cgImage!
    let imageWidth = resizedCGImage.width
    let imageHeight = resizedCGImage.height
    let pixelCount = imageWidth * imageHeight
    var rgbaBuffer = [UInt8](repeating: 0, count: pixelCount * 4)
    let context = CGContext(
      data: &rgbaBuffer,
      width: imageWidth,
      height: imageHeight,
      bitsPerComponent: 8,
      bytesPerRow: imageWidth * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    )!
    context.draw(resizedCGImage, in: CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
    var planarRGB = [UInt8](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      let sourceOffset = pixelIndex * 4
      planarRGB[pixelIndex] = rgbaBuffer[sourceOffset]
      planarRGB[pixelIndex + pixelCount] = rgbaBuffer[sourceOffset + 1]
      planarRGB[pixelIndex + pixelCount * 2] = rgbaBuffer[sourceOffset + 2]
    }
    return Image(data: Data(planarRGB), width: targetWidth, height: scaledHeight, channels: 3)
  }
}

class MultimodalRunnerTest: XCTestCase {
  let systemPrompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
  let assistantPrompt = "ASSISTANT: "
  let userPrompt = "What's on the picture?"
  let sequenceLength = 768

  func test() {
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "llava", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "tokenizer", ofType: "bin"),
          let imagePath = bundle.path(forResource: "IMG_0005", ofType: "jpg"),
          let image = UIImage(contentsOfFile: imagePath) else {
      XCTFail("Couldn't find model or tokenizer files")
      return
    }
    let runner = MultimodalRunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
    var text = ""

    do {
      try runner.generate([
        MultimodalInput(systemPrompt),
        MultimodalInput(image.asImage()),
        MultimodalInput("\(userPrompt) \(assistantPrompt)"),
      ], sequenceLength: sequenceLength) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertTrue(text.lowercased().contains("waterfall"))

    text = ""
    runner.reset()
    do {
      try runner.generate([
        MultimodalInput(systemPrompt),
        MultimodalInput(image.asImage()),
        MultimodalInput("\(userPrompt) \(assistantPrompt)"),
      ], sequenceLength: sequenceLength) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertTrue(text.lowercased().contains("waterfall"))
  }
}
