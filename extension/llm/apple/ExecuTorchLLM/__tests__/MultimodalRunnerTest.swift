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
    let targetSide = CGFloat(336)
    let scale = max(targetSide / size.width, targetSide / size.height)
    let scaledSize = CGSize(width: size.width * scale, height: size.height * scale)
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    let scaledImage = UIGraphicsImageRenderer(size: scaledSize, format: format).image { _ in
      draw(in: CGRect(origin: .zero, size: scaledSize))
    }
    guard let scaledCGImage = scaledImage.cgImage else {
      return Image(data: Data(), width: 336, height: 336, channels: 3)
    }
    let cropRect = CGRect(
      x: ((scaledSize.width - targetSide) * 0.5).rounded(.down),
      y: ((scaledSize.height - targetSide) * 0.5).rounded(.down),
      width: targetSide.rounded(.down),
      height: targetSide.rounded(.down)
    )
    let croppedCGImage = scaledCGImage.cropping(to: cropRect) ?? scaledCGImage
    let imageWidth = croppedCGImage.width
    let imageHeight = croppedCGImage.height
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
    context.draw(croppedCGImage, in: CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
    var planarRGB = [UInt8](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      let sourceOffset = pixelIndex * 4
      planarRGB[pixelIndex] = rgbaBuffer[sourceOffset]
      planarRGB[pixelIndex + pixelCount] = rgbaBuffer[sourceOffset + 1]
      planarRGB[pixelIndex + pixelCount * 2] = rgbaBuffer[sourceOffset + 2]
    }
    return Image(data: Data(planarRGB), width: 336, height: 336, channels: 3)
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
          let uiImage = UIImage(contentsOfFile: imagePath) else {
      XCTFail("Couldn't find model or tokenizer files")
      return
    }
    let runner = MultimodalRunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
    var text = ""

    do {
      try runner.generate([
        MultimodalInput(systemPrompt),
        MultimodalInput(uiImage.asImage()),
        MultimodalInput("\(userPrompt) \(assistantPrompt)"),
      ], Config {
        $0.sequenceLength = sequenceLength
      }) { token in
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
        MultimodalInput(uiImage.asImage()),
        MultimodalInput("\(userPrompt) \(assistantPrompt)"),
      ], Config {
        $0.sequenceLength = sequenceLength
      }) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertTrue(text.lowercased().contains("waterfall"))
  }
}
