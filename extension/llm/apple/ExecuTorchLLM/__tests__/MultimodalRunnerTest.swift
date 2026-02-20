/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
import ExecuTorchLLM
import XCTest

extension UIImage {
  func centerCropped(to sideSize: CGFloat) -> UIImage {
    precondition(sideSize > 0)
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    format.opaque = false
    return UIGraphicsImageRenderer(size: CGSize(width: sideSize, height: sideSize), format: format)
      .image { _ in
        let scaleFactor = max(sideSize / size.width, sideSize / size.height)
        let scaledWidth = size.width * scaleFactor
        let scaledHeight = size.height * scaleFactor
        let originX = (sideSize - scaledWidth) / 2
        let originY = (sideSize - scaledHeight) / 2
        draw(in: CGRect(x: originX, y: originY, width: scaledWidth, height: scaledHeight))
      }
  }

  func rgbBytes() -> [UInt8]? {
    guard let cgImage = cgImage else { return nil }
    let pixelWidth = Int(cgImage.width)
    let pixelHeight = Int(cgImage.height)
    let pixelCount = pixelWidth * pixelHeight
    let bytesPerPixel = 4
    let bytesPerRow = pixelWidth * bytesPerPixel
    var rgbaBuffer = [UInt8](repeating: 0, count: pixelCount * bytesPerPixel)
    guard let context = CGContext(
      data: &rgbaBuffer,
      width: pixelWidth,
      height: pixelHeight,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else { return nil }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))

    var rgbBytes = [UInt8](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      let sourceIndex = pixelIndex * bytesPerPixel
      rgbBytes[pixelIndex] = rgbaBuffer[sourceIndex + 0]
      rgbBytes[pixelIndex + pixelCount] = rgbaBuffer[sourceIndex + 1]
      rgbBytes[pixelIndex + 2 * pixelCount] = rgbaBuffer[sourceIndex + 2]
    }
    return rgbBytes
  }

  func rgbBytesNormalized(mean: [Float] = [0, 0, 0], std: [Float] = [1, 1, 1]) -> [Float]? {
    precondition(mean.count == 3 && std.count == 3)
    precondition(std[0] != 0 && std[1] != 0 && std[2] != 0)
    guard let rgbBytes = rgbBytes() else { return nil }
    let pixelCount = rgbBytes.count / 3
    var rgbBytesNormalized = [Float](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      rgbBytesNormalized[pixelIndex] =
        (Float(rgbBytes[pixelIndex]) / 255.0 - mean[0]) / std[0]
      rgbBytesNormalized[pixelIndex + pixelCount] =
        (Float(rgbBytes[pixelIndex + pixelCount]) / 255.0 - mean[1]) / std[1]
      rgbBytesNormalized[pixelIndex + 2 * pixelCount] =
        (Float(rgbBytes[pixelIndex + 2 * pixelCount]) / 255.0 - mean[2]) / std[2]
    }
    return rgbBytesNormalized
  }

  func asImage(_ sideSize: CGFloat) -> Image {
    return Image(
      data: Data(centerCropped(to: sideSize).rgbBytes() ?? []),
      width: Int(sideSize),
      height: Int(sideSize),
      channels: 3
    )
  }

  func asNormalizedImage(
    _ sideSize: CGFloat,
    mean: [Float] = [0.485, 0.456, 0.406],
    std: [Float] = [0.229, 0.224, 0.225]
  ) -> Image {
    return Image(
      float: (centerCropped(to: sideSize).rgbBytesNormalized(mean: mean, std: std) ?? []).withUnsafeBufferPointer { Data(buffer: $0) },
      width: Int(sideSize),
      height: Int(sideSize),
      channels: 3
    )
  }
}

class MultimodalRunnerTest: XCTestCase {
  let systemPrompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

  func testGemma() {
    let chatTemplate = "<start_of_turn>user\n%@<end_of_turn>\n<start_of_turn>model"
    let userPrompt = "What's on the picture?"
    let sideSize: CGFloat = 896
    let sequenceLength = 768
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "gemma3", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "gemma3_tokenizer", ofType: "model"),
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
        MultimodalInput(uiImage.asNormalizedImage(sideSize)),
        MultimodalInput(String(format: chatTemplate, userPrompt)),
      ], Config {
        $0.sequenceLength = sequenceLength
        $0.maximumNewTokens = 128
        $0.temperature = 0
      }) { token in
        text += token
        if token == "<end_of_turn>" {
          runner.stop()
        }
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
        MultimodalInput(uiImage.asNormalizedImage(sideSize)),
        MultimodalInput(String(format: chatTemplate, userPrompt)),
      ], Config {
        $0.sequenceLength = sequenceLength
        $0.maximumNewTokens = 128
        $0.temperature = 0
      }) { token in
        text += token
        if token == "<end_of_turn>" {
          runner.stop()
        }
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertTrue(text.lowercased().contains("waterfall"))
  }

  func testLLaVA() {
    let chatTemplate = "USER: %@ ASSISTANT: "
    let userPrompt = "What's on the picture?"
    let sideSize: CGFloat = 336
    let sequenceLength = 768
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "llava", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "llava_tokenizer", ofType: "bin"),
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
        MultimodalInput(uiImage.asImage(sideSize)),
        MultimodalInput(String(format: chatTemplate, userPrompt)),
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
        MultimodalInput(uiImage.asImage(sideSize)),
        MultimodalInput(String(format: chatTemplate, userPrompt)),
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

  func testVoxtral() throws {
    let chatTemplate = "%@[/INST]"
    let userPrompt = "What is the audio about?"
    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(forResource: "voxtral", ofType: "pte"),
          let tokenizerPath = bundle.path(forResource: "voxtral_tokenizer_tekken", ofType: "json"),
          let audioPath = bundle.path(forResource: "voxtral_input_features", ofType: "bin") else {
      XCTFail("Couldn't find model or tokenizer files")
      return
    }
    let runner = MultimodalRunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
    var audioData = try Data(contentsOf: URL(fileURLWithPath: audioPath), options: .mappedIfSafe)
    let floatSize = MemoryLayout<Float>.size
    guard audioData.count % floatSize == 0 else {
      XCTFail("Invalid audio data")
      return
    }
    let bins = 128
    let frames = 3000
    let batchSize = audioData.count / floatSize / (bins * frames)
    var text = ""

    do {
      try runner.generate([
        MultimodalInput("<s>[INST][BEGIN_AUDIO]"),
        MultimodalInput(Audio(
          float: audioData,
          batchSize: batchSize,
          bins: bins,
          frames: frames
        )),
        MultimodalInput(String(format: chatTemplate, userPrompt)),
      ], Config {
        $0.maximumNewTokens = 128
        $0.temperature = 0
      }) { token in
        text += token
      }
    } catch {
      XCTFail("Failed to generate text with error \(error)")
    }
    XCTAssertTrue(text.lowercased().contains("tattoo"))
  }
}
