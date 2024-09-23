/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI
import UniformTypeIdentifiers

import LLaMARunner

class RunnerHolder: ObservableObject {
  var runner: Runner?
  var llavaRunner: LLaVARunner?
}

extension UIImage {
  func resized(to newSize: CGSize) -> UIImage {
    let format = UIGraphicsImageRendererFormat.default()
    let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
    let image = renderer.image { _ in
      draw(in: CGRect(origin: .zero, size: newSize))
    }
    return image
  }

  func toRGBArray() -> [UInt8]? {
    guard let cgImage = self.cgImage else {
      NSLog("Failed to get CGImage from UIImage")
      return nil
    }

    let width = cgImage.width
    let height = cgImage.height
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let bitsPerComponent = 8
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      NSLog("Failed to create CGContext")
      return nil
    }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let pixelBuffer = context.data else {
      NSLog("Failed to get pixel data from CGContext")
      return nil
    }

    let pixelData = pixelBuffer.bindMemory(to: UInt8.self, capacity: width * height * bytesPerPixel)

    var rgbArray = [UInt8](repeating: 0, count: width * height * 3)

    for y in 0..<height {
      for x in 0..<width {
        let pixelIndex = (y * width + x) * bytesPerPixel
        let r = UInt8(pixelData[pixelIndex])
        let g = UInt8(pixelData[pixelIndex + 1])
        let b = UInt8(pixelData[pixelIndex + 2])

        let rgbIndex = (y * width + x)
        rgbArray[rgbIndex] = r
        rgbArray[rgbIndex + height * width] = g
        rgbArray[rgbIndex + 2 * height * width] = b
      }
    }

    return rgbArray
  }
}

struct ContentView: View {
  @State private var prompt = ""
  @State private var messages: [Message] = []
  @State private var showingLogs = false
  @State private var pickerType: PickerType?
  @State private var isGenerating = false
  @State private var shouldStopGenerating = false
  @State private var shouldStopShowingToken = false
  private let runnerQueue = DispatchQueue(label: "org.pytorch.executorch.llama")
  @StateObject private var runnerHolder = RunnerHolder()
  @StateObject private var resourceManager = ResourceManager()
  @StateObject private var resourceMonitor = ResourceMonitor()
  @StateObject private var logManager = LogManager()

  @State private var isImagePickerPresented = false
  @State private var selectedImage: UIImage?
  @State private var imagePickerSourceType: UIImagePickerController.SourceType = .photoLibrary

  @State private var showingSettings = false

  enum PickerType {
    case model
    case tokenizer
  }

  private var placeholder: String {
    resourceManager.isModelValid ? resourceManager.isTokenizerValid ? "Prompt..." : "Select Tokenizer..." : "Select Model..."
  }

  private var title: String {
    resourceManager.isModelValid ? resourceManager.isTokenizerValid ? resourceManager.modelName : "Select Tokenizer..." : "Select Model..."
  }

  private var modelTitle: String {
    resourceManager.isModelValid ? resourceManager.modelName : "Select Model..."
  }

  private var tokenizerTitle: String {
    resourceManager.isTokenizerValid ? resourceManager.tokenizerName : "Select Tokenizer..."
  }

  private var isInputEnabled: Bool { resourceManager.isModelValid && resourceManager.isTokenizerValid }

  var body: some View {
    NavigationView {
      VStack {
        if showingSettings {
          VStack(spacing: 20) {
            Form {
              Section(header: Text("Model and Tokenizer")
                        .font(.headline)
                        .foregroundColor(.primary)) {
                Button(action: { pickerType = .model }) {
                  Label(resourceManager.modelName == "" ? modelTitle : resourceManager.modelName, systemImage: "doc")
                }
                Button(action: { pickerType = .tokenizer }) {
                  Label(resourceManager.tokenizerName == "" ? tokenizerTitle : resourceManager.tokenizerName, systemImage: "doc")
                }
              }
            }
          }
        }

        MessageListView(messages: $messages)
          .gesture(
            DragGesture().onChanged { value in
              if value.translation.height > 10 {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
              }
            }
          )
        HStack {
          Button(action: {
            imagePickerSourceType = .photoLibrary
            isImagePickerPresented = true
          }) {
            Image(systemName: "photo.on.rectangle")
              .resizable()
              .scaledToFit()
              .frame(width: 24, height: 24)
          }
          .background(Color.clear)
          .cornerRadius(8)

          Button(action: {
            if UIImagePickerController.isSourceTypeAvailable(.camera) {
              imagePickerSourceType = .camera
              isImagePickerPresented = true
            } else {
              print("Camera not available")
            }
          }) {
            Image(systemName: "camera")
              .resizable()
              .scaledToFit()
              .frame(width: 24, height: 24)
          }
          .background(Color.clear)
          .cornerRadius(8)

          TextField(placeholder, text: $prompt, axis: .vertical)
            .padding(8)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(20)
            .lineLimit(1...10)
            .overlay(
              RoundedRectangle(cornerRadius: 20)
                .stroke(isInputEnabled ? Color.blue : Color.gray, lineWidth: 1)
            )
            .disabled(!isInputEnabled)

          Button(action: isGenerating ? stop : generate) {
            Image(systemName: isGenerating ? "stop.circle" : "arrowshape.up.circle.fill")
              .resizable()
              .aspectRatio(contentMode: .fit)
              .frame(height: 28)
          }
          .disabled(isGenerating ? shouldStopGenerating : (!isInputEnabled || prompt.isEmpty))
        }
        .padding([.leading, .trailing, .bottom], 10)
        .sheet(isPresented: $isImagePickerPresented, onDismiss: addSelectedImageMessage) {
          ImagePicker(selectedImage: $selectedImage, sourceType: imagePickerSourceType)
        }
      }
      .navigationBarTitle(title, displayMode: .inline)
      .navigationBarItems(leading:
                            Button(action: {
                              showingSettings.toggle()
                            }) {
                              Image(systemName: "gearshape")
                                .imageScale(.large)
                            })
      .navigationBarItems(trailing:
                            HStack {
                              Menu {
                                Section(header: Text("Memory")) {
                                  Text("Used: \(resourceMonitor.usedMemory) Mb")
                                  Text("Available: \(resourceMonitor.availableMemory) Mb")
                                }
                              } label: {
                                Text("\(resourceMonitor.usedMemory) Mb")
                              }
                              .onAppear {
                                resourceMonitor.start()
                              }
                              .onDisappear {
                                resourceMonitor.stop()
                              }
                              Button(action: { showingLogs = true }) {
                                Image(systemName: "list.bullet.rectangle")
                              }
                            }
      )
      .sheet(isPresented: $showingLogs) {
        NavigationView {
          LogView(logManager: logManager)
        }
      }
      .fileImporter(
        isPresented: Binding<Bool>(
          get: { pickerType != nil },
          set: { if !$0 { pickerType = nil } }
        ),
        allowedContentTypes: allowedContentTypes(),
        allowsMultipleSelection: false
      ) { [pickerType] result in
        handleFileImportResult(pickerType, result)
      }
      .onAppear {
        do {
          try resourceManager.createDirectoriesIfNeeded()
        } catch {
          withAnimation {
            messages.append(Message(type: .info, text: "Error creating content directories: \(error.localizedDescription)"))
          }
        }
      }
    }
    .navigationViewStyle(StackNavigationViewStyle())
  }

  private func addSelectedImageMessage() {
    if let selectedImage {
      messages.append(Message(image: selectedImage))
    }
  }

  private func generate() {
    guard !prompt.isEmpty else { return }
    isGenerating = true
    shouldStopGenerating = false
    shouldStopShowingToken = false
    let text = prompt
    let seq_len = 768 // text: 256, vision: 768
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath
    let useLlama = modelPath.range(of: "llama", options: .caseInsensitive) != nil

    prompt = ""
    hideKeyboard()
    showingSettings = false

    runnerQueue.async {
      defer {
        DispatchQueue.main.async {
          isGenerating = false
        }
      }

      if useLlama {
        runnerHolder.runner = runnerHolder.runner ?? Runner(modelPath: modelPath, tokenizerPath: tokenizerPath)
      } else {
        runnerHolder.llavaRunner = runnerHolder.llavaRunner ?? LLaVARunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
      }

      guard !shouldStopGenerating else { return }
      if useLlama {
        messages.append(Message(text: text))
        messages.append(Message(type: .llamagenerated))

        if let runner = runnerHolder.runner, !runner.isloaded() {
          var error: Error?
          let startLoadTime = Date()
          do {
            try runner.load()
          } catch let loadError {
            error = loadError
          }

          let loadTime = Date().timeIntervalSince(startLoadTime)
          DispatchQueue.main.async {
            withAnimation {
              var message = messages.removeLast()
              message.type = .info
              if let error {
                message.text = "Model loading failed: error \((error as NSError).code)"
              } else {
                message.text = "Model loaded in \(String(format: "%.2f", loadTime)) s"
              }
              messages.append(message)
              if error == nil {
                messages.append(Message(type: .llamagenerated))
              }
            }
          }
          if error != nil {
            return
          }
        }
      } else {
        messages.append(Message(text: text))
        messages.append(Message(type: .llavagenerated))

        if let runner = runnerHolder.llavaRunner, !runner.isloaded() {
          var error: Error?
          let startLoadTime = Date()
          do {
            try runner.load()
          } catch let loadError {
            error = loadError
          }

          let loadTime = Date().timeIntervalSince(startLoadTime)
          DispatchQueue.main.async {
            withAnimation {
              var message = messages.removeLast()
              message.type = .info
              if let error {
                message.text = "Model loading failed: error \((error as NSError).code)"
              } else {
                message.text = "Model loaded in \(String(format: "%.2f", loadTime)) s"
              }
              messages.append(message)
              if error == nil {
                messages.append(Message(type: .llavagenerated))
              }
            }
          }
          if error != nil {
            return
          }
        }
      }

      guard !shouldStopGenerating else {
        DispatchQueue.main.async {
          withAnimation {
            _ = messages.removeLast()
          }
        }
        return
      }
      do {
        var tokens: [String] = []
        var rgbArray: [UInt8]?
        let MAX_WIDTH = 336.0
        var newHeight = 0.0
        var imageBuffer: UnsafeMutableRawPointer?

        if let img = selectedImage {
          let llava_prompt = "\(text) ASSISTANT"

          newHeight = MAX_WIDTH * img.size.height / img.size.width
          let resizedImage = img.resized(to: CGSize(width: MAX_WIDTH, height: newHeight))
          rgbArray = resizedImage.toRGBArray()
          imageBuffer = UnsafeMutableRawPointer(mutating: rgbArray)

          try runnerHolder.llavaRunner?.generate(imageBuffer!, width: MAX_WIDTH, height: newHeight, prompt: llava_prompt, sequenceLength: seq_len) { token in

            if token != llava_prompt {
              if token == "</s>" {
                shouldStopGenerating = true
                runnerHolder.runner?.stop()
              } else {
                tokens.append(token)
                if tokens.count > 2 {
                  let text = tokens.joined()
                  let count = tokens.count
                  tokens = []
                  DispatchQueue.main.async {
                    withAnimation {
                      var message = messages.removeLast()
                      message.text += text
                      message.tokenCount += count
                      message.dateUpdated = Date()
                      messages.append(message)
                    }
                  }
                }
                if shouldStopGenerating {
                  runnerHolder.runner?.stop()
                }
              }
            }
          }
        } else {
          let llama3_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\(text)<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

          try runnerHolder.runner?.generate(llama3_prompt, sequenceLength: seq_len) { token in

            NSLog(">>> token={\(token)}")
            if token != llama3_prompt && !shouldStopShowingToken {
              // hack to fix the issue that extension/llm/runner/text_token_generator.h
              // keeps generating after <|eot_id|>
              if token == "<|eot_id|>" {
                shouldStopShowingToken = true
              } else {
                tokens.append(token.trimmingCharacters(in: .newlines))
                if tokens.count > 2 {
                  let text = tokens.joined()
                  let count = tokens.count
                  tokens = []
                  DispatchQueue.main.async {
                    withAnimation {
                      var message = messages.removeLast()
                      message.text += text
                      message.tokenCount += count
                      message.dateUpdated = Date()
                      messages.append(message)
                    }
                  }
                }
                if shouldStopGenerating {
                  runnerHolder.runner?.stop()
                }
              }
            }
          }
        }
      } catch {
        DispatchQueue.main.async {
          withAnimation {
            var message = messages.removeLast()
            message.type = .info
            message.text = "Text generation failed: error \((error as NSError).code)"
            messages.append(message)
          }
        }
      }
    }
  }

  private func stop() {
    shouldStopGenerating = true
  }

  private func allowedContentTypes() -> [UTType] {
    guard let pickerType else { return [] }
    switch pickerType {
    case .model:
      return [UTType(filenameExtension: "pte")].compactMap { $0 }
    case .tokenizer:
      return [UTType(filenameExtension: "bin"), UTType(filenameExtension: "model")].compactMap { $0 }
    }
  }

  private func handleFileImportResult(_ pickerType: PickerType?, _ result: Result<[URL], Error>) {
    switch result {
    case .success(let urls):
      guard let url = urls.first, let pickerType else {
        withAnimation {
          messages.append(Message(type: .info, text: "Failed to select a file"))
        }
        return
      }
      runnerQueue.async {
        runnerHolder.runner = nil
      }
      switch pickerType {
      case .model:
        resourceManager.modelPath = url.path
      case .tokenizer:
        resourceManager.tokenizerPath = url.path
      }
    case .failure(let error):
      withAnimation {
        messages.append(Message(type: .info, text: "Failed to select a file: \(error.localizedDescription)"))
      }
    }
  }
}

extension View {
  func hideKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
  }
}
