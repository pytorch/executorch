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
}

struct ContentView: View {
  @State private var prompt = ""
  @State private var messages: [Message] = []
  @State private var showingLogs = false
  @State private var pickerType: PickerType?
  @State private var isGenerating = false
  @State private var shouldStopGenerating = false
  private let runnerQueue = DispatchQueue(label: "org.pytorch.executorch.llama")
  @StateObject private var runnerHolder = RunnerHolder()
  @StateObject private var resourceManager = ResourceManager()
  @StateObject private var resourceMonitor = ResourceMonitor()
  @StateObject private var logManager = LogManager()

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
        MessageListView(messages: $messages)
          .gesture(
            DragGesture().onChanged { value in
              if value.translation.height > 10 {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
              }
            }
          )
        HStack {
          Menu {
            Section(header: Text("Model")) {
              Button(action: { pickerType = .model }) {
                Label(modelTitle, systemImage: "doc")
              }
            }
            Section(header: Text("Tokenizer")) {
              Button(action: { pickerType = .tokenizer }) {
                Label(tokenizerTitle, systemImage: "doc")
              }
            }
          } label: {
            Image(systemName: "ellipsis.circle")
              .resizable()
              .aspectRatio(contentMode: .fit)
              .frame(height: 28)
          }
          .disabled(isGenerating)

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
      }
      .navigationBarTitle(title, displayMode: .inline)
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
  }

  private func generate() {
    guard !prompt.isEmpty else { return }
    isGenerating = true
    shouldStopGenerating = false
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let seq_len = 128
    prompt = ""
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath

    messages.append(Message(text: text))
    messages.append(Message(type: .generated))

    runnerQueue.async {
      defer {
        DispatchQueue.main.async {
          isGenerating = false
        }
      }
      runnerHolder.runner = runnerHolder.runner ?? Runner(modelPath: modelPath, tokenizerPath: tokenizerPath)
      guard !shouldStopGenerating else { return }
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
              message.text = "Model loaded in \(String(format: "%.1f", loadTime)) s"
            }
            messages.append(message)
            if error == nil {
              messages.append(Message(type: .generated))
            }
          }
        }
        if error != nil {
          return
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
        try runnerHolder.runner?.generate(text, sequenceLength: seq_len) { token in
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
      return [UTType(filenameExtension: "bin")].compactMap { $0 }
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
