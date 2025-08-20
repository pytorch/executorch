/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Example script demonstrating multimodal runner usage with audio support

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/program.h>

using namespace executorch::extension::llm;

/**
 * Load audio data from a .pt tensor file
 * @param tensor_path Path to the .pt tensor file
 * @return Audio object with the loaded tensor data
 */
Audio load_audio_from_tensor(const std::string& tensor_path) {
  Audio audio;
  
  std::ifstream file(tensor_path, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open tensor file: " << tensor_path << std::endl;
    return audio;
  }

  // Read file size
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // For this example, we assume the .pt file contains raw float32 data
  // In practice, you would need to properly parse the PyTorch tensor format
  if (file_size % sizeof(float) != 0) {
    std::cerr << "Error: File size is not a multiple of float size" << std::endl;
    return audio;
  }

  size_t num_floats = file_size / sizeof(float);
  audio.data.resize(num_floats);
  
  file.read(reinterpret_cast<char*>(audio.data.data()), file_size);
  
  // Set default audio properties (these should be configured based on your data)
  audio.sample_rate = 16000;  // 16kHz
  audio.channels = 1;         // Mono
  audio.num_samples = num_floats;

  std::cout << "Loaded audio tensor: " << num_floats << " samples at " 
            << audio.sample_rate << "Hz" << std::endl;
  
  return audio;
}

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " <model_path> <tokenizer_path> [options]\n";
  std::cout << "\nOptions:\n";
  std::cout << "  --audio <path>     Path to .pt tensor file containing audio data\n";
  std::cout << "  --text <text>      Text input for the model\n";
  std::cout << "  --help             Show this help message\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << program_name << " model.pte tokenizer.json --text \"Transcribe this audio:\" --audio audio.pt\n";
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  std::string tokenizer_path = argv[2];
  std::string audio_path;
  std::string text_input = "Process this input:";

  // Parse command line arguments
  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    
    if (arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--audio" && i + 1 < argc) {
      audio_path = argv[++i];
    } else if (arg == "--text" && i + 1 < argc) {
      text_input = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
  auto tokenizer = load_tokenizer(tokenizer_path);
  if (!tokenizer) {
    std::cerr << "Error: Failed to load tokenizer" << std::endl;
    return 1;
  }

  std::cout << "Creating multimodal runner with model: " << model_path << std::endl;
  auto runner = create_multimodal_runner(model_path, std::move(tokenizer));
  if (!runner) {
    std::cerr << "Error: Failed to create multimodal runner" << std::endl;
    return 1;
  }

  std::cout << "Loading model..." << std::endl;
  auto load_result = runner->load();
  if (load_result != executorch::runtime::Error::Ok) {
    std::cerr << "Error: Failed to load model" << std::endl;
    return 1;
  }

  // Prepare multimodal inputs
  std::vector<MultimodalInput> inputs;
  
  // Add text input
  inputs.emplace_back(make_text_input(text_input));
  std::cout << "Added text input: " << text_input << std::endl;
  
  // Add audio input if provided
  if (!audio_path.empty()) {
    auto audio = load_audio_from_tensor(audio_path);
    if (audio.data.empty()) {
      std::cerr << "Error: Failed to load audio from " << audio_path << std::endl;
      return 1;
    }
    inputs.emplace_back(make_audio_input(std::move(audio)));
    std::cout << "Added audio input from: " << audio_path << std::endl;
  }

  // Configure generation parameters
  GenerationConfig config;
  config.max_new_tokens = 100;
  config.temperature = 0.7f;
  config.top_p = 0.9f;

  std::cout << "\nGenerating response..." << std::endl;
  
  // Set up callbacks
  std::string generated_text;
  auto token_callback = [&](const std::string& token) {
    std::cout << token << std::flush;
    generated_text += token;
  };
  
  auto stats_callback = [](const Stats& stats) {
    // Optional: handle generation statistics
  };

  // Generate response
  auto generate_result = runner->generate(inputs, config, token_callback, stats_callback);
  
  std::cout << std::endl;
  
  if (generate_result != executorch::runtime::Error::Ok) {
    std::cerr << "Error: Generation failed" << std::endl;
    return 1;
  }

  std::cout << "\nGeneration completed successfully!" << std::endl;
  std::cout << "Generated text: " << generated_text << std::endl;

  return 0;
}