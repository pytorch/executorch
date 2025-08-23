/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic encoder prefiller that handles multimodal inputs (text, image and
// audio (to be implemented)) to prefill the KV cache of a multimodal LLM.
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>

#include <iostream>

namespace executorch::extension::llm {

MultimodalPrefiller::MultimodalPrefiller(
    Module* module,
    MultimodalDecoderRunner* decoder_runner,
    Tokenizer* tokenizer,
    IOManager* io_manager)
    : module_(module),
      text_decoder_runner_(decoder_runner),
      tokenizer_(tokenizer),
      io_manager_(io_manager) {}

/**
 * Prefill an LLM Module with the given multimodal input.
 * @param input The multimodal input (text, image or audio) to the multimodal
 * LLM.
 * @param start_pos The starting position in KV cache of the input in the LLM
 * @return logits of the prefill.
 */
Result<uint64_t> MultimodalPrefiller::prefill(
    const MultimodalInput& input,
    int64_t& start_pos) {
  std::cout << std::endl;
  // Check if input is image
  ::executorch::runtime::EValue encoder_output;
  if (input.is_image()) {
    Image image = input.get_image();
    auto image_tensor = executorch::extension::from_blob(
        image.data.data(),
        {3, image.height, image.width},
        ::executorch::aten::ScalarType::Byte);

    // Run image encoder
    auto image_encoder_outputs =
        ET_UNWRAP(module_->execute(kImageEncoderMethod, image_tensor));

    encoder_output = image_encoder_outputs[0];
  } else if (input.is_audio()) {
    Audio audio = input.get_audio();
    
    // Use the original tensor shape as intended
    std::cout << "Creating tensor with original shape [" << audio.batch_size << ", " << audio.n_bins << ", " << audio.n_frames << "]" << std::endl;
    auto audio_tensor = executorch::extension::from_blob(
        const_cast<float*>(reinterpret_cast<const float*>(audio.data.data())),
        {audio.batch_size, audio.n_bins, audio.n_frames},
        ::executorch::aten::ScalarType::Float);

    // Print first few values from audio_tensor input to validate
    if (audio_tensor->numel() > 0) {
      const float* data = audio_tensor->const_data_ptr<float>();
      std::cout << "First 10 audio_tensor values: ";
      for (int i = 0; i < 10 && i < audio_tensor->numel(); ++i) {
        std::cout << data[i] << " ";
      }
      std::cout << std::endl;
    }
    
    // Run audio encoder
    auto audio_encoder_result = module_->execute(kAudioEncoderMethod, audio_tensor);
    if (audio_encoder_result.error() != ::executorch::runtime::Error::Ok) {
        std::cout << "Audio encoder execution failed with error: " << static_cast<int>(audio_encoder_result.error()) << std::endl;
        return ::executorch::runtime::Error::Internal;
    }
    auto audio_encoder_outputs = audio_encoder_result.get();
    
    // Print some values from audio encoder outputs
    std::cout << "Audio encoder outputs size: " << audio_encoder_outputs.size() << std::endl;
    if (audio_encoder_outputs.size() > 0) {
      auto& output_tensor = audio_encoder_outputs[0].toTensor();
      std::cout << "Audio encoder output shape: ";
      for (int i = 0; i < output_tensor.dim(); ++i) {
        std::cout << output_tensor.size(i) << " ";
      }
      std::cout << std::endl;
      
      // Print first 10 values if tensor has data
      if (output_tensor.numel() > 0) {
        const float* data = output_tensor.const_data_ptr<float>();
        std::cout << "First 10 audio encoder output values: ";
        for (int i = 0; i < 10 && i < output_tensor.numel(); ++i) {
          std::cout << data[i] << " ";
        }
        std::cout << std::endl;
      }
    }
    
    encoder_output = audio_encoder_outputs[0];
  } else if (input.is_text()) {
    std::cout << "Prefilling text..." << std::endl;
    // For text input, we don't need to run the image encoder.
    // Instead, we run the text encoder to get the encoder output.
    auto& text = input.get_text();
    std::cout << "Prefilling: " << text << std::endl;
    std::vector<uint64_t> tokens =
        ET_UNWRAP_TOKENIZER(tokenizer_->encode(text));
    std::cout << "encoded tokens: ";
    for (auto token : tokens) {
      std::cout << token << ", ";
    }
    // TODO: since this is not the right tekken.json tokenizer, the special tokens are not getting
    // encoded properly, so the output of <s>[INST][BEGIN_AUDIO] is 9 tokens instead of 3, resulting it
    // [1, 9, 3072], not [1, 3, 3072].
    auto text_tensor = executorch::extension::from_blob(
        tokens.data(),
        {1, static_cast<aten::SizesType>(tokens.size())},
        ::executorch::aten::ScalarType::Long);

    // Print text_tensor values
    std::cout << "text_tensor shape: ";
    for (int i = 0; i < text_tensor->dim(); ++i) {
      std::cout << text_tensor->size(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "text_tensor values: ";
    const uint64_t* tensor_data = text_tensor->const_data_ptr<uint64_t>();
    for (int i = 0; i < text_tensor->numel(); ++i) {
      std::cout << tensor_data[i] << " ";
    }
    std::cout << std::endl;

    // Run token embedding
    auto token_embedding_outputs =
        ET_UNWRAP(module_->execute(kTokenEmbeddingMethod, text_tensor));

    // Print some values from token embedding outputs
    std::cout << "Token embedding outputs size: " << token_embedding_outputs.size() << std::endl;
    if (token_embedding_outputs.size() > 0) {
      auto& output_tensor = token_embedding_outputs[0].toTensor();
      std::cout << "Token embedding output shape: ";
      for (int i = 0; i < output_tensor.dim(); ++i) {
        std::cout << output_tensor.size(i) << " ";
      }
      std::cout << std::endl;
      std::cout << "Token embedding output dtype: " << static_cast<int>(output_tensor.scalar_type()) << std::endl;
      
      // Print first few values from each token if tensor has data
      if (output_tensor.numel() > 0) {
        if (output_tensor.scalar_type() == ::executorch::aten::ScalarType::Float) {
          const float* data = output_tensor.const_data_ptr<float>();
          std::cout << "Token embedding output values:" << std::endl;
          int batch_size = output_tensor.size(0);
          int seq_len = output_tensor.size(1);
          int hidden_dim = output_tensor.size(2);
          
          for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
              std::cout << "  Token " << s << " (first 10 values): ";
              for (int h = 0; h < 10 && h < hidden_dim; ++h) {
                int idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                std::cout << data[idx] << " ";
              }
              std::cout << std::endl;
            }
          }
        } else if (output_tensor.scalar_type() == ::executorch::aten::ScalarType::Half) {
          const ::executorch::aten::Half* data = output_tensor.const_data_ptr<::executorch::aten::Half>();
          std::cout << "Token embedding output values:" << std::endl;
          int batch_size = output_tensor.size(0);
          int seq_len = output_tensor.size(1);
          int hidden_dim = output_tensor.size(2);
          
          for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
              std::cout << "  Token " << s << " (first 10 values): ";
              for (int h = 0; h < 10 && h < hidden_dim; ++h) {
                int idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                std::cout << static_cast<float>(data[idx]) << " ";
              }
              std::cout << std::endl;
            }
          }
        } else {
          std::cout << "Token embedding output has unsupported dtype for printing values" << std::endl;
        }
      }
    }

    encoder_output = token_embedding_outputs[0];
  } else {
    std::cout << "unsupported input type" << std::endl;
    ET_LOG(Error, "Unsupported input type");
    // For all other input types (e.g., audio), return error
    return ::executorch::runtime::Error::NotSupported;
  }

  // run text model
  std::cout << "Preflling decoder..." << std::endl;
  // Make it so that cache_position goes from start_pos to start_pos + encoder_output.size(1). e.g. if start_pos = 2 and encoder_output.size(1) = 5, cache_position_tensor should be [2, 3, 4, 5, 6].
  int64_t seq_len = encoder_output.toTensor().size(1);
  std::vector<int64_t> cache_positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i) {
    cache_positions[i] = start_pos + i;
  }
  auto cache_position_tensor = ::executorch::extension::from_blob(
      cache_positions.data(), {seq_len}, executorch::aten::ScalarType::Long);
  auto prefill_result = module_->execute(kTextModelMethod, {cache_position_tensor, encoder_output});
  if (prefill_result.error() != ::executorch::runtime::Error::Ok) {
    std::cout << "Decoder execution failed with error: " << static_cast<int>(prefill_result.error()) << std::endl;
    return ::executorch::runtime::Error::Internal;
  }
  auto prefill_outputs = prefill_result.get();
  auto outputs_res = prefill_outputs[0].toTensor();

  // Print decoder output values
  std::cout << "Decoder output shape: ";
  for (int i = 0; i < outputs_res.dim(); ++i) {
    std::cout << outputs_res.size(i) << " ";
  }
  std::cout << std::endl;
  std::cout << "Decoder output dtype: " << static_cast<int>(outputs_res.scalar_type()) << std::endl;
  
  // Print first few values if tensor has data
  if (outputs_res.numel() > 0) {
    if (outputs_res.scalar_type() == ::executorch::aten::ScalarType::Float) {
      const float* data = outputs_res.const_data_ptr<float>();
      std::cout << "First 10 decoder output values: ";
      for (int i = 0; i < 10 && i < outputs_res.numel(); ++i) {
        std::cout << data[i] << " ";
      }
      std::cout << std::endl;
    } else if (outputs_res.scalar_type() == ::executorch::aten::ScalarType::Half) {
      const ::executorch::aten::Half* data = outputs_res.const_data_ptr<::executorch::aten::Half>();
      std::cout << "First 10 decoder output values: ";
      for (int i = 0; i < 10 && i < outputs_res.numel(); ++i) {
        std::cout << static_cast<float>(data[i]) << " ";
      }
      std::cout << std::endl;
    } else {
      std::cout << "Decoder output has unsupported dtype for printing values" << std::endl;
    }
  }

  

  // Update the start_pos, which is only available inside this function.
  // outputs_res can have only one logits.
  start_pos += encoder_output.toTensor().size(1);
  std::cout << "post prefill start_pos: " << start_pos << std::endl;

  return static_cast<uint64_t>(
      text_decoder_runner_->logits_to_token(outputs_res));
}

/**
 * Load the Module for encoder prefill purpose.
 * @return The error code.
 */
::executorch::runtime::Error MultimodalPrefiller::load() {
  if (is_method_loaded()) {
    return ::executorch::runtime::Error::Ok;
  }
  // token_embeddings and text_model have to show up in method names.
  std::cout << "Loading text embedding model" << std::endl;
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTokenEmbeddingMethod));
  std::cout << "Loading decoder model" << std::endl;
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));

  std::unordered_set<std::string> methods =
      ET_UNWRAP(module_->method_names(), "Failed to get method names");
  std::cout << "methods: ";
  for (const auto& method : methods) {
       std::cout << method << ", ";
   }
  std::cout << std::endl;
  

  // Load image_encoder method if exists.
  if (methods.find(kImageEncoderMethod) != methods.end()) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kImageEncoderMethod));
  }

  if (methods.find(kAudioEncoderMethod) != methods.end()) {
    std::cout << "Loading audio model" << std::endl;
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kAudioEncoderMethod));
    std::cout << "is_method_loaded(\"audio_encoder\"): " << module_->is_method_loaded(kAudioEncoderMethod) << std::endl;
  }

  return ::executorch::runtime::Error::Ok;
}

/**
 * Check if the required methods in the Module is loaded.
 * @return True if the Module is loaded, false otherwise.
 */
bool MultimodalPrefiller::is_method_loaded() {
  ::executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
      module_->method_names();
  if (!module_->is_method_loaded(kTokenEmbeddingMethod)) {
    return false;
  }
  if (!module_->is_method_loaded(kTextModelMethod)) {
    return false;
  }
  if (methods_res.error() != ::executorch::runtime::Error::Ok) {
    ET_CHECK_MSG(false, "Failed to get method names");
  }
  std::unordered_set<std::string> methods = methods_res.get();
  if (methods.find(kImageEncoderMethod) != methods.end()) {
    return module_->is_method_loaded(kImageEncoderMethod);
  }
  return true;
}

} // namespace executorch::extension::llm
